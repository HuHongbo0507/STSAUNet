"""Operator-level inference accounting including functional convolutions/grid sampling."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from stsaunet.model import STSAUNet
from .curve_metrics import write_json


class OperationCounter(TorchDispatchMode):
    ignored={
        'aten.cat.default','aten.clone.default','aten.copy_.default','aten.expand.default',
        'aten.new_zeros.default','aten.select.int','aten.slice.Tensor','aten.stack.default',
        'aten.unsqueeze.default','aten.view.default','aten.zeros.default','aten.zeros_like.default',
        'aten.detach.default','aten.empty.memory_format','aten.empty_like.default','aten._to_copy.default',
        'aten.permute.default','aten.as_strided.default','aten.reshape.default','aten.contiguous.default'}
    elementwise={'aten.add.Tensor','aten.add.Scalar','aten.sub.Tensor','aten.sub.Scalar',
                 'aten.mul.Tensor','aten.mul.Scalar','aten.div.Tensor','aten.div.Scalar',
                 'aten.rsub.Scalar','aten.neg.default'}
    activations={'aten.relu_.default','aten.relu.default','aten.sigmoid.default','aten.tanh.default'}
    samplers={'aten.cudnn_grid_sampler.default','aten.grid_sampler_2d.default'}

    def __init__(self):
        super().__init__()
        self.calls=Counter();self.unsupported=Counter();self.totals=Counter()
        self.by_module=defaultdict(Counter);self.stack=[];self.ignored_calls=Counter()

    def add(self,category,value):
        self.totals[category]+=int(value)
        for path in self.stack:
            self.by_module[path][category]+=int(value)

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}));name=str(func);self.calls[name]+=1
        if name=='aten.convolution.default':
            weight=args[1]
            if args[6]:
                self.unsupported[name+'.transposed']+=1
            else:
                macs=result.numel()*weight.shape[1]*math.prod(weight.shape[2:])
                self.add('convolution_macs',macs)
                self.add('convolution_flops',2*macs)
                if args[2] is not None:self.add('bias_add_flops',result.numel())
        elif name in self.samplers:
            # Bilinear, normalized coordinates, zero padding, align_corners=True.
            if name=='aten.grid_sampler_2d.default' and (args[2]!=0 or args[3]!=0 or not args[4]):
                self.unsupported[name+'.nondefault_sampling']+=1
            else:
                n,c,h,w=result.shape
                # Coordinate transforms: 4 arithmetic ops/site. Four shared bilinear
                # weights: 8/site. Four multiplies + three additions/channel/site.
                self.add('grid_sample_estimated_flops',n*h*w*(12+7*c))
                self.add('grid_sample_output_values',result.numel())
        elif name in self.elementwise:
            self.add('elementwise_flops',result.numel())
        elif name in self.activations:
            # One abstract intrinsic/comparison per element, not hardware expansion.
            self.add('activation_intrinsic_proxy_ops',result.numel())
        elif name=='aten.linspace.default':
            self.add('grid_construction_estimated_flops',2*result.numel())
        elif name in self.ignored:
            self.ignored_calls[name]+=1
        else:
            self.unsupported[name]+=1
        return result


def profile_checkpoint(checkpoint,output,device='auto',shape=None):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=dict(saved['model_config']);config.setdefault('unfolding_mode','framewise')
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if shape is None:shape=(1,config.get('window_size',5),1,256,256)
    model=STSAUNet(**config).to(device).eval();model.load_state_dict(saved['model_state'],strict=True)
    x=torch.zeros(tuple(shape),device=device)
    if shape[0]!=1:raise ValueError('Comparative inference reporting requires batch=1')
    counter=OperationCounter();handles=[]
    for path,module in model.named_modules():
        label=path or 'model'
        def before(module,inputs,name=label):counter.stack.append(name)
        def after(module,inputs,output):counter.stack.pop()
        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    with torch.no_grad(),counter:prediction=model(x)
    for handle in handles:handle.remove()
    arithmetic=sum(counter.totals[key] for key in ('convolution_flops','bias_add_flops','elementwise_flops',
                                                   'grid_sample_estimated_flops','grid_construction_estimated_flops'))
    total=arithmetic+counter.totals['activation_intrinsic_proxy_ops']
    parameters=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    result=dict(checkpoint=str(Path(checkpoint).resolve()),model_config=config,input_shape=list(shape),
                output_shape=list(prediction['logits'].shape),torch_version=torch.__version__,device=device,
                total_params=parameters,trainable_params=trainable,params_M=parameters/1e6,trainable_params_M=trainable/1e6,
                module_params={name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()},
                module_params_M={name:sum(p.numel() for p in module.parameters())/1e6 for name,module in model.named_children()},
                convolution_MACs=counter.totals['convolution_macs'],GMACs=counter.totals['convolution_macs']/1e9,
                convolution_FLOPs=counter.totals['convolution_flops'],convolution_GFLOPs=counter.totals['convolution_flops']/1e9,
                extra_arithmetic_estimated_FLOPs=arithmetic-counter.totals['convolution_flops'],
                arithmetic_estimated_FLOPs=arithmetic,total_estimated_FLOPs_with_intrinsic_proxy=total,
                total_estimated_GFLOPs_with_intrinsic_proxy=total/1e9,categories=dict(counter.totals),
                operator_calls=dict(counter.calls),by_module_inclusive={k:dict(v) for k,v in counter.by_module.items()},
                unsupported_operators=dict(counter.unsupported),intentionally_ignored_operations=dict(counter.ignored_calls),
                covered_under_declared_rules=not counter.unsupported,estimated=True,
                convention=dict(multiply_add='1 MAC = 2 FLOPs; convolution bias adds counted separately',
                                grid_sample='N*Hout*Wout*(12+7*C): normalized coordinate/weight arithmetic plus bilinear dot product',
                                grid_sample_assumptions='bilinear, zero padding, align_corners=True; four neighbors assumed, including boundary sites',
                                nonlinear='ReLU comparison, sigmoid and tanh each counted as one intrinsic proxy per output; not hardware instruction FLOPs',
                                linspace='two arithmetic operations per coordinate; negligible approximate grid construction cost',
                                ignored='allocations, views, copies, indexing, memory traffic, integer/floor/boundary logic; not floating arithmetic',
                                scope='full model forward, both segmentation and auxiliary reconstruction heads; one clip -> one center prediction',
                                exclusions='data loading, sigmoid applied by external evaluator, loss, backward, optimizer; clip count is not divided by frame count'),
                caution='Declared analytical operation estimate, not an exact hardware instruction count. Unsupported arithmetic would make this partial.')
    write_json(output,result)
    print(f'Params={parameters} | convolution={result["GMACs"]:.9f} GMACs | declared total={total/1e9:.9f} GFLOP-equivalent estimate | unsupported={dict(counter.unsupported)}',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--shape',nargs=5,type=int,metavar=('B','T','C','H','W'),
                        help='Defaults to one grayscale clip at 256x256 with T from the checkpoint.')
    args=parser.parse_args();profile_checkpoint(args.checkpoint,args.output,args.device,args.shape)


if __name__=='__main__':main()
