"""Export real center-frame predictions, pooled pixel curves and object metrics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, default_collate

from stsaunet.dataset import InfraredSequenceDataset
from stsaunet.evaluation import save_example
from stsaunet.metrics import MetricAccumulator
from stsaunet.model import STSAUNet
from stsaunet.prepared import prepared_root
from .curve_metrics import compute_curves, fixed_counts_summary, plot_curves, save_curves, write_json


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=dict(saved['model_config']);config.setdefault('unfolding_mode','framewise')
    model=STSAUNet(**config).to(device).eval()
    model.load_state_dict(saved['model_state'],strict=True)
    return saved,model


def evaluate_checkpoint(checkpoint,output_dir,data_root,dataset,split='val',
                        device='auto',batch_size=2,model_name='STSAUNet',source=None,annotation=None,
                        mode='exact',bins=4096,threshold=0.5,cache_dir=None,examples=False):
    start=time.perf_counter()
    dataset_root = prepared_root(data_root, dataset)
    if dataset_root is None:
        raise ValueError('Curve evaluation requires a prepared dataset with manifest.jsonl. Use evaluate.py for legacy layouts.')
    checkpoint=Path(checkpoint).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else torch.device(device)
    saved,model=load_model(checkpoint,device)
    ds=InfraredSequenceDataset(data_root,dataset,split,window_size=saved['model_config']['window_size'],image_size=256)
    sources=set(source.split(',')) if source else None
    annotations=set(annotation.split(',')) if annotation else None
    indices=[i for i,row in enumerate(ds.records) if (not sources or row['source_dataset'] in sources)
             and (not annotations or row['annotation_type'] in annotations)]
    if not indices:
        raise ValueError('No samples matched the requested split/source/annotation')
    cache=Path(cache_dir).resolve() if cache_dir else out/'cache'
    cache.mkdir(parents=True,exist_ok=True)
    expected=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),dataset=dataset,
                  data_root=str(Path(data_root).resolve()),split=split,indices=indices,
                  split_sha256=file_hash(ds.root/('manifest.jsonl' if split=='all' else f'splits/{split}.jsonl')),
                  window_size=saved['model_config']['window_size'],dtype='float32',activation='sigmoid(logits)')
    manifest_file=cache/'cache_metadata.json'
    reused=False
    if manifest_file.exists():
        metadata=json.loads(manifest_file.read_text(encoding='utf-8'))
        if metadata.get('identity') != expected or not metadata.get('complete'):
            raise ValueError(f'Cache identity differs or export incomplete: {cache}')
        reused=True
    else:
        for filename in ('probabilities.npy','targets.npy','prediction_index.json'):
            if (cache/filename).exists():
                raise FileExistsError(f'Incomplete cache must be preserved and a fresh directory used: {cache}')
        scores=np.lib.format.open_memmap(cache/'probabilities.npy',mode='w+',dtype=np.float32,shape=(len(indices),256,256))
        truth=np.lib.format.open_memmap(cache/'targets.npy',mode='w+',dtype=np.uint8,shape=scores.shape)
        loader=DataLoader(Subset(ds,indices),batch_size=batch_size,shuffle=False,num_workers=0)
        records=[];offset=0
        with torch.no_grad():
            for step,batch in enumerate(loader,1):
                prediction=model(batch['frames'].to(device))
                probability=torch.sigmoid(prediction['logits']).cpu().numpy()[:,0]
                if not np.isfinite(probability).all():
                    raise FloatingPointError('Nonfinite real predictions')
                count=len(probability)
                scores[offset:offset+count]=probability
                truth[offset:offset+count]=(batch['mask'].numpy()[:,0]>=.5).astype(np.uint8)
                for j in range(count):
                    original=indices[offset+j];row=ds.records[original]
                    records.append(dict(cache_index=offset+j,dataset_index=original,sequence=batch['sequence'][j],
                                        name=batch['name'][j],source=row['source_dataset'],annotation=row['annotation_type'],
                                        generated=row.get('generated',False),image=row['images'][3],mask=row['masks'][3],
                                        reference_frame=4))
                offset+=count
                if step%100==0 or step==len(loader):
                    print(f'{model_name} {split}: prediction {offset}/{len(indices)}',flush=True)
                del prediction
        assert offset==len(indices)
        scores.flush();truth.flush();del scores,truth
        write_json(cache/'prediction_index.json',records)
        write_json(manifest_file,dict(identity=expected,complete=True,samples=len(indices)))
    del model
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    probabilities=np.load(cache/'probabilities.npy',mmap_mode='r')
    targets=np.load(cache/'targets.npy',mmap_mode='r')
    records=json.loads((cache/'prediction_index.json').read_text(encoding='utf-8'))
    assert probabilities.shape==targets.shape==(len(records),256,256)
    grouping=defaultdict(list)
    for i,row in enumerate(records):
        for key in ('overall','source/'+row['source'],'annotation/'+row['annotation']):grouping[key].append(i)
    results={}
    for group,rows in grouping.items():
        metric=MetricAccumulator(threshold=threshold)
        for i in rows:metric.update(probabilities[i],targets[i])
        fixed=fixed_counts_summary(metric)
        scores=probabilities.reshape(-1) if group=='overall' else probabilities[rows].reshape(-1)
        labels=targets.reshape(-1) if group=='overall' else targets[rows].reshape(-1)
        summary,arrays=compute_curves(scores,labels,mode=mode,bins=bins)
        destination=out if group=='overall' else out/'groups'/group
        save_curves(destination,arrays)
        plot_curves([(model_name,arrays,summary)],destination/'roc','roc')
        plot_curves([(model_name,arrays,summary)],destination/'roc_low_fpr','roc',True)
        plot_curves([(model_name,arrays,summary)],destination/'pr','pr')
        results[group]=dict(samples=len(rows),fixed_threshold=threshold,**fixed,**summary)
        write_json(destination/'group_metrics.json',results[group])
        print(f'{model_name} {split} {group}: IoU={fixed["IoU"]:.6f} AP={summary["AP"]:.6f} ROC-AUC={summary["ROC_AUC"]:.6f}',flush=True)
        del scores,labels,arrays
    exported=[]
    if examples:
        counts=defaultdict(int)
        for i,row in enumerate(records):
            if counts[row['source']]>=2:continue
            counts[row['source']]+=1
            relative=f'{row["source"]}/sample-{counts[row["source"]]:02d}'
            destination=out/'examples'/relative
            if not destination.exists():
                batch=default_collate([ds[row['dataset_index']]])
                save_example(destination,batch,0,probabilities[i],threshold)
            exported.append(dict(path=relative,**row))
    result=dict(model_name=model_name,checkpoint=str(checkpoint),checkpoint_epoch=saved.get('epoch'),
                checkpoint_sha256=expected['checkpoint_sha256'],dataset=dataset,split=split,samples=len(records),
                groups=results,cache=str(cache),cache_reused=reused,curve_mode=mode,bins=bins if mode=='histogram' else None,
                score_definition='sigmoid(center-frame logits); no per-image rescaling; float32 before threshold',
                threshold=threshold,examples=exported,elapsed_seconds=time.perf_counter()-start,
                definitions=dict(ROC='pixel FPR=FP/(FP+TN), TPR=TP/(TP+FN)',
                                 AP='sklearn average_precision_score; no interpolation',
                                 PR_AUC='separately named trapezoidal PR area',
                                 PD='8-connected centroids, greedy one-to-one match within 3 pixels',
                                 FA='false-positive pixels / true background pixels',
                                 undefined='NaN and undefined_reason; whole-group absent classes are not silently scored',
                                 plots='only collinear horizontal ROC / vertical PR runs compressed; all corners retained; metric/NPZ/CSV points unchanged'),
                annotation_note='Declared annotation types are reported separately.')
    write_json(out/'metrics.json',result)
    write_json(out/'prediction_index.json',records)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--data-root',required=True,help='External prepared dataset directory or its parent.')
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--split',choices=('train','val','test','all'),default='val')
    parser.add_argument('--device',default='auto');parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--model-name',default='STSAUNet');parser.add_argument('--source');parser.add_argument('--annotation')
    parser.add_argument('--mode',choices=('exact','histogram'),default='exact');parser.add_argument('--bins',type=int,default=4096)
    parser.add_argument('--threshold',type=float,default=.5);parser.add_argument('--cache-dir');parser.add_argument('--examples',action='store_true')
    args=parser.parse_args();evaluate_checkpoint(**vars(args))


if __name__=='__main__':main()
