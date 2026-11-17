from __future__ import annotations

import argparse
import csv
from collections import Counter
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from stsaunet.dataset import InfraredSequenceDataset
from stsaunet.evaluation import evaluate_loader
from stsaunet.experiment import audit_prepared, check_unfolding, environment_info, sha256, write_json
from stsaunet.losses import STSAUNetLoss
from stsaunet.model import STSAUNet
from stsaunet.sampling import make_sampler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train STSAUNet with joint temporal-channel unfolding.',
        fromfile_prefix_chars='@',
        epilog='Read one flag or value per line from @configs/wide9000_50ep.args. Later flags override earlier values.',
    )
    parser.add_argument('--data-root', required=True, help='External dataset directory or its parent.')
    parser.add_argument('--dataset', default='NUDT-MIRSDT')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--boundary', choices=('replicate','reflect'), default='replicate')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--stages', type=int, default=6)
    parser.add_argument('--unfold-channels', type=int, default=32)
    parser.add_argument('--feature-channels', type=int, default=32)
    parser.add_argument('--max-offset', type=float, default=3.0)
    parser.add_argument('--unfolding-mode', choices=('joint_tc','framewise'), default='joint_tc')
    parser.add_argument('--reconstruction-weight', type=float, default=0.01)
    parser.add_argument('--alignment-weight', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--check-inputs', action='store_true', help='Full-size real prepared training sample audit and optimizer step.')
    parser.add_argument('--evaluate-at-end', action='store_true', help='Single grouped test pass using validation-selected best.pt.')
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--sampling', choices=('uniform','real_synthetic','source_balanced'), default='uniform')
    parser.add_argument('--segmentation-loss', choices=('baseline','hard_bce'), default='baseline')
    parser.add_argument('--hard-negative-ratio', type=float, default=3.0)
    parser.add_argument('--minimum-negatives', type=int, default=256)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return device


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if min(args.epochs,args.batch_size,args.stages,args.unfold_channels,args.feature_channels) < 1 or args.workers < 0:
        raise ValueError('Invalid positive training parameter')
    device = choose_device(args.device)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else Path('runs')/f'{args.dataset}-{stamp}'
    output_dir = output_dir.resolve()
    if output_dir.exists() and any((output_dir/name).exists() for name in ('status.json','config.json','train.log','best.pt','last.pt','history.csv')):
        raise FileExistsError(f'Run artifacts already exist: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    state = dict(status='preparing',pid=os.getpid(),output_dir=str(output_dir),completed_epochs=0,target_epochs=args.epochs)
    def status(**updates):
        state.update(updates)
        state['updated_at'] = datetime.now().astimezone().isoformat()
        write_json(output_dir/'status.json',state)
    def report(message):
        line = f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}'
        print(line, flush=True)
        with (output_dir/'train.log').open('a',encoding='utf-8') as stream:
            stream.write(line+'\n')
    status()
    try:
        report(f'Device: {device} | PID: {os.getpid()} | run: {output_dir}')
        write_json(output_dir/'environment.json',environment_info())
        # Reconstructable code snapshot for this exact run, independent of later edits.
        source_root = Path(__file__).resolve().parent
        source_paths = [source_root/'train.py',source_root/'evaluate.py',source_root/'infer.py',source_root/'verify_data.py',source_root/'README.md']
        source_paths += sorted((source_root/'stsaunet').rglob('*.py'))
        hashes = {}
        for source in source_paths:
            relative = source.relative_to(source_root)
            target = output_dir/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes())
            hashes[str(relative)] = sha256(source)
        write_json(output_dir/'source_sha256.json',hashes)
        data_root = Path(args.data_root)
        prepared_root = data_root if (data_root/'manifest.jsonl').is_file() else data_root/args.dataset.split('@',1)[0]
        if (prepared_root/'manifest.jsonl').is_file():
            audit = audit_prepared(prepared_root)
            write_json(output_dir/'data_audit.json',audit)
            report('Active manifest and split audit passed: '+json.dumps(audit['splits'],ensure_ascii=False))
        common_data = dict(data_root=args.data_root,dataset=args.dataset,window_size=args.window_size,
                           image_size=args.image_size if args.image_size>0 else None,boundary=args.boundary,
                           val_fraction=args.val_fraction,seed=args.seed)
        train_data = InfraredSequenceDataset(split='train',**common_data)
        val_data = InfraredSequenceDataset(split='val',**common_data)
        model_config = dict(window_size=args.window_size,stages=args.stages,unfold_channels=args.unfold_channels,
                            feature_channels=args.feature_channels,max_offset=args.max_offset,unfolding_mode=args.unfolding_mode)
        config = vars(args).copy()
        loss_config = dict(segmentation_mode=args.segmentation_loss,
                           hard_negative_ratio=args.hard_negative_ratio,minimum_negatives=args.minimum_negatives)
        config.update(model=model_config,device=str(device),train_summary=train_data.summary(),val_summary=val_data.summary(),
                      reconstruction_definition='0.5 * (center fused reconstruction MSE + all-frame final unfolded reconstruction MSE)',
                      loss_definition='BCEWithLogits + soft IoU + 0.01 reconstruction + 0.01 alignment at defaults',
                      fresh_initialization=True,threshold=0.5,precision='float32',command=[sys.executable,*sys.argv],
                      loss_config=loss_config,samples_per_epoch=len(train_data),
                      sampling_definition='uniform without replacement' if args.sampling=='uniform' else 'inverse-group-frequency with replacement; fixed len(dataset) draws')
        write_json(output_dir/'config.json',config)
        report(f'Train samples: {len(train_data)} | validation samples: {len(val_data)} | model: {model_config}')
        if args.check_inputs:
            status(status='checking_inputs')
            set_seed(args.seed)
            oom = False
            try:
                checked = check_unfolding(train_data,model_config,args.batch_size,device,args.lr,args.reconstruction_weight,args.alignment_weight,loss_config)
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:
                if args.batch_size == 1:
                    raise RuntimeError('Full configuration preflight OOM at batch_size=1')
                report('Preflight OOM: reduce batch size to 1; retain 7 frames, 256x256, stages and widths.')
                args.batch_size = 1
                config['batch_size'] = 1
                config['batch_size_fallback'] = 'CUDA OOM at requested batch_size=2 during full-config preflight'
                write_json(output_dir/'config.json',config)
                gc.collect()
                torch.cuda.empty_cache()
                set_seed(args.seed)
                checked = check_unfolding(train_data,model_config,1,device,args.lr,args.reconstruction_weight,args.alignment_weight,loss_config)
            write_json(output_dir/'input_check.json',checked)
            report('Input check passed: '+json.dumps({key:checked[key] for key in ('loader_shape','packed_shape','joint_center_target_delta','other_batch_sample_delta','losses','cuda_peak_allocated_gib','elapsed_seconds')}))
        # Preflight never changes the formal model initialization or sample shuffle.
        set_seed(args.seed)
        sampler = make_sampler(train_data,args.sampling,args.seed)
        train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler,
                                  num_workers=args.workers,pin_memory=device.type=='cuda')
        val_loader = DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
        model = STSAUNet(**model_config).to(device)
        criterion = STSAUNetLoss(args.reconstruction_weight,args.alignment_weight,**loss_config)
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.999))
        best_iou, best_epoch = -1.0, 0
        start = time.perf_counter()
        loss_keys = ['loss','loss_segmentation','loss_bce','loss_soft_iou','loss_reconstruction','loss_alignment']
        fields = ['epoch','train_loss','val_loss','val_iou','val_precision','val_recall','val_f1','lr','epoch_seconds','train_samples','val_samples']
        fields += [f'train_{key}' for key in loss_keys[1:]]+[f'val_{key}' for key in loss_keys[1:]]
        with (output_dir/'history.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            stream.flush()
            for epoch in range(1,args.epochs+1):
                model.train()
                epoch_start = time.perf_counter()
                train_sums = {key:0.0 for key in loss_keys}
                train_count = 0
                source_draws, annotation_draws, unique_sequences = Counter(), Counter(), set()
                status(status='training',epoch=epoch,step=0,steps_per_epoch=len(train_loader),samples_seen=0,batch_size=args.batch_size)
                for step,batch in enumerate(train_loader,1):
                    frames = batch['frames'].to(device,non_blocking=True)
                    image = batch['image'].to(device,non_blocking=True)
                    mask = batch['mask'].to(device,non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(frames)
                    losses = criterion(output,image,mask,frames)
                    if not torch.isfinite(losses['loss']):
                        raise FloatingPointError(f'Non-finite training loss at epoch={epoch}, step={step}')
                    losses['loss'].backward()
                    optimizer.step()
                    count = frames.shape[0]
                    for key in loss_keys:
                        train_sums[key] += float(losses[key].detach())*count
                    train_count += count
                    source_draws.update(batch.get('source_dataset', ['unspecified']*count))
                    annotation_draws.update(batch.get('annotation_type', ['pixel_mask']*count))
                    unique_sequences.update(batch['sequence'])
                    if step % args.progress_every == 0 or step == 1 or step == len(train_loader):
                        elapsed = time.perf_counter()-epoch_start
                        eta = elapsed/step*(len(train_loader)-step)
                        report(f'epoch {epoch:03d}/{args.epochs} step {step}/{len(train_loader)} | train_loss={train_sums["loss"]/train_count:.6f} | elapsed={elapsed:.1f}s train_eta={eta:.1f}s')
                        status(status='training',step=step,samples_seen=train_count,train_loss=train_sums['loss']/train_count,train_eta_seconds=eta)
                    del output,losses,frames,image,mask
                if train_count != len(train_data):
                    raise RuntimeError('Training epoch did not traverse the full split')
                write_json(output_dir/'sampling'/f'epoch-{epoch:03d}.json',dict(epoch=epoch,draws=train_count,
                           optimizer_steps=step,unique_clips=len(unique_sequences),sources=dict(source_draws),
                           annotations=dict(annotation_draws),policy=args.sampling))
                status(status='validating',samples_seen=train_count)
                validation = evaluate_loader(model,val_loader,device,criterion,compute_objects=False,report=report,progress_every=args.progress_every)
                metrics = validation['groups']['overall']
                improved = metrics['IoU']>best_iou
                if improved:
                    best_iou,best_epoch = metrics['IoU'],epoch
                checkpoint = dict(epoch=epoch,model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
                                  model_config=model_config,run_config=config,val_metrics=metrics,best_iou=best_iou,best_epoch=best_epoch,
                                  python_rng_state=random.getstate(),numpy_rng_state=np.random.get_state(),torch_rng_state=torch.get_rng_state(),
                                  cuda_rng_states=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
                save_checkpoint(output_dir/'last.pt',checkpoint)
                if improved:
                    save_checkpoint(output_dir/'best.pt',checkpoint)
                write_json(output_dir/'validation'/f'epoch-{epoch:03d}.json',validation)
                row = dict(epoch=epoch,train_loss=train_sums['loss']/train_count,val_loss=validation['losses']['loss'],
                           val_iou=metrics['IoU'],val_precision=metrics['Precision'],val_recall=metrics['Recall'],val_f1=metrics['F1'],
                           lr=optimizer.param_groups[0]['lr'],epoch_seconds=time.perf_counter()-epoch_start,
                           train_samples=train_count,val_samples=validation['samples'])
                row.update({f'train_{key}':train_sums[key]/train_count for key in loss_keys[1:]})
                row.update({f'val_{key}':validation['losses'][key] for key in loss_keys[1:]})
                writer.writerow(row)
                stream.flush()
                report(f'epoch {epoch:03d}/{args.epochs} complete | train_loss={row["train_loss"]:.6f} val_loss={row["val_loss"]:.6f} IoU={metrics["IoU"]:.6f} F1={metrics["F1"]:.6f} | best_epoch={best_epoch}')
                status(status='epoch_complete',completed_epochs=epoch,best_epoch=best_epoch,best_val_iou=best_iou,last_val_metrics=metrics)
        summary = dict(training_complete=True,completed_epochs=args.epochs,best_epoch=best_epoch,best_val_iou=best_iou,
                       best_checkpoint=str(output_dir/'best.pt'),last_checkpoint=str(output_dir/'last.pt'),training_seconds=time.perf_counter()-start)
        write_json(output_dir/'training_summary.json',summary)
        report(f'Training complete. Best validation IoU={best_iou:.6f} at epoch {best_epoch}')
        if args.evaluate_at_end:
            status(status='testing')
            del optimizer,checkpoint
            gc.collect()
            if device.type=='cuda': torch.cuda.empty_cache()
            checkpoint = torch.load(output_dir/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['model_state'])
            test_data = InfraredSequenceDataset(split='test',**common_data)
            test_loader = DataLoader(test_data,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
            testing = evaluate_loader(model,test_loader,device,criterion,compute_objects=True,
                                      examples_dir=output_dir/'examples',report=report,progress_every=args.progress_every)
            testing.update(checkpoint=str(output_dir/'best.pt'),selected_epoch=best_epoch,split='test',selection='best pooled validation IoU; fixed threshold 0.5; test evaluated once after training')
            write_json(output_dir/'test_metrics.json',testing)
            report('Test complete: '+json.dumps(testing['groups'],ensure_ascii=False))
            summary['test_metrics'] = testing['groups']
        write_json(output_dir/'summary.json',summary)
        status(status='completed',**summary)
    except BaseException as error:
        report(traceback.format_exc())
        status(status='failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
