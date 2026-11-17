"""Manifest-based FIRST clips: one center-frame sample per seven-frame clip."""
from __future__ import annotations

from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def prepared_root(data_root,dataset):
    base=dataset.split('@',1)[0]
    root=Path(data_root)
    if (root/'manifest.jsonl').is_file():return root
    candidate=root/base
    return candidate if (candidate/'manifest.jsonl').is_file() else None


class PreparedClipDataset(Dataset):
    def __init__(self,root,dataset,split,window_size=5,image_size=256,sequence=None):
        if window_size not in [1,3,5,7]:raise ValueError('Prepared FIRST uses 1,3,5,7 frames around frame 4')
        if image_size not in [None,256]:raise ValueError('Prepared FIRST must preserve its verified 256x256 crops')
        if split not in ['train','val','test','all']:raise ValueError('invalid split')
        self.root=Path(root);self.dataset_name=dataset;self.split=split;self.window_size=window_size
        source=dataset.split('@',1)[1] if '@' in dataset else None
        p=self.root/'manifest.jsonl' if split=='all' else self.root/'splits'/f'{split}.jsonl'
        self.records=[];self.sequence_records=defaultdict(list)
        parents={}
        with p.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if source and row['source_dataset']!=source:continue
                key=f"{row['source_dataset']}/{row['source_sequence']}/{row['clip_id']}"
                if sequence and sequence not in [key,row['source_sequence'],row['parent_sequence']]:continue
                if len(row['images'])!=7 or len(row['masks'])!=7 or row['center_frame']!=4:
                    raise ValueError('invalid seven-frame manifest row '+key)
                if split!='all' and row['split']!=split:raise ValueError('split manifest mismatch')
                parent=(row['source_dataset'],row['parent_sequence'])
                if parent in parents and parents[parent]!=row['split']:raise ValueError('parent split leakage')
                parents[parent]=row['split']
                row['_key']=key
                self.records.append(row);self.sequence_records[key].append(row)
        if not self.records:raise ValueError(f'No samples for dataset={dataset}, split={split}; WideIRSTD has training only')
        self.split_info=dict(source='verified explicit FIRST clip manifest; no per-frame re-splitting',source_filter=source)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        row=self.records[index];radius=self.window_size//2
        def read(relative,mask=False):
            p=(self.root/relative).resolve()
            if not p.is_relative_to(self.root.resolve()):raise ValueError('manifest path leaves dataset root')
            with Image.open(p) as im:a=np.array(im.convert('L'),dtype=np.float32)
            if a.shape!=(256,256):raise ValueError('prepared image dimensions changed')
            if mask:
                if not np.all((a==0)|(a==255)):raise ValueError('nonbinary prepared mask')
                return (a/255)[None]
            return (a/255)[None]
        frames=np.stack([read(p) for p in row['images'][3-radius:4+radius]])
        return dict(frames=torch.from_numpy(frames),image=torch.from_numpy(frames[radius].copy()),
                    mask=torch.from_numpy(read(row['masks'][3],True)),sequence=row['_key'],
                    name=Path(row['images'][3]).name,frame_index=4,original_size=(256,256),
                    source_dataset=row['source_dataset'],annotation_type=row['annotation_type'])

    def summary(self):
        return dict(dataset=self.dataset_name,root=str(self.root),split=self.split,samples=len(self),
                    source_counts=dict(Counter(r['source_dataset'] for r in self.records)),
                    annotation_counts=dict(Counter(r['annotation_type'] for r in self.records)),
                    sequences={k:len(v) for k,v in self.sequence_records.items()},
                    split_info=self.split_info,window_size=self.window_size,image_size=256,reference_frame=4)
