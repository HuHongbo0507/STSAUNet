"""Plot real PR exports and AP; no synthetic classification data."""
import argparse,json
from pathlib import Path
import numpy as np
from .curve_metrics import plot_curves

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 entries=[];identity=None
 for value in a.inputs:
  directory=Path(value);meta=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
  index=json.loads((directory/'prediction_index.json').read_text(encoding='utf-8'))
  current=[(x['sequence'],x['name'],x['mask']) for x in index]
  if identity is not None and current!=identity:raise ValueError('Methods must use identical samples and order')
  identity=current
  entries.append((meta['model_name'],dict(np.load(directory/'curves.npz')),meta['groups']['overall']))
 plot_curves(entries,a.output,'pr')
if __name__=='__main__':main()
