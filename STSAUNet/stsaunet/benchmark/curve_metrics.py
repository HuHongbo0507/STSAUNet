"""Real pixel ROC/PR curves. No generated classifier data or per-image scaling."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True), encoding='utf-8')
    temporary.replace(path)


def fixed_counts_summary(metric):
    result=metric.compute()
    result['TN']=float(metric.background_pixels-metric.fp)
    result['FPR']=metric.fp/metric.background_pixels if metric.background_pixels else float('nan')
    positives=metric.tp+metric.fn
    result['TPR']=metric.tp/positives if positives else float('nan')
    return result


def compute_curves(probabilities, target, mode='exact', bins=4096):
    scores = np.asarray(probabilities).reshape(-1)
    truth = np.asarray(target).reshape(-1)
    if len(scores) != len(truth) or not len(scores):
        raise ValueError('Nonempty score/label arrays of equal size are required')
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise ValueError('Expected finite sigmoid probabilities in [0,1]')
    if not np.all((truth == 0) | (truth == 1)):
        raise ValueError('Labels must be binary')
    positive = int(np.count_nonzero(truth))
    negative = len(truth)-positive
    reason = None
    if not positive:
        reason = 'no positive pixels: TPR, ROC-AUC and AP undefined'
    elif not negative:
        reason = 'no negative pixels: FPR and ROC-AUC undefined'
    if mode == 'exact' and positive and negative:
        fpr, tpr, thresholds = roc_curve(truth, scores, drop_intermediate=False)
        precision, recall, pr_thresholds = precision_recall_curve(truth, scores, drop_intermediate=False)
        roc_auc = float(auc(fpr,tpr))
        ap = float(average_precision_score(truth,scores))
    elif mode == 'exact':
        order = np.argsort(scores,kind='stable')[::-1]
        ordered = scores[order]
        stops = np.r_[np.flatnonzero(np.diff(ordered)),len(order)-1]
        tp = np.r_[0,np.cumsum(truth[order],dtype=np.int64)[stops]]
        fp = np.r_[0,stops+1]-tp
        thresholds = np.r_[np.inf,ordered[stops]]
        fpr = fp/negative if negative else np.full(len(tp),np.nan)
        tpr = tp/positive if positive else np.full(len(tp),np.nan)
        p = np.divide(tp,tp+fp,out=np.ones(len(tp),dtype=float),where=(tp+fp)>0)
        precision = np.r_[p[1:][::-1],1.0]
        recall = np.r_[tpr[1:][::-1],0.0]
        pr_thresholds = thresholds[1:][::-1]
        roc_auc,ap = float('nan'),(1.0 if positive else float('nan'))
    elif mode == 'histogram':
        if bins < 2:
            raise ValueError('At least two histogram bins required')
        pos_hist,neg_hist = np.zeros(bins,np.int64),np.zeros(bins,np.int64)
        for start in range(0,len(scores),1_000_000):
            p = scores[start:start+1_000_000]
            y = truth[start:start+1_000_000].astype(bool)
            pos_hist += np.histogram(p[y],bins=bins,range=(0,1))[0]
            neg_hist += np.histogram(p[~y],bins=bins,range=(0,1))[0]
        tp,fp = np.r_[0,np.cumsum(pos_hist[::-1])],np.r_[0,np.cumsum(neg_hist[::-1])]
        thresholds = np.r_[np.inf,np.linspace(0,1,bins+1)[:-1][::-1]]
        tpr = tp/positive if positive else np.full(len(tp),np.nan)
        fpr = fp/negative if negative else np.full(len(tp),np.nan)
        p = np.divide(tp,tp+fp,out=np.ones(len(tp),float),where=tp+fp>0)
        precision,recall = np.r_[p[1:][::-1],1.0],np.r_[tpr[1:][::-1],0.0]
        pr_thresholds = thresholds[1:][::-1]
        roc_auc = float(auc(fpr,tpr)) if positive and negative else float('nan')
        ap = float(np.sum(np.diff(tpr)*p[1:])) if positive else float('nan')
    else:
        raise ValueError('mode must be exact or histogram')
    low = {}
    for budget in (1e-4,1e-3):
        candidates = np.flatnonzero(np.isfinite(fpr) & np.isfinite(tpr) & (fpr <= budget))
        if len(candidates):
            index = int(candidates[np.argmax(tpr[candidates])])
            threshold = float(thresholds[index])
            low[str(budget)] = dict(tpr=float(tpr[index]),actual_fpr=float(fpr[index]),
                                   threshold=threshold if np.isfinite(threshold) else 'inf')
        else:
            low[str(budget)] = dict(tpr=float('nan'),actual_fpr=float('nan'),threshold=None)
    arrays = dict(threshold=thresholds,fpr=fpr,tpr=tpr,pr_threshold=pr_thresholds,
                  precision=precision,recall=recall)
    summary = dict(ROC_AUC=roc_auc,AP=ap,positive_pixels=positive,negative_pixels=negative,
                   undefined_reason=reason,low_fpr=low,mode=mode,
                   threshold_resolution='all distinct observed float32 scores' if mode=='exact' else 1/bins,
                   low_fpr_rule='maximum observed TPR with FPR <= budget; no interpolation; earliest point breaks ties',
                   pr_auc_trapezoid=float(auc(recall,precision)) if positive else float('nan'))
    return summary,arrays


def save_curves(directory, arrays):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'curves.npz',**arrays)
    np.savetxt(directory/'roc.csv',np.column_stack([arrays[key] for key in ('threshold','fpr','tpr')]),
               delimiter=',',header='threshold,fpr,tpr',comments='',fmt='%.17g')
    # PR's final (precision=1, recall=0) has no associated score threshold.
    np.savetxt(directory/'pr.csv',np.column_stack([np.r_[arrays['pr_threshold'],np.nan],arrays['precision'],arrays['recall']]),
               delimiter=',',header='threshold,precision,recall',comments='',fmt='%.17g')


def plot_curves(entries, output, kind='roc', low_fpr=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig,axis=plt.subplots(figsize=(7,6),layout='constrained')
    for label,arrays,summary in entries:
        x,y=(arrays['fpr'],arrays['tpr']) if kind=='roc' else (arrays['recall'],arrays['precision'])
        visible=np.flatnonzero(np.isfinite(x)&np.isfinite(y))
        if low_fpr:
            visible=visible[x[visible]<=1e-3]
        # Keep all corners, removing only exactly collinear horizontal ROC /
        # vertical PR runs. Uniform index decimation would miss rare positives.
        if len(visible)>2:
            key=y[visible] if kind=='roc' else x[visible]
            changes=np.flatnonzero(np.diff(key)!=0)
            corners=np.unique(np.r_[0,len(visible)-1,changes,changes+1])
            visible=visible[corners]
        score=summary['ROC_AUC' if kind=='roc' else 'AP']
        axis.plot(x[visible],y[visible],linewidth=1.5,label=f'{label} ({score:.4f})')
    if kind=='roc' and not low_fpr:
        axis.plot([0,1],[0,1],'--',color='gray',linewidth=.8)
    # PR random baseline is class prevalence, not the descending diagonal.
    if kind=='pr' and entries:
        summary=entries[0][2]; p=summary['positive_pixels'];n=summary['negative_pixels']
        axis.axhline(p/(p+n),color='gray',linestyle='--',linewidth=.8,label='Positive prevalence')
    axis.set(xlabel='FPR' if kind=='roc' else 'Recall',ylabel='TPR' if kind=='roc' else 'Precision',
             xlim=(0,1e-3 if low_fpr else 1),ylim=(0,1),
             title=('ROC (low FPR)' if low_fpr else 'ROC curve') if kind=='roc' else 'Precision–Recall curve')
    axis.grid(alpha=.25);axis.legend(fontsize=8,loc='best')
    for extension in ('png','svg'):
        fig.savefig(output.with_suffix('.'+extension),dpi=250)
    plt.close(fig)
