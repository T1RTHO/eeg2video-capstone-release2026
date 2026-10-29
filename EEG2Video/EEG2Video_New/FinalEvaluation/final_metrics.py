"""Descriptive full-test metrics. Does not implement paper classifier-based identification."""
import csv,json,zipfile,hashlib,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from tqdm import tqdm
from final_common import read_json,write_json,sha,manifest


def measure(ref,gen):
    for a in (ref,gen):
        if a.shape!=(6,288,512,3) or a.dtype!=np.uint8:raise ValueError('Expected six uint8 RGB frames at 288x512')
    r=ref.astype(np.float64)/255.;g=gen.astype(np.float64)/255.
    mse=float(np.mean((r-g)**2))
    ssim=float(np.mean([structural_similarity(ref[i],gen[i],channel_axis=-1,data_range=255,win_size=7,gaussian_weights=False,use_sample_covariance=True) for i in range(6)]))
    dr=np.diff(r,axis=0);dg=np.diff(g,axis=0)
    return {'rgb_mse_01':mse,'psnr_db':float(-10*np.log10(mse)) if mse>0 else None,
        'exact_pixel_match':bool(mse==0),'ssim_rgb_7x7':ssim,
        'reference_adjacent_mae_255':float(np.abs(dr).mean()*255),
        'generated_adjacent_mae_255':float(np.abs(dg).mean()*255),
        'temporal_difference_mae_255':float(np.abs(dg-dr).mean()*255),
        'unique_generated_frames':len({hashlib.sha256(f.tobytes()).hexdigest() for f in gen})}


def evaluate(root,out):
    print('Evaluating completed clips on CPU; reference RGB is used only for evaluation.',flush=True)
    root=Path(root);out=Path(out);rows=manifest(root)[1200:];records=[];details=[]
    for i,r in enumerate(tqdm(rows,desc='Evaluate test clips',unit='clip')):
        folder=out/'clips'/r['sample_id'];done=folder/'complete.json'
        if not done.exists():continue
        d=read_json(done);gp=folder/'generated_rgb.npy';rp=root/r['rgb_array'].replace('\\','/')
        if d.get('sample_id')!=r['sample_id'] or d.get('ground_truth_conditioning') is not False:raise ValueError('Invalid completion record')
        if sha(gp)!=d['generated_sha256']:raise ValueError('Generated RGB changed')
        if sha(rp)!=d['reference_sha256']:raise ValueError('Reference RGB changed')
        ref=np.load(rp,allow_pickle=False);gen=np.load(gp,allow_pickle=False)
        records.append({'sample_id':r['sample_id'],'test_index':i,**measure(ref,gen)})
        details.append(d)
    if not records:raise ValueError('No complete clips to evaluate')
    report=out/'evaluation';report.mkdir(exist_ok=True)
    with (report/'per_clip_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    metric_names=['rgb_mse_01','psnr_db','ssim_rgb_7x7','reference_adjacent_mae_255','generated_adjacent_mae_255','temporal_difference_mae_255']
    aggregates={}
    for k in metric_names:
        values=[r[k] for r in records if r[k] is not None]
        aggregates[k]={'mean':float(np.mean(values)) if values else None,'median':float(np.median(values)) if values else None,'valid_clips':len(values)}
    summary={'status':'complete' if len(records)==200 else 'partial','completed_clips':len(records),'planned_clips':200,
        'missing_test_indices':[i for i in range(200) if i not in {r['test_index'] for r in records}],
        'aggregates':aggregates,'exact_match_clips':sum(r['exact_pixel_match'] for r in records),
        'exactly_static_generated_clips':sum(r['unique_generated_frames']==1 for r in records),
        'generation_seconds_sum':sum(d['seconds'] for d in details),
        'metric_definitions':{'psnr_db':'From pooled RGB MSE over all six frames per clip; exact matches have infinite PSNR, stored as null and separately counted.',
        'ssim_rgb_7x7':'scikit-image SSIM, uint8 RGB, data_range255, uniform 7x7 window, sample covariance; mean of six corresponding frame scores.',
        'temporal_difference_mae_255':'MAE between signed adjacent frame differences of generated and reference RGB. Sensitive to spatial mismatch and appearance; not motion accuracy.'},
        'limitations':['Single subject/session and Block7 only; not the paper multi-session aggregate.',
        'Block7 has been inspected previously; not a pristine blind test.',
        'Reported SSIM settings are explicit; numerical equivalence to released evaluation is not established.',
        'Paper image/video classifier identification metrics are pending; this is not the complete paper metric suite.',
        'Only one fixed seed per clip; no best-of-many selection or uncertainty across subjects.',
        'DANA and beta provenance is documented in protocol_notes.md; prior scheduler_first diagnostic outputs remain separate.']}
    write_json(report/'summary.json',summary);write_json(report/'completion_records.json',details)
    import importlib.metadata,sys
    write_json(report/'evaluation_environment.json',{'python':sys.version,'packages':{k:importlib.metadata.version(k) for k in ['numpy','scikit-image','matplotlib']},'evaluator_sha256':sha(Path(__file__))})
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,k,label in zip(axes.flat[:3],['ssim_rgb_7x7','psnr_db','rgb_mse_01'],['RGB SSIM (7x7)','PSNR (dB)','RGB MSE (0–1)']):
        v=[r[k] for r in records if r[k] is not None];ax.hist(v,bins=min(20,max(1,len(v))),color='#267b9b');ax.set_xlabel(label);ax.set_ylabel('Clips')
    ax=axes.flat[3];x=[r['reference_adjacent_mae_255'] for r in records];y=[r['generated_adjacent_mae_255'] for r in records]
    ax.scatter(x,y,s=18,alpha=.7);lim=max(x+y+[1]);ax.plot([0,lim],[0,lim],'--',color='gray');ax.set_xlabel('Reference frame change (0–255)');ax.set_ylabel('Generated frame change (0–255)')
    fig.suptitle(f'Single-subject evaluation: {len(records)}/200 clips ({summary["status"]})');fig.tight_layout();fig.savefig(report/'metric_overview.png',dpi=180);plt.close(fig)
    body=f'<h1>EEG2Video: {len(records)}/200 test clips</h1><p>Status: {summary["status"]}. Original MLP, paper beta mapping, released500 DANA, 100 DDIM steps.</p><img src="metric_overview.png"><p><a href="../index.html">All generated comparisons</a> | <a href="per_clip_metrics.csv">Per-clip CSV</a></p><table><tr><th>Metric</th><th>Mean</th><th>Median</th></tr>'
    for k,v in aggregates.items():body+=f'<tr><td>{k}</td><td>{v["mean"]}</td><td>{v["median"]}</td></tr>'
    body+='</table><h2>Limits of interpretation</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in summary['limitations'])+'</ul>'
    body+='<h2>Fixed examples (clips 1, 51, 101, 151, 200)</h2>'
    fixed=[]
    for i in [0,50,100,150,199]:
        sid=rows[i]['sample_id'];f=out/'clips'/sid/'comparison.png'
        if f.exists():
            body+=f'<h3>{sid}</h3><img src="../clips/{sid}/comparison.png">';fixed.append(f)
    (report/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>EEG2Video evaluation</title><style>body{font:16px system-ui;max-width:1150px;margin:auto;padding:30px;color:#173247}img{width:100%}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style>'+body,encoding='utf-8')
    # Portable evidence includes evaluation and the linked local gallery assets.
    archive=out/'final_evaluation_evidence.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(report.iterdir()):
            if f.is_file():z.write(f,'evaluation/'+f.name)
        z.write(out/'run_config.json','run_config.json')
        # Fixed examples only: all 200 numbers remain in CSV. No outcome-based curation.
        cards=[]
        for f in fixed:
            for name in ['comparison.png','comparison.gif','complete.json']:
                item=f.parent/name
                if item.exists():z.write(item,item.relative_to(out))
            sid=f.parent.name;cards.append(f'<h2>{sid}</h2><img style="width:100%" src="clips/{sid}/comparison.png">')
        z.writestr('index.html','<!doctype html><meta charset="utf-8"><h1>Fixed example subset</h1><p>All available clip metrics are in evaluation/per_clip_metrics.csv. This portable subset includes predetermined examples only.</p>'+''.join(cards))
        for name in ['protocol_notes.md','run_final_generation.py','final_metrics.py','final_common.py','final_media.py']:
            f=Path(__file__).parent/name
            if f.exists():z.write(f,'code/'+name)
    print('REPORT:',report/'index.html');print('UPLOAD:',archive)
    print(f'Completed {len(records)}/200. '+('Generation complete; paper classifier metrics remain to be evaluated.' if len(records)==200 else 'Resume the same command to continue.'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();evaluate(a.root,a.out or a.root/'outputs/final_evaluation/subject01_session01_block07/paper_beta_released500_v1')
