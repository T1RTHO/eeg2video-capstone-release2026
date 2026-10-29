"""Released-code-style classifier agreement on the 200 completed RGB videos."""
import argparse,csv,hashlib,json,sys,zipfile,math
from pathlib import Path
import numpy as np
MODELS={
 'image':('google/vit-base-patch16-224','3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3'),
 'video':('MCG-NJU/videomae-base-finetuned-kinetics','488eb9a0565f257b32866000305c8178965eb9f6')}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def scoring(pred,ref,n,seed):
 # Upstream's union of top-3 targets versus one shared set of n-1 distractors.
 pred=np.asarray(pred);ref=np.asarray(ref)
 if pred.ndim!=1 or pred.shape!=ref.shape or not np.isfinite(pred).all() or not np.isfinite(ref).all():raise ValueError('Invalid logits')
 target=np.argsort(ref,kind='stable')[-3:]
 pool=np.setdiff1d(np.arange(len(pred)),target)
 rng=np.random.default_rng(seed);wins=0;strict_wins=0;ties=0
 best=float(pred[target].max());top1=float(pred[int(np.argmax(ref))])
 for _ in range(100):
  chosen=rng.choice(pool,n-1,replace=False);other=float(pred[chosen].max())
  # Conservative ties: upstream argsort tie behavior is device dependent.
  wins+=int(best>other);strict_wins+=int(top1>other);ties+=int(best==other)
 return wins/100,strict_wins/100,ties

def validate(root,source):
 s=read(source/'evaluation/summary.json');cfg=read(source/'run_config.json')
 if s.get('status')!='complete' or s.get('completed_clips')!=200:raise ValueError('Need completed 200-clip generation')
 if cfg.get('conditioning')!='original_mlp' or cfg.get('protocol_version')!='final_paper_beta_released500_v1':raise ValueError('Unexpected generation protocol')
 with (root/'outputs/prepare_video_clips/paired_samples.csv').open(encoding='utf-8-sig',newline='') as f:allrows=list(csv.DictReader(f))
 if len(allrows)!=1400 or sha(root/'outputs/prepare_video_clips/paired_samples.csv')!=cfg['manifest_sha256']:raise ValueError('Manifest mismatch')
 rows=allrows[1200:];inputs=[]
 from tqdm import tqdm
 for i,r in enumerate(tqdm(rows,desc='Verify 200 RGB pairs',unit='clip')):
  expected=f'sub01_session01_block07_clip{i+1:03d}'
  if r['sample_id']!=expected or r['split']!='test':raise ValueError('Sample mapping mismatch')
  gp=source/'clips'/expected/'generated_rgb.npy';rp=root/r['rgb_array'].replace('\\','/');done=read(gp.parent/'complete.json')
  gh=sha(gp);rh=sha(rp)
  if gh!=done['generated_sha256'] or rh!=done['reference_sha256'] or done['ground_truth_conditioning'] is not False:raise ValueError('RGB evidence changed')
  for p in (gp,rp):
   a=np.load(p,mmap_mode='r',allow_pickle=False)
   if a.shape!=(6,288,512,3) or a.dtype!=np.uint8:raise ValueError('Invalid RGB '+str(p))
  inputs.append({'sample_id':expected,'generated':str(gp),'reference':str(rp),'generated_sha256':gh,'reference_sha256':rh})
 return inputs,cfg

def infer(kind,inputs,out,device,session_clips):
 import torch
 from tqdm import tqdm
 from transformers import ViTImageProcessor,ViTForImageClassification,VideoMAEImageProcessor,VideoMAEForVideoClassification
 folder=out/'logits'/kind;folder.mkdir(parents=True,exist_ok=True)
 todo=[]
 for i,item in enumerate(inputs):
  p=folder/(item['sample_id']+'.npz');meta=p.with_suffix('.json')
  if meta.exists():
   if not p.exists() or read(meta).get('sha256')!=sha(p):raise ValueError('Cached logits changed')
  else:todo.append(i)
 todo=todo[:session_clips]
 if not todo:print(kind+': all logits cached',flush=True);return
 model_id,revision=MODELS[kind]
 print(f'Loading {kind} classifier: {model_id}. First run downloads model weights with Hugging Face progress.',flush=True)
 if kind=='image':
  processor=ViTImageProcessor.from_pretrained(model_id,revision=revision)
  model,info=ViTForImageClassification.from_pretrained(model_id,revision=revision,torch_dtype=torch.float32,output_loading_info=True)
 else:
  processor=VideoMAEImageProcessor.from_pretrained(model_id,revision=revision)
  model,info=VideoMAEForVideoClassification.from_pretrained(model_id,revision=revision,num_frames=6,torch_dtype=torch.float32,output_loading_info=True)
 if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
  raise ValueError('Classifier loading mismatch; no random/unmatched weights allowed: '+str(info))
 expected_classes=1000 if kind=='image' else 400
 if model.config.num_labels!=expected_classes:raise ValueError('Wrong classifier label space')
 model.to(device).eval()
 write(out/(kind+'_classifier_config.json'),model.config.to_dict());write(out/(kind+'_processor_config.json'),processor.to_dict())
 with torch.inference_mode():
  for i in tqdm(todo,desc=kind+' classifier',unit='clip',dynamic_ncols=True):
   item=inputs[i];results={}
   for side in ['reference','generated']:
    a=np.load(item[side],allow_pickle=False)
    if kind=='image':
     # Six-frame batch fits laptop GPU; no gradient computation.
     x=processor(images=list(a),return_tensors='pt')
    else:x=processor(list(a),return_tensors='pt')
    logits=model(**{k:v.to(device) for k,v in x.items()}).logits.float().cpu().numpy()
    expected=(6,1000) if kind=='image' else (1,400)
    if logits.shape!=expected or not np.isfinite(logits).all():raise ValueError('Classifier output invalid')
    results[side]=logits
   p=folder/(item['sample_id']+'.npz');temp=p.with_suffix('.tmp')
   with temp.open('wb') as f:np.savez_compressed(f,**results)
   temp.replace(p);write(p.with_suffix('.json'),{'sha256':sha(p),'sample_id':item['sample_id']})
 del model
 if device=='cuda':torch.cuda.empty_cache()

def report(inputs,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from tqdm import tqdm
 results=[];labels=[]
 complete=all((out/'logits'/kind/(x['sample_id']+'.json')).exists() for kind in MODELS for x in inputs)
 if not complete:
  counts={k:sum((out/'logits'/k/(x['sample_id']+'.json')).exists() for x in inputs) for k in MODELS}
  write(out/'status.json',{'status':'partial','cached_clips':counts,'planned_per_classifier':200})
  print('Partial cache:',counts,'; repeat with --resume. Full-set scoring waits for all pairs.',flush=True);return
 for kind in MODELS:
  arrays=[];cfg=read(out/(kind+'_classifier_config.json'));names=cfg['id2label']
  for x in inputs:
   p=out/'logits'/kind/(x['sample_id']+'.npz')
   if sha(p)!=read(p.with_suffix('.json'))['sha256']:raise ValueError('Logit cache hash mismatch')
   with np.load(p,allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in ['reference','generated']})
  for i,x in enumerate(tqdm(inputs,desc='Score '+kind,unit='clip')):
   pred=arrays[i]['generated'];ref=arrays[i]['reference'];shuffled=arrays[(i+5)%200]['reference']
   # Fixed five-position cyclic shift: another source 10-second clip in the block.
   for f in range(len(pred)):
    labels.append({'sample_id':x['sample_id'],'classifier':kind,'frame':f+1 if kind=='image' else 0,
     'reference_top3':' | '.join(names[str(int(k))] for k in np.argsort(ref[f])[-3:][::-1]),
     'generated_top3':' | '.join(names[str(int(k))] for k in np.argsort(pred[f])[-3:][::-1])})
   for n in [2,40]:
    for pairing,target in [('matched',ref),('shifted_control',shuffled)]:
     scores=[scoring(pred[f],target[f],n,20261024+i*1000+f*10+n) for f in range(len(pred))]
     results.append({'sample_id':x['sample_id'],'classifier':kind,'n_way':n,'pairing':pairing,
      'top3_reference_agreement':float(np.mean([a[0] for a in scores])),
      'top1_reference_sensitivity':float(np.mean([a[1] for a in scores])),
      'tie_trials':int(sum(a[2] for a in scores))})
 for name,data in [('per_clip_identification.csv',results),('classifier_labels.csv',labels)]:
  with (out/name).open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 aggregates=[]
 for kind in MODELS:
  for n in [2,40]:
   def vals(pair):return np.array([r['top3_reference_agreement'] for r in results if r['classifier']==kind and r['n_way']==n and r['pairing']==pair])
   a=vals('matched');b=vals('shifted_control')
   aggregates.append({'classifier':kind,'n_way':n,'matched_mean':float(a.mean()),'shifted_control_mean':float(b.mean()),'matched_minus_control':float((a-b).mean()),'clips':200})
 summary={'status':'complete','clips':200,'frames':1200,'trials_per_frame_or_video':100,'top_k':1,'accepted_reference_labels':3,
 'aggregates':aggregates,'metric_name':'Released-code-style classifier agreement (documented adaptation)',
 'limitations':['Paper Appendix A describes CLIP image classifier; released code uses ViT. This implementation follows released ViT scoring.',
 'Official Google ViT and MCG-NJU VideoMAE checkpoints replace author-specific mirrors; weight identity with those mirrors is unverified.',
 'Uses lossless RGB rather than GIF, float32 rather than float16, fixed sampling seeds, and conservative tie handling.',
 'VideoMAE receives six frames as in released code; pretrained checkpoint default is 16 frames. Fixed sinusoidal positions are built for six frames; pretrained parameter shapes are checked.',
 'Reference top3 classifier predictions are pseudo-labels, not SEED-DV ground-truth categories. N-way distractors come from 1000 or 400 classifier labels, not the 40 dataset classes.',
 'Top3-any acceptance means usual 50% and 2.5% chance lines are inappropriate. Under exchangeable continuous random rankings its expectation is 3/(N+2), but actual classifier bias differs. Shifted control is empirical and may include related categories.',
 'Top1 sensitivity retains exclusion of all top3 reference labels when sampling distractors; it is supplementary, not a separate standard benchmark.',
 'One subject/session, inspected test block and one seed per clip. No population inference. Scores are not proof of faithful geometry or motion.']}
 write(out/'summary.json',summary);write(out/'status.json',{'status':'complete','cached_clips':{'image':200,'video':200}})
 fig,ax=plt.subplots(figsize=(9,5));x=np.arange(4);a=[r['matched_mean'] for r in aggregates];b=[r['shifted_control_mean'] for r in aggregates]
 ax.bar(x-.18,a,.36,label='Matched reference',color='#267b9b');ax.bar(x+.18,b,.36,label='Shifted reference control',color='#a5aeba')
 ax.set_xticks(x,[f'{r["classifier"].title()} {r["n_way"]}-way' for r in aggregates]);ax.set_ylim(0,1);ax.set_ylabel('Classifier agreement');ax.set_title('200 clips: reference top-3 acceptance');ax.legend();fig.tight_layout();fig.savefig(out/'identification_scores.png',dpi=180);plt.close(fig)
 import html
 body='<h1>Classifier agreement evaluation</h1><p>200 clips, original MLP. Documented adaptation of the released evaluation.</p><img style="width:100%" src="identification_scores.png"><table><tr><th>Classifier</th><th>N-way</th><th>Matched</th><th>Shifted control</th></tr>'
 for r in aggregates:body+=f'<tr><td>{r["classifier"]}</td><td>{r["n_way"]}</td><td>{r["matched_mean"]:.4f}</td><td>{r["shifted_control_mean"]:.4f}</td></tr>'
 body+='</table><h2>Interpretation limits</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in summary['limitations'])+'</ul>'
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>EEG2Video classifier agreement</title><style>body{font:16px system-ui;max-width:1000px;margin:auto;padding:30px}td,th{padding:10px;border:1px solid #ccc}</style>'+body,encoding='utf-8')
 archive=out/'identification_evidence.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file() and p!=archive and p.suffix in ['.json','.csv','.png','.html','.npz']:z.write(p,p.relative_to(out))
  for name in ['run_identification.py','PROTOCOL.md','upstream_metrics_reference.txt']:z.write(Path(__file__).with_name(name),'code/'+name)
 print('REPORT:',out/'index.html');print('UPLOAD:',archive)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--resume',action='store_true');p.add_argument('--session-clips',type=int,default=200);a=p.parse_args()
 if not 1<=a.session_clips<=200:p.error('session-clips must be 1..200 per classifier')
 import torch,transformers,importlib.metadata
 from tqdm import tqdm
 if a.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable. Check evaluation environment or use --device cpu.')
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 root=a.root.resolve();source=root/'outputs/final_evaluation/subject01_session01_block07/paper_beta_released500_v1'
 out=root/'outputs/final_evaluation/subject01_session01_block07/identification_v1'
 print('Stage 1/4: verify full-test outputs',flush=True);inputs,generation=validate(root,source)
 out.mkdir(parents=True,exist_ok=True)
 config={'version':1,'models':MODELS,'device':a.device,'dtype':'float32','frames':6,'trials':100,'seed':20261024,'shifted_control_offset':5,'script_sha256':sha(Path(__file__)),'generation_config_sha256':sha(source/'run_config.json'),'inputs':inputs,'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','Pillow']}}
 # JSON converts tuples to lists; canonicalize before resume equality.
 config=json.loads(json.dumps(config));cp=out/'run_config.json'
 if cp.exists():
  if not a.resume:raise ValueError('Output exists; use --resume')
  if read(cp)!=config:raise ValueError('Configuration/inputs changed; retain the original files and environment')
 write(cp,config)
 print('Stage 2/4: image classifier',flush=True);infer('image',inputs,out,a.device,a.session_clips)
 print('Stage 3/4: video classifier',flush=True);infer('video',inputs,out,a.device,a.session_clips)
 print('Stage 4/4: matched and shifted-reference scoring, report and evidence',flush=True);report(inputs,out)

if __name__=='__main__':main()
