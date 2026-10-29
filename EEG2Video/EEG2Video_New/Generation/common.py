import csv
import hashlib
import json
from pathlib import Path
import numpy as np

COMMIT = '18af805753bc89e796625cdaaca1d0844bee22a8'

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write_json(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8'); tmp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def manifest(root):
    rr = rows(root/'outputs/prepare_video_clips/paired_samples.csv')
    if len(rr) != 1400 or len({r['sample_id'] for r in rr}) != 1400:
        raise ValueError('Expected 1400 unique paired samples')
    for i,r in enumerate(rr):
        if int(r['block']) != i//200+1 or r['split'] != ('train' if i<1200 else 'test') or int(r['split_index']) != i%1200:
            raise ValueError('Unexpected manifest order/split')
    return rr

def targets(root):
    p=root/'outputs/extract_training_targets'
    s=read_json(p/'summary.json')
    if s['status']!='complete': raise ValueError('Target extraction is incomplete')
    return p,s

def base_model(root, full=False):
    from huggingface_hub import snapshot_download
    _,s=targets(root)
    original=Path(s['resolved_model'])
    needed=['vae','text_encoder','tokenizer'] + (['unet','scheduler'] if full else [])
    if original.is_dir() and all((original/k/'config.json').is_file() if k in ['vae','text_encoder','unet'] else (original/k).is_dir() for k in needed):
        return original
    revision=s.get('resolved_revision')
    if not revision or revision=='local directory':
        raise ValueError('Original local model unavailable: restore '+str(original))
    return Path(snapshot_download(s['model_source'], revision=revision,
        allow_patterns=[k+'/*' for k in needed]+['model_index.json'],
        ignore_patterns=['*.safetensors','*.fp16.*','*.non_ema.*','*.onnx','*.msgpack','*.h5']))

def predictions(root):
    specs={'visual':('my_autoregressive_transformer_glmnet','block7_predicted_visual_latents.npy',(200,6,4,36,64)),
           'semantic':('eeg_text/full_window_400fft','block7_predicted_text_embeddings.npy',(200,77,768)),
           'dynamic':('train_dynamic_glmnet','block7_predicted_dynamic_labels.npy',(200,))}
    expected=[r['sample_id'] for r in manifest(root)[1200:]]; arrays={}; hashes={}
    for key,(folder,name,shape) in specs.items():
        p=root/'outputs'/folder/name
        a=np.load(p,mmap_mode='r',allow_pickle=False)
        if a.shape!=shape or not np.isfinite(a).all(): raise ValueError('Invalid '+str(p))
        rr=rows(p.parent/'prediction_mapping.csv')
        if [r['sample_id'] for r in rr]!=expected or [int(r['prediction_index']) for r in rr]!=list(range(200)):
            raise ValueError('Sample mapping mismatch: '+str(p))
        arrays[key]=a; hashes[key]=sha(p)
    if not np.isin(arrays['dynamic'],[0,1]).all(): raise ValueError('Dynamic labels must be 0=slow,1=fast')
    return arrays,hashes

def dana(z, label, alpha, same, diverse):
    # Numpy reference, also used in tests: z is F,C,H,W.
    if label not in (0,1) or not 0<=alpha<=1: raise ValueError('Invalid DANA parameters')
    beta=0.2 if label==1 else 0.3
    return np.sqrt(alpha)*z + np.sqrt(1-alpha)*(np.sqrt(beta)*same+np.sqrt(1-beta)*diverse)

def save_csv(p, rr):
    if not rr: return
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
