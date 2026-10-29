"""Subject 1 GLMNet + Seq2Seq, paper-oriented implementation.
Run from D:/Capstone/work in eeg2video. Outputs: outputs/<script stem>/.
Source: XuanhaoLiu/EEG2Video, commit 18af805753bc89e796625cdaaca1d0844bee22a8.
GLMNet: EEG-VP/models.py shallownet + glfnet, adapted to T=100, output=512.
Branch width 256 follows the repository glfnet example, not a specified
reconstruction hyperparameter in the paper. Encoder flatten width is computed
from convolution/pooling geometry (the released T//200 formula fails at T=100).
Transformer norm_first=True follows paper Eq.1; hidden-token autoregression
and other unspecified Transformer settings follow the released Seq2Seq code.
No decoder positional encoding is invented. Zero seed output is excluded.
Fixed 200 epochs on Blocks 1-6; Block 7 EEG is predicted only after training.
Micro-batch BatchNorm differs from true batch 32. Single subject and existing
mean-VAE targets remain documented adaptations. This is not an exact reproduction.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SOURCE_COMMIT = '18af805753bc89e796625cdaaca1d0844bee22a8'
LOCAL_CHANNELS = ['PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

class ShallowBranch(nn.Module):
    def __init__(self, channels, samples=100, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,40,(1,25)), nn.Conv2d(40,40,(channels,1)),
            nn.BatchNorm2d(40), nn.ELU(), nn.AvgPool2d((1,51),(1,5)), nn.Dropout(.5))
        pooled = (samples - 25 + 1 - 51)//5 + 1
        if pooled < 1: raise ValueError('EEG window is too short.')
        self.out = nn.Linear(40*pooled,out_dim)
    def forward(self,x): return self.out(self.net(x).flatten(1))

class GLMNetEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.globalnet = ShallowBranch(62)
        self.occipital_localnet = ShallowBranch(12)
        self.out = nn.Linear(512,512)
    def forward(self,x):
        return self.out(torch.cat((self.globalnet(x),self.occipital_localnet(x[:,:,50:62,:])),dim=1))

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        p=torch.arange(7).unsqueeze(1); d=torch.exp(torch.arange(0,512,2)*(-math.log(10000.)/512))
        pe=torch.zeros(1,7,512); pe[0,:,0::2]=torch.sin(p*d); pe[0,:,1::2]=torch.cos(p*d)
        self.register_buffer('pe',pe)
    def forward(self,x): return x+self.pe[:,:x.shape[1]]


class myTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_embedding=GLMNetEmbedding()
        self.positional_encoding=PositionalEncoding()
        self.transformer_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(512,4,batch_first=True,norm_first=True),2)
        self.transformer_decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(512,4,batch_first=True,norm_first=True),4)
        self.predictor=nn.Linear(512,4*36*64)
    def forward(self,x):
        b=x.shape[0]
        memory=self.transformer_encoder(self.positional_encoding(self.eeg_embedding(x.reshape(b*7,1,62,100)).reshape(b,7,512)))
        tokens=memory.new_zeros(b,1,512)
        for _ in range(6):
            n=tokens.shape[1]
            mask=torch.triu(torch.ones(n,n,device=x.device,dtype=torch.bool),diagonal=1)
            decoded=self.transformer_decoder(tokens,memory,tgt_mask=mask)
            tokens=torch.cat([tokens,decoded[:,-1:]],dim=1)
        # Official code selects [:,:-1], including the EEG-independent zero seed.
        # Select all SIX generated tokens instead. No target video is a model input.
        return self.predictor(tokens[:,1:]).reshape(b,6,4,36,64)


def windows(a):
    if a.ndim!=3 or a.shape[1:]!=(62,400): raise ValueError(f'Wrong EEG shape: {a.shape}')
    return np.stack([a[:,:,s:s+100] for s in range(0,301,50)],axis=1)


def prepare_eeg(a,mean=None,std=None):
    x=windows(a).astype(np.float64)
    if not np.isfinite(x).all(): raise ValueError('Non-finite EEG.')
    if mean is None:
        mean=x.mean(axis=0); std=x.std(axis=0); std=np.where(std<1e-12,1.,std)
    x=((x-mean)/std).astype(np.float32)
    return torch.from_numpy(x),mean,std


def save_json(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2),encoding='utf-8'); tmp.replace(path)


def load_checkpoint(path):
    # Only load checkpoints created locally by this script.
    try: return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError: return torch.load(path,map_location='cpu')


def predict(model,x,device,micro):
    model.eval(); values=[]
    with torch.no_grad():
        for part in x.split(micro): values.append(model(part.to(device)).cpu())
    return torch.cat(values)


def smoke_test(device):
    torch.manual_seed(17)
    model=myTransformer().to(device)
    x=torch.randn(2,7,62,100,device=device)
    model.eval()
    with torch.no_grad():
        a=model(x); b=model(x.flip(0))
        assert a.shape==(2,6,4,36,64) and torch.isfinite(a).all()
        if torch.allclose(a[0,0],b[0,0]): raise RuntimeError('First frame is independent of EEG.')
    model.train(); model(x).square().mean().backward()
    for name in ['globalnet','occipital_localnet']:
        grad=getattr(model.eeg_embedding,name).net[0].weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum()==0:
            raise RuntimeError(f'Missing/invalid gradient in {name}.')
    print('PASS: output shape, finite values, first-frame EEG dependence, gradients in BOTH GLMNet branches.',flush=True)
    del model,x,a,b
    if device.type=='cuda': torch.cuda.empty_cache()


def manifest(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def fingerprint(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def save_plots(out,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    ax.plot([h['epoch'] for h in history],[h['train_mse'] for h in history],label='Training (dropout active)')
    ax.set(xlabel='Epoch',ylabel='Latent MSE',title='GLMNet + Seq2Seq: Blocks 1-6')
    ax.legend(); ax.grid(alpha=.2); fig.savefig(out/'training_curve.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    ax.set(xlim=(0,10),ylim=(0,6)); ax.axis('off')
    nodes={'input':(5,5.5,'Each EEG window\n62 channels x 100 samples'),
           'global':(2.5,4,'Global ShallowNet\nAll 62 channels -> 256 features'),
           'local':(7.5,4,'Local ShallowNet\nChannels 50-61 -> 256 features'),
           'fuse':(5,2.6,'Concatenate + linear layer\n512-dimensional EEG embedding'),
           'seq':(5,1.3,'7 embeddings + position encoding\nTransformer: 2 encoder / 4 decoder layers'),
           'output':(5,.2,'6 predicted frame latents: 6 x 4 x 36 x 64')}
    for x,y,label in nodes.values():
        ax.text(x,y,label,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.4',fc='#e8f1fb',ec='#365a80'))
    for s,t in [('input','global'),('input','local'),('global','fuse'),('local','fuse'),('fuse','seq'),('seq','output')]:
        x,y,_=nodes[s]; xx,yy,_=nodes[t]
        ax.annotate('',xy=(xx,yy+.37),xytext=(x,y-.37),arrowprops=dict(arrowstyle='->',color='#365a80'))
    fig.savefig(out/'model_architecture.png',dpi=160); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--micro-batch',type=int,default=4,help='Memory batch; optimizer batch is 32. BatchNorm sees this smaller batch.')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--smoke-test',action='store_true',help='Check forward/backward only, without loading dataset or training.')
    args=ap.parse_args()
    if not 1<=args.micro_batch<=32: raise ValueError('Micro-batch must be between 1 and 32.')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in this environment.')
    device=torch.device(args.device)
    print('Checking GLMNet and Seq2Seq before training...',flush=True)
    smoke_test(device)
    if args.smoke_test: return
    root=args.root.resolve(); out=root/'outputs'/Path(__file__).stem
    ed=root/'outputs'/'segment_raw_signals_200Hz'; td=root/'outputs'/'extract_training_targets'
    if json.loads((td/'summary.json').read_text(encoding='utf-8')).get('status')!='complete':
        raise ValueError('Target extraction is incomplete.')
    er=manifest(ed/'sample_mapping.csv'); tr=manifest(td/'sample_mapping.csv')
    if len(er)!=1400 or len(tr)!=1400: raise ValueError('Wrong manifest length.')
    for i,(e,t) in enumerate(zip(er,tr)):
        if any(e[k]!=t[k] for k in ['sample_id','split','split_index','block','clip_in_block']):
            raise ValueError('EEG/target manifest mismatch.')
        if e['sample_id']!=f'sub01_session01_block{i//200+1:02d}_clip{i%200+1:03d}':
            raise ValueError('Unexpected sample order.')
        expected_split='train' if i<1200 else 'test'
        if e['split']!=expected_split or int(e['split_index'])!=(i if i<1200 else i-1200):
            raise ValueError('Unexpected split or row index.')
    locs=root/'SEED-DV'/'channel_62_pos.locs'
    channels=[line.split()[-1] for line in locs.read_text().splitlines() if line.strip()]
    if len(channels)!=62 or channels[50:62]!=LOCAL_CHANNELS:
        raise ValueError('Channel order differs from released GLMNet. Do not train until resolved.')
    a=np.load(ed/'train_eeg.npy',mmap_mode='r',allow_pickle=False)
    y=np.load(td/'train_visual_latents.npy',mmap_mode='r',allow_pickle=False)
    if a.shape!=(1200,62,400) or y.shape!=(1200,6,4,36,64): raise ValueError('Wrong training shapes.')
    if not np.isfinite(y).all(): raise ValueError('Non-finite target latents.')
    print('Checking input fingerprints...',flush=True)
    cfg=dict(architecture_version='glmnet_v1',source_commit=SOURCE_COMMIT,seed=42,epochs=200,
             cosine_horizon_epochs=200,train_samples=1200,training_blocks=[1,2,3,4,5,6],test_block=7,
             micro_batch=args.micro_batch,effective_batch=32,learning_rate=.0005,
             branch_width=256,embedding_dim=512,norm_first=True,
             local_channel_indices=list(range(50,62)),local_channel_names=LOCAL_CHANNELS,
             source_files={str(p.relative_to(root)):fingerprint(p) for p in
                [ed/'train_eeg.npy',td/'train_visual_latents.npy',td/'sample_mapping.csv',td/'summary.json',locs]},
             script_sha256=fingerprint(Path(__file__)))
    if (out/'run_config.json').exists():
        if json.loads((out/'run_config.json').read_text())!=cfg: raise ValueError('Run settings/data changed. Use a new output folder.')
        if (out/'summary.json').exists(): print(f'Already complete: {out}'); return
    elif out.exists() and any(out.iterdir()): raise ValueError('Output folder is nonempty without run metadata.')
    out.mkdir(parents=True,exist_ok=True); save_json(out/'run_config.json',cfg)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if device.type=='cuda': torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    x,mean,std=prepare_eeg(a)
    np.savez(out/'eeg_normalisation.npz',mean=mean,std=std)
    targets=torch.from_numpy(np.array(y,dtype=np.float32))
    np.save(out/'training_mean_visual_latents.npy',targets.mean(dim=0).numpy())
    model=myTransformer().to(device)
    optim=torch.optim.Adam(model.parameters(),lr=.0005)
    loader=DataLoader(TensorDataset(x,targets),batch_size=32,shuffle=True,num_workers=0)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=200*len(loader))
    latest=out/'latest_checkpoint.pt'; history=[]; start=0
    if latest.exists():
        ck=load_checkpoint(latest)
        if ck['config']!=cfg: raise ValueError('Checkpoint configuration mismatch.')
        model.load_state_dict(ck['model']); optim.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        for values in optim.state.values():
            for k,v in values.items():
                if torch.is_tensor(v) and k!='step': values[k]=v.to(device)
        start=ck['epoch']; history=ck['history']; torch.set_rng_state(ck['torch_rng'])
        if device.type=='cuda' and ck['cuda_rng'] is not None: torch.cuda.set_rng_state_all(ck['cuda_rng'])
        print(f'Resuming after epoch {start}.',flush=True)
    save_plots(out,history)
    print(f'GLMNet: 1200 training samples; 200 epochs; optimizer batch 32; micro-batch {args.micro_batch}.',flush=True)
    print('No selection/refit phase. Block 7 targets are never loaded. BatchNorm uses micro-batch statistics.',flush=True)
    wall=time.perf_counter()
    for ep in range(start,200):
        model.train(); train_sum=0.; count=0
        for bi,(bx,by) in enumerate(loader,1):
            optim.zero_grad(set_to_none=True)
            for j in range(0,len(bx),args.micro_batch):
                xx=bx[j:j+args.micro_batch].to(device); yy=by[j:j+args.micro_batch].to(device)
                loss=(model(xx)-yy).square().mean()
                if not torch.isfinite(loss): raise RuntimeError('Non-finite loss.')
                (loss*len(xx)/len(bx)).backward(); train_sum+=loss.item()*len(xx); count+=len(xx)
            optim.step(); scheduler.step()
            completed=(ep-start)*len(loader)+bi; total=(200-start)*len(loader)
            elapsed=time.perf_counter()-wall
            print(f'\rEpoch {ep+1}/200 batch {bi}/{len(loader)} | train MSE {train_sum/count:.6f} | ETA {elapsed/completed*(total-completed)/60:.1f} min',end='',flush=True)
        history.append(dict(epoch=ep+1,train_mse=train_sum/count,learning_rate=optim.param_groups[0]['lr']))
        ck=dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),
                config=cfg,epoch=ep+1,history=history,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        torch.save(ck,out/'latest_checkpoint.tmp'); (out/'latest_checkpoint.tmp').replace(latest)
        with (out/'training_log.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
        print('',flush=True)
        if (ep+1)%10==0: save_plots(out,history)
    torch.save(dict(model=model.state_dict(),epoch=200,config=cfg),out/'final_model.pt')
    test=np.load(ed/'test_eeg.npy',mmap_mode='r',allow_pickle=False)
    if test.shape!=(200,62,400): raise ValueError('Wrong test EEG shape.')
    tx,_,_=prepare_eeg(test,mean,std)
    pred=predict(model,tx,device,args.micro_batch).numpy()
    if pred.shape!=(200,6,4,36,64) or not np.isfinite(pred).all(): raise ValueError('Invalid predictions.')
    np.save(out/'block7_predicted_visual_latents.npy',pred)
    with (out/'prediction_mapping.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['prediction_index','sample_id','block','clip_in_block'])
        w.writeheader()
        for i,r in enumerate(er[1200:]): w.writerow(dict(prediction_index=i,**{k:r[k] for k in ['sample_id','block','clip_in_block']}))
    save_plots(out,history)
    summary=dict(cfg,status='complete',final_train_mse=history[-1]['train_mse'],
        final_learning_rate=history[-1]['learning_rate'],prediction_shape=list(pred.shape),
        adjacent_frame_latent_mse=np.mean(np.diff(pred.astype(np.float64),axis=1)**2,axis=(0,2,3,4)).tolist(),
        test_accuracy='Not measured: no test targets loaded.',
        paper_alignment=['GLMNet global/local ShallowNet branches; 512-dimensional embeddings.',
                        '7 windows of 100 samples, stride 50; Transformer 2 encoder / 4 decoder layers.',
                        'Pre-normalization follows paper Eq.1.',
                        'MSE; Adam 0.0005; full 200-epoch cosine schedule; Blocks 1-6 train, Block 7 test.'],
        documented_choices=['Branch width 256 follows repository example; paper reconstruction width unspecified.',
                            'ShallowNet flatten size computed for 100 samples, replacing broken T//200 expression.',
                            'Hidden-token autoregression, four heads, FFN 2048, dropout 0.1 retained from released Seq2Seq.',
                            'Zero starting token excluded from six predicted outputs.',
                            'Micro-batch BatchNorm differs from batch 32; accumulated gradients do not remove this difference.',
                            'Single subject only; existing deterministic mean-VAE targets and nominal frame sampling retained.',
                            'Coordinatewise standardization fitted on all 1200 training samples only.',
                            'GLMNet integration is reconstructed from paper and released classification encoder; not a released GLMNet Seq2Seq checkpoint.'])
    save_json(out/'summary.json',summary)
    print(f'Complete: {out}',flush=True)


if __name__=='__main__': main()
