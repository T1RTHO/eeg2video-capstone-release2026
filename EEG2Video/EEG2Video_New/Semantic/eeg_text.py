"""Train the full released EEG2Video semantic MLP on CPU.
Run from D:/Capstone/work in eeg2video. Requires torch>=2.1, numpy, matplotlib.
Default: full_window_400fft DE features, explicitly correcting source truncation.
Optional --feature-variant repository_200fft preserves the original extraction.
The feature choice is fixed for a run and recorded in every checkpoint.
200 epochs, batch 32, Adam 5e-4, full cosine schedule, Blocks 1-6 training.
Per-clip frozen CLIP text targets, with sample order verified against manifests.
Block 7 targets are NEVER loaded. Block 7 EEG features are predicted after training.
Four hidden ReLU layers + output layer (five Linear layers), exactly as released.
Paper describes four layers; the counting/architecture discrepancy is recorded.
Source commit: 18af805753bc89e796625cdaaca1d0844bee22a8, XuanhaoLiu/EEG2Video,
EEG2Video/EEG2Video_New/Semantic/eeg_text.py.
"""
import argparse
import csv
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

GIB=1024**3
EPOCHS=200
EXPECTED_PARAMETERS=894559136

class CLIP(nn.Module):

    def __init__(self, width=10000):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(310, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 77 * 768))

    def forward(self, eeg):
        return self.mlp(eeg)

def memory_status():
    """Process resident/peak memory and total/available physical system RAM."""
    if os.name == 'nt':
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD)] + [(name, ctypes.c_ulonglong) for name in ['ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile', 'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual']]

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage']]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(ms)
        pc = PROCESS_MEMORY_COUNTERS_EX()
        pc.cb = ctypes.sizeof(pc)
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(ms)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(pc), pc.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(total_gib=ms.ullTotalPhys / GIB, available_gib=ms.ullAvailPhys / GIB, process_rss_gib=pc.WorkingSetSize / GIB, process_peak_rss_gib=pc.PeakWorkingSetSize / GIB, process_private_gib=pc.PrivateUsage / GIB)

    def fields(path):
        values = {}
        for line in Path(path).read_text().splitlines():
            key, _, value = line.partition(':')
            if value.strip().endswith('kB'):
                values[key] = float(value.split()[0]) * 1024 / GIB
        return values
    m = fields('/proc/meminfo')
    p = fields('/proc/self/status')
    return dict(total_gib=m['MemTotal'], available_gib=m['MemAvailable'], process_rss_gib=p['VmRSS'], process_peak_rss_gib=p['VmHWM'], process_private_gib=None)

def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def check_rows(*collections):
    for rows in collections:
        if len(rows)!=1400:raise ValueError('Expected 1400 manifest rows.')
        for i,r in enumerate(rows):
            expected=f'sub01_session01_block{i//200+1:02d}_clip{i%200+1:03d}'
            if (r['sample_id']!=expected or r['split']!=('train' if i<1200 else 'test')
                or int(r['split_index'])!=(i if i<1200 else i-1200)):
                raise ValueError(f'Manifest order/split mismatch at row {i}.')
    for items in zip(*collections):
        for other in items[1:]:
            if any(items[0][key]!=other[key] for key in ['sample_id','split','split_index','block','clip_in_block']):
                raise ValueError('Feature and text sample mappings differ.')


def fit_normalisation(a):
    x=np.asarray(a,dtype=np.float64)
    mean=x.mean(axis=0);std=x.std(axis=0);std=np.where(std<1e-12,1.,std)
    return mean,std


def apply_normalisation(a,mean,std):
    x=((np.asarray(a,dtype=np.float64)-mean)/std).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Non-finite normalised features.')
    return torch.from_numpy(x)


def save_checkpoint(path,model,optim,scheduler,epoch,history,cfg):
    # No tensor cloning: state_dict refers to existing CPU storage.
    # Atomic replacement keeps the prior complete checkpoint until save succeeds.
    needed=sum(p.numel()*p.element_size() for p in model.parameters())*3+256*1024**2
    if shutil.disk_usage(path.parent).free<needed:
        raise RuntimeError('Insufficient free disk space for an atomic checkpoint (~10 GiB for full model).')
    state=dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),
               epoch=epoch,history=history,config=cfg,torch_rng=torch.get_rng_state())
    tmp=path.with_suffix('.tmp')
    try:
        torch.save(state,tmp);tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def initialise(cfg,latest,steps_per_epoch):
    if latest.exists():
        print('Loading checkpoint (~10 GiB); please wait...',flush=True)
        # Only load this script's locally created checkpoint. Do not use mmap:
        # its file mapping can prevent atomic replacement on Windows.
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['config']!=cfg:raise ValueError('Checkpoint configuration differs from this run.')
        # Construct on meta, then adopt loaded weights to avoid a second model copy.
        with torch.device('meta'):model=CLIP()
        model.load_state_dict(ck['model'],assign=True)
        optim=torch.optim.Adam(model.parameters(),lr=.0005,foreach=False)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=EPOCHS*steps_per_epoch)
        optim.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        start=ck['epoch'];history=ck['history'];torch.set_rng_state(ck['torch_rng'])
        del ck;gc.collect()
        print(f'Resuming after completed epoch {start}.',flush=True)
    else:
        torch.manual_seed(42);model=CLIP()
        optim=torch.optim.Adam(model.parameters(),lr=.0005,foreach=False)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=EPOCHS*steps_per_epoch)
        start=0;history=[]
    if sum(p.numel() for p in model.parameters())!=EXPECTED_PARAMETERS:raise ValueError('Unexpected architecture.')
    return model,optim,scheduler,start,history


def train_epoch(model,optim,scheduler,loader,epoch,progress=None):
    model.train();total=0.;count=0
    for bi,(x,y) in enumerate(loader,1):
        optim.zero_grad(set_to_none=True)
        prediction=model(x)
        loss=(prediction-y).square().mean()
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {bi}.')
        loss.backward();optim.step();scheduler.step()
        total+=loss.item()*len(x);count+=len(x)
        if progress:progress(bi,total/count)
    optim.zero_grad(set_to_none=True)  # Release ~3.33 GiB before saving/plotting.
    return total/count


def save_log(out,history):
    with (out/'training_log.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)


def plot(out,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    ax.plot([r['epoch'] for r in history],[r['train_mse'] for r in history],color='#147D92',label='Training MSE')
    ax.set(xlabel='Epoch',ylabel='Text embedding MSE',title='Semantic predictor training on Blocks 1-6')
    ax.grid(alpha=.2);ax.legend();fig.savefig(out/'training_curve.png',dpi=160);plt.close(fig)
    if not (out/'model_architecture.png').exists():
        fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True);ax.axis('off')
        labels=['DE features from 62 channels x 5 bands: 310 values',
                'Standardise using training mean and standard deviation',
                'Four hidden layers: 10000 units each, ReLU activation',
                'Linear output: 59136 values reshaped to 77 x 768',
                'MSE against frozen CLIP caption embeddings during training']
        for i,label in enumerate(labels):
            y=.9-i*.19
            ax.text(.5,y,label,ha='center',va='center',fontsize=11,
                    bbox=dict(boxstyle='round,pad=.5',fc='#E8EFF3',ec='#147D92'))
            if i<4:ax.annotate('',xy=(.5,y-.13),xytext=(.5,y-.055),arrowprops=dict(arrowstyle='->'))
        fig.savefig(out/'model_architecture.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--feature-variant',choices=['full_window_400fft','repository_200fft'],default='full_window_400fft')
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--checkpoint-every',type=int,default=5,help='Save resumable state every N completed epochs; default 5.')
    args=ap.parse_args()
    if args.threads<1 or args.checkpoint_every<1:raise ValueError('Threads/checkpoint interval must be positive.')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    root=args.root.resolve();out=root/'outputs'/Path(__file__).stem/args.feature_variant
    features=root/'outputs'/'extract_DE_PSD_features_1per2s';fd=features/args.feature_variant
    targets=root/'outputs'/'extract_training_targets';segments=root/'outputs'/'segment_raw_signals_200Hz'
    for p in [features/'summary.json',fd/'train_de_features.npy',fd/'test_de_features.npy']:
        if not p.exists():raise FileNotFoundError('First run: python EEG_preprocessing\\extract_DE_PSD_features_1per2s.py\nMissing: '+str(p))
    for folder in [features,targets]:
        if json.loads((folder/'summary.json').read_text(encoding='utf-8')).get('status')!='complete':
            raise ValueError(f'Incomplete preprocessing: {folder}')
    er=read_rows(segments/'sample_mapping.csv');fr=read_rows(features/'sample_mapping.csv');tr=read_rows(targets/'sample_mapping.csv')
    check_rows(er,fr,tr)
    a=np.load(fd/'train_de_features.npy',allow_pickle=False)
    y=np.load(targets/'train_text_embeddings.npy',mmap_mode='r',allow_pickle=False)
    if a.shape!=(1200,310) or y.shape!=(1200,77,768):raise ValueError(f'Unexpected shapes: {a.shape}, {y.shape}')
    if not np.isfinite(a).all() or not np.isfinite(y).all():raise ValueError('Non-finite training data.')
    print('Checking paired training data and fingerprints...',flush=True)
    tracked=[fd/'train_de_features.npy',features/'sample_mapping.csv',features/'summary.json',
             targets/'train_text_embeddings.npy',targets/'sample_mapping.csv',targets/'summary.json']
    cfg=dict(architecture_version='released_semantic_mlp_v1',architecture=[310,10000,10000,10000,10000,59136],
        parameters=EXPECTED_PARAMETERS,epochs=EPOCHS,cosine_horizon_epochs=EPOCHS,batch_size=32,
        seed=42,device='cpu',dtype='float32',threads=args.threads,optimizer='Adam',learning_rate=.0005,
        foreach=False,feature_variant=args.feature_variant,train_samples=1200,train_blocks=[1,2,3,4,5,6],test_block=7,
        data_hashes={str(p.relative_to(root)):sha256(p) for p in tracked},script_sha256=sha256(Path(__file__)),
        source_commit='18af805753bc89e796625cdaaca1d0844bee22a8')
    if (out/'run_config.json').exists():
        if json.loads((out/'run_config.json').read_text())!=cfg:raise ValueError('Run configuration/data changed; use a separate results folder.')
        if (out/'summary.json').exists():print(f'Already complete: {out}');return
    elif out.exists() and any(out.iterdir()):raise ValueError('Nonempty output folder without run metadata.')
    before=memory_status()
    print(f'Available RAM: {before["available_gib"]:.1f} GiB. Feature variant: {args.feature_variant}',flush=True)
    if before['available_gib']<20:raise RuntimeError('Need 20 GiB available RAM before starting. Close unused applications, then rerun.')
    latest=out/'latest_checkpoint.pt'
    required=12*GIB if latest.exists() else 24*GIB
    if shutil.disk_usage(root).free<required:raise RuntimeError(f'Need at least {required/GIB:.0f} GiB free disk for checkpoints and final weights.')
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'run_config.json',cfg)
    mean,std=fit_normalisation(a);x=apply_normalisation(a,mean,std)
    np.savez(out/'eeg_normalisation.npz',mean=mean,std=std)
    # About 271 MiB; model/optimizer dominate RAM. No test caption targets loaded.
    text=torch.from_numpy(np.array(y,dtype=np.float32).reshape(1200,59136))
    np.save(out/'training_mean_text_embedding.npy',text.mean(dim=0).reshape(77,768).numpy())
    loader=DataLoader(TensorDataset(x,text),batch_size=32,shuffle=True,num_workers=0)
    model,optim,scheduler,start,history=initialise(cfg,latest,len(loader))
    plot(out,history)
    print(f'CPU semantic training: {EPOCHS} epochs, full model, batch 32.\n'
          f'Checkpoint every {args.checkpoint_every} epochs (~10 GiB each, replacing the previous checkpoint).\n'
          'Interrupted runs resume from the last saved epoch. Test targets are never loaded.',flush=True)
    wall=time.perf_counter();saved_epoch=start
    try:
        for ep in range(start,EPOCHS):
            ep_start=time.perf_counter()
            def progress(bi,mse):
                done=(ep-start)*len(loader)+bi;total=(EPOCHS-start)*len(loader)
                elapsed=time.perf_counter()-wall
                print(f'\rEpoch {ep+1}/{EPOCHS} batch {bi}/{len(loader)} | train MSE {mse:.6f} | ETA {elapsed/done*(total-done)/3600:.2f} h',end='',flush=True)
            mse=train_epoch(model,optim,scheduler,loader,ep+1,progress)
            ms=memory_status()
            history.append(dict(epoch=ep+1,train_mse=mse,learning_rate=optim.param_groups[0]['lr'],
                                epoch_compute_seconds=time.perf_counter()-ep_start,
                                process_peak_rss_gib=ms['process_peak_rss_gib'],available_ram_gib=ms['available_gib']))
            save_log(out,history);print('',flush=True)
            if (ep+1)%args.checkpoint_every==0 or ep+1==EPOCHS:
                print('Saving resumable checkpoint; please wait...',flush=True)
                save_checkpoint(latest,model,optim,scheduler,ep+1,history,cfg);saved_epoch=ep+1
                plot(out,history)
                print(f'Checkpoint saved after epoch {saved_epoch}.',flush=True)
    except KeyboardInterrupt:
        print(f'\nStopped. Rerun the same command to resume after saved epoch {saved_epoch}.',flush=True)
        return
    # Discard optimizer moments before final export/inference to free RAM.
    del optim,scheduler,loader;gc.collect()
    print('Saving final model (~3.33 GiB)...',flush=True)
    final_tmp=out/'final_model.tmp'
    torch.save(dict(state_dict=model.state_dict(),config=cfg,epoch=EPOCHS),final_tmp)
    final_tmp.replace(out/'final_model.pt')
    test=np.load(fd/'test_de_features.npy',allow_pickle=False)
    if test.shape!=(200,310):raise ValueError('Wrong test DE shape.')
    tx=apply_normalisation(test,mean,std);model.eval()
    output=np.lib.format.open_memmap(out/'block7_predicted_text_embeddings.npy',mode='w+',dtype=np.float32,shape=(200,77,768))
    with torch.no_grad():
        for pos in range(0,200,32):
            pred=model(tx[pos:pos+32]).reshape(-1,77,768)
            if not torch.isfinite(pred).all():raise RuntimeError('Non-finite test prediction.')
            output[pos:pos+len(pred)]=pred.numpy()
    output.flush();del output
    with (out/'prediction_mapping.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['prediction_index','sample_id','block','clip_in_block']);w.writeheader()
        for i,r in enumerate(er[1200:]):w.writerow(dict(prediction_index=i,**{k:r[k] for k in ['sample_id','block','clip_in_block']}))
    plot(out,history)
    atomic_json(out/'summary.json',dict(status='complete',config=cfg,
        final_training_mse=history[-1]['train_mse'],final_learning_rate=history[-1]['learning_rate'],
        prediction_shape=[200,77,768],test_metrics='Not evaluated; test caption targets not loaded.',
        checkpoint_every_epochs=args.checkpoint_every,normalisation='Coordinate-wise training-only mean/std; zero-variance dimensions use std 1.',
        documented_choices=['CPU float32 with Adam foreach=False based on measured hardware benchmark.',
                            'Default full-window 400-point FFT corrects truncation; repository_200fft option preserves original extraction.',
                            'MLP follows released four hidden layers plus output (five Linear layers); paper says four layers.',
                            'Per-clip caption embeddings retained; released caption regrouping/repetition is not copied.',
                            'Presentation order preserved with manifest checks; no class sorting.',
                            'Single subject; full 200-epoch cosine schedule; no validation-based selection or refit.']))
    print(f'Complete: {out}\nBlock 7 semantic predictions: (200,77,768).',flush=True)


if __name__=='__main__':main()
