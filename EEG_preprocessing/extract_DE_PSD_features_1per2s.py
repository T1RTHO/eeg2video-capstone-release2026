"""Subject 1 frequency features for EEG2Video semantic/dynamic prediction.
Run from project root in eeg2video. Requires numpy and matplotlib only.
Saves TWO explicitly named variants; no model training or automatic selection.
repository_200fft reproduces the released DE_PSD arithmetic (including truncation).
full_window_400fft changes FFT length to retain the complete 400-sample window,
scaling the repository frequency-bin formula accordingly. It is an adaptation,
not a verified published implementation. No filtering/standardisation is applied.
Source: XuanhaoLiu/EEG2Video, EEG_preprocessing/DE_PSD.py and
extract_DE_PSD_features_1per2s.py, commit 18af805753bc89e796625cdaaca1d0844bee22a8.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

BANDS=[(1,4),(4,8),(8,14),(14,31),(31,99)]
NAMES=['delta','theta','alpha','beta','gamma']
MODES={'repository_200fft':200,'full_window_400fft':400}


def extract_features(data,nfft):
    """Input (...,62,400); return DE, PSD (...,62,5), and clamped count.
    Retains original nonstandard Hann window, bin boundaries, mean squared
    magnitudes, and log2(100*E) definition. PSD is the repository's band-power
    proxy, not a calibrated density in units/Hz. Only FFT length differs.
    """
    x=np.asarray(data,dtype=np.float64)
    if x.shape[-2:]!=(62,400) or not np.isfinite(x).all():
        raise ValueError('Expected finite EEG with last dimensions (62,400).')
    if nfft not in (200,400): raise ValueError('Expected FFT length 200 or 400.')
    window=.5-.5*np.cos(2*np.pi*np.arange(1,401)/401)
    magnitude=np.abs(np.fft.fft(x*window,n=nfft,axis=-1)[...,:nfft//2])
    powers=[]
    for lo,hi in BANDS:
        first=int(lo/200*nfft);last=int(hi/200*nfft)
        powers.append(np.mean(magnitude[...,first-1:last]**2,axis=-1))
    psd=np.stack(powers,axis=-1)
    # Original math.log fails on zero power; floor is recorded explicitly.
    tiny=np.finfo(np.float64).tiny
    clamped=int(np.count_nonzero(psd<tiny))
    de=np.log2(100*np.maximum(psd,tiny))
    if not np.isfinite(de).all() or not np.isfinite(psd).all():
        raise ValueError('Non-finite features; inspect raw EEG amplitude.')
    return de,psd,clamped


def fingerprint(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)


def plots(out,features):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    lo=min(float(v.min()) for v in features.values());hi=max(float(v.max()) for v in features.values())
    for ax,(name,value) in zip(axes,features.items()):
        im=ax.imshow(value,aspect='auto',origin='lower',vmin=lo,vmax=hi,cmap='viridis')
        ax.set_title(name);ax.set_xlabel('Frequency band');ax.set_ylabel('EEG channel index')
        ax.set_xticks(range(5),NAMES,rotation=30)
    fig.colorbar(im,ax=list(axes),label='Repository DE definition: log2(100 x band power)')
    fig.suptitle('First training clip: feature extraction variants')
    fig.savefig(out/'feature_comparison.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.cwd())
    args=ap.parse_args();root=args.root.resolve()
    source=root/'outputs'/'segment_raw_signals_200Hz'
    out=root/'outputs'/Path(__file__).stem
    files=[source/'train_eeg.npy',source/'test_eeg.npy',source/'sample_mapping.csv']
    with files[2].open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=1400:raise ValueError('Expected 1400 manifest rows.')
    for i,r in enumerate(rows):
        expected=f'sub01_session01_block{i//200+1:02d}_clip{i%200+1:03d}'
        if (r['sample_id']!=expected or r['split']!=('train' if i<1200 else 'test')
            or int(r['split_index'])!=(i if i<1200 else i-1200)):
            raise ValueError('Unexpected EEG row order or split.')
    train=np.load(files[0],mmap_mode='r',allow_pickle=False)
    test=np.load(files[1],mmap_mode='r',allow_pickle=False)
    if train.shape!=(1200,62,400) or test.shape!=(200,62,400):raise ValueError('Wrong EEG array shapes.')
    cfg=dict(subject=1,session=1,sampling_rate=200,input_samples_per_clip=400,
             bands_hz=BANDS,band_names=NAMES,variants=MODES,
             source_hashes={p.name:fingerprint(p) for p in files},script_hash=fingerprint(Path(__file__)))
    if out.exists() and any(out.iterdir()):
        if (out/'summary.json').exists():
            previous=json.loads((out/'summary.json').read_text())
            if previous.get('config')==cfg and previous.get('status')=='complete':
                print(f'Already complete: {out}');return
        raise FileExistsError(f'Existing/incomplete results at {out}; rename this folder before rerunning.')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'run_config.json',cfg)
    shutil.copyfile(files[2],out/'sample_mapping.csv')
    started=time.perf_counter();examples={};stats={}
    for mode,nfft in MODES.items():
        folder=out/mode;folder.mkdir()
        de_all=np.empty((1400,62,5),dtype=np.float64);psd_all=np.empty_like(de_all);clamped=0
        for block in range(7):
            data=train[block*200:(block+1)*200] if block<6 else test
            de,psd,count=extract_features(data,nfft)
            de_all[block*200:(block+1)*200]=de;psd_all[block*200:(block+1)*200]=psd;clamped+=count
            completed=list(MODES).index(mode)*7+block+1;elapsed=time.perf_counter()-started
            print(f'\r{mode}: Block {block+1}/7 | {completed}/14 total | elapsed {elapsed:.1f}s | ETA {elapsed/completed*(14-completed):.1f}s',end='',flush=True)
        print()
        for name,a in [('de',de_all),('psd',psd_all)]:
            np.save(folder/f'train_{name}_features.npy',a[:1200].reshape(1200,310).astype(np.float32))
            np.save(folder/f'test_{name}_features.npy',a[1200:].reshape(200,310).astype(np.float32))
            # Original repository-compatible grouping, still in presentation order.
            np.save(folder/f'sub1_{name}.npy',a.reshape(7,40,5,62,5))
        examples[mode]=de_all[0]
        stats[mode]=dict(fft_length=nfft,zero_power_entries_clamped=clamped,
                        effective_samples_used=min(nfft,400),de_range=[float(de_all.min()),float(de_all.max())])
    plots(out,examples)
    summary=dict(status='complete',config=cfg,variant_statistics=stats,
        train_shape=[1200,310],test_shape=[200,310],train_blocks=[1,2,3,4,5,6],test_block=7,
        feature_order='channel-major: channel 0 five bands, channel 1 five bands, ... channel 61 five bands',
        standardisation='None here. Fit mean/std on training features only in the predictor trainer.',
        variant_selection='No variant selected for training by this script.',
        differences=['Repository FFT length 200 truncates the 400-sample windowed signal to its first 200 samples.',
                     'Full-window variant uses FFT length 400 and scaled source bin indices; not verified as published implementation.',
                     'Both retain the source Hann formula, inclusive overlapping band-bin convention and log2(100*E) DE.',
                     'A float64 tiny floor prevents log(0); number of affected entries is reported.',
                     'NumPy FFT replaces scipy.fftpack FFT; reference equivalence is numerical, not bitwise.',
                     'No class sorting; the existing paired manifest order is preserved.'],
        source_commit='18af805753bc89e796625cdaaca1d0844bee22a8',
        runtime_seconds=round(time.perf_counter()-started,2))
    write_json(out/'summary.json',summary)
    print(f'Complete: {out}\nEach variant: training (1200,310), test (200,310). No model trained.')


if __name__=='__main__':main()
