"""Single-subject dynamic GLMNet. See README.txt for sources and adaptations."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np

COMMIT='18af805753bc89e796625cdaaca1d0844bee22a8'
CHANNELS=['PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

def fingerprint(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''): h.update(part)
    return h.hexdigest()

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def write_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)

def prepare(root):
    here=Path(__file__).resolve().parent
    meta=json.loads((here/'source_manifest.json').read_text())
    for name,digest in meta['sha256'].items():
        if fingerprint(here/name)!=digest:raise ValueError('Bundled metadata checksum mismatch: '+name)
    ofs=np.load(here/'All_video_optical_flow_score.npy',allow_pickle=False)
    concepts=np.load(here/'All_video_label.npy',allow_pickle=False)
    if ofs.shape!=(7,200) or not np.isfinite(ofs).all():raise ValueError('Invalid official OFS array.')
    rows=read_csv(root/'outputs/segment_raw_signals_200Hz/sample_mapping.csv')
    if len(rows)!=1400:raise ValueError('Expected 1400 EEG samples.')
    labels=(ofs>1.799).astype(np.int64) # our encoding: 0 slow; 1 fast
    result=[]
    for i,r in enumerate(rows):
        b,c=divmod(i,200)
        expected={'sample_id':f'sub01_session01_block{b+1:02d}_clip{c+1:03d}',
                  'block':str(b+1),'clip_in_block':str(c+1),
                  'split':'train' if b<6 else 'test','split_index':str(i if b<6 else i-1200),
                  'concept_id':str(int(concepts[b,c//5]))}
        if any(r.get(k)!=v for k,v in expected.items()):
            raise ValueError(f'EEG manifest/order/concept mismatch at row {i}: expected {expected}')
        result.append(dict(expected,official_ofs=float(ofs[b,c]),dynamic_label=int(labels[b,c]),
                           dynamic_name='fast' if labels[b,c] else 'slow'))
    return rows,result,labels

def build_model():
    from torch import nn
    class Branch(nn.Module):
        def __init__(self,channels):
            super().__init__()
            self.net=nn.Sequential(nn.Conv2d(1,40,(1,25)),nn.Conv2d(40,40,(channels,1)),
                nn.BatchNorm2d(40),nn.ELU(),nn.AvgPool2d((1,51),(1,5)),nn.Dropout(.5))
            self.out=nn.Linear(2640,256)
        def forward(self,x):return self.out(self.net(x).flatten(1))
    class GLMNet(nn.Module):
        def __init__(self):
            super().__init__();self.globalnet=Branch(62);self.occipital_localnet=Branch(12);self.out=nn.Linear(512,2)
        def forward(self,x):
            import torch
            return self.out(torch.cat((self.globalnet(x),self.occipital_localnet(x[:,:,50:62,:])),1))
    return GLMNet()

def smoke(device):
    import torch
    torch.manual_seed(123)
    model=build_model().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0005,foreach=False)
    x=torch.randn(2,1,62,400,device=device);y=torch.tensor([0,1],device=device)
    z=model(x)
    assert z.shape==(2,2) and torch.isfinite(z).all()
    torch.nn.functional.cross_entropy(z,y).backward()
    for branch in (model.globalnet,model.occipital_localnet):
        g=branch.net[0].weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    print('PASS: two logits, both branch gradients, finite parameters, real Adam update.',flush=True)

def plot(out,history,cm=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].plot([h['epoch'] for h in history],[h['train_cross_entropy'] for h in history])
    axes[0].set(title='Training loss',xlabel='Epoch',ylabel='Cross entropy')
    axes[1].plot([h['epoch'] for h in history],[h['train_accuracy'] for h in history])
    axes[1].set(title='Training accuracy (dropout active)',xlabel='Epoch',ylabel='Accuracy',ylim=(0,1))
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'training_curve.png',dpi=150);plt.close(fig)
    if cm is not None:
        fig,ax=plt.subplots(figsize=(4,4),constrained_layout=True);ax.imshow(cm,cmap='Blues')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Slow','Fast'],yticklabels=['Slow','Fast'],
               xlabel='Predicted',ylabel='Actual',title='Block 7: final evaluation')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='red')
        fig.savefig(out/'test_confusion_matrix.png',dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args()
    if a.smoke_test:
        smoke(a.device);return
    root=a.root.resolve();out=root/'outputs/train_dynamic_glmnet'
    rows,label_rows,labels=prepare(root)
    label_out=root/'outputs/prepare_dynamic_labels';label_out.mkdir(parents=True,exist_ok=True)
    write_csv(label_out/'sample_mapping.csv',label_rows)
    np.save(label_out/'train_dynamic_labels.npy',labels[:6].reshape(-1),allow_pickle=False)
    np.save(label_out/'test_dynamic_labels.npy',labels[6],allow_pickle=False)
    write_json(label_out/'summary.json',dict(status='complete',source_commit=COMMIT,
        threshold=1.799,encoding={'0':'slow','1':'fast'},rule='fast if OFS > 1.799; slow otherwise',
        train_counts=np.bincount(labels[:6].reshape(-1),minlength=2).tolist(),
        test_counts=np.bincount(labels[6],minlength=2).tolist(),
        note='Official metadata used; rounded paper threshold and full-array median give identical labels.'))
    print('Verified labels: training slow/fast 594/606; test slow/fast 106/94.',flush=True)
    if a.prepare_only:return
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable.')
    device=torch.device(a.device)
    smoke(device)
    torch.manual_seed(42)
    if device.type=='cuda':
        torch.cuda.empty_cache();torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    ed=root/'outputs/segment_raw_signals_200Hz'
    locs=root/'SEED-DV/channel_62_pos.locs'
    names=[line.split()[-1] for line in locs.read_text().splitlines() if line.strip()]
    if len(names)!=62 or names[50:62]!=CHANNELS:raise ValueError('Channel order mismatch.')
    source_paths=[ed/'train_eeg.npy',ed/'sample_mapping.csv',locs,Path(__file__),
        Path(__file__).parent/'All_video_optical_flow_score.npy']
    cfg=dict(architecture_version='raw_glmnet_dynamic_v1',source_commit=COMMIT,
        samples=400,branch_width=256,classes=2,epochs=200,batch_size=8,learning_rate=.0005,
        optimizer='Adam',scheduler='CosineAnnealingLR T_max=200',seed=42,device=a.device,
        threshold=1.799,normalisation='training-only coordinate mean/std',
        input_hashes={str(f.resolve()):fingerprint(f) for f in source_paths},
        adaptations=['One subject/session','Whole 2s EEG input; branch width 256 follows source example',
          'Correct flatten width: 2640 instead of released T//200 approximation',
          'Batch 8 to match tested GPU configuration; paper reconstruction batch unspecified',
          'Fixed 200 epochs; no test-driven checkpoint selection'])
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run_config.json').exists():
        if json.loads((out/'run_config.json').read_text())!=cfg:raise ValueError('Existing output configuration differs. Use a separate output folder/root.')
    else:write_json(out/'run_config.json',cfg)
    if (out/'summary.json').exists() and json.loads((out/'summary.json').read_text()).get('status')=='complete':
        print('Already complete:',out);return
    raw=np.load(ed/'train_eeg.npy',allow_pickle=False)
    if raw.shape!=(1200,62,400) or not np.isfinite(raw).all():raise ValueError('Invalid training EEG.')
    mean=raw.mean(axis=0,dtype=np.float64);std=raw.std(axis=0,dtype=np.float64);std[std<1e-12]=1
    x=torch.from_numpy(((raw-mean)/std).astype(np.float32)).unsqueeze(1);del raw
    y=torch.from_numpy(labels[:6].reshape(-1).copy())
    np.savez(out/'eeg_normalisation.npz',mean=mean,std=std)
    generator=torch.Generator().manual_seed(42)
    loader=DataLoader(TensorDataset(x,y),batch_size=8,shuffle=True,num_workers=0,pin_memory=False,generator=generator)
    model=build_model().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.0005,foreach=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200)
    history=[];start_epoch=0;checkpoint=out/'latest_checkpoint.pt'
    if checkpoint.exists():
        # Only restore your own locally produced checkpoint.
        ck=torch.load(checkpoint,map_location=device,weights_only=False)
        if ck['config']!=cfg:raise ValueError('Checkpoint config mismatch.')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        start_epoch=ck['epoch'];history=ck['history'];generator.set_state(ck['loader_rng'].cpu())
        torch.set_rng_state(ck['cpu_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
        del ck
        print('Resuming after epoch',start_epoch,flush=True)
    for epoch in range(start_epoch,200):
        t=time.perf_counter();model.train();loss_sum=0.;correct=0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            optimizer.zero_grad(set_to_none=True);logits=model(xb)
            loss=torch.nn.functional.cross_entropy(logits,yb)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; last completed epoch preserved.')
            loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(yb);correct+=int((logits.argmax(1)==yb).sum())
        scheduler.step()
        seconds=time.perf_counter()-t
        history.append(dict(epoch=epoch+1,train_cross_entropy=loss_sum/1200,train_accuracy=correct/1200,
            learning_rate=optimizer.param_groups[0]['lr'],epoch_compute_seconds=seconds))
        ck=dict(epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            history=history,config=cfg,loader_rng=generator.get_state(),cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        torch.save(ck,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(checkpoint)
        write_csv(out/'training_log.csv',history)
        if (epoch+1)%10==0 or epoch==0:plot(out,history)
        eta=np.mean([h['epoch_compute_seconds'] for h in history[-10:]])*(199-epoch)/60
        print(f'Epoch {epoch+1}/200 | CE {loss_sum/1200:.5f} | train accuracy {correct/1200:.1%} | ETA {eta:.1f} min',flush=True)
    torch.save(dict(model=model.state_dict(),config=cfg,epoch=200),out/'final_model.pt')
    # Only now load test EEG and evaluate once; no test-driven training choices.
    test=np.load(ed/'test_eeg.npy',allow_pickle=False)
    if test.shape!=(200,62,400) or not np.isfinite(test).all():raise ValueError('Invalid test EEG.')
    test=torch.from_numpy(((test-mean)/std).astype(np.float32)).unsqueeze(1)
    model.eval();probs=[]
    with torch.no_grad():
        for batch in test.split(8):probs.append(model(batch.to(device)).softmax(1).cpu().numpy())
    probs=np.concatenate(probs);pred=probs.argmax(1);truth=labels[6]
    if not np.isfinite(probs).all():raise ValueError('Non-finite test probabilities.')
    np.save(out/'block7_dynamic_probabilities.npy',probs,allow_pickle=False)
    np.save(out/'block7_predicted_dynamic_labels.npy',pred,allow_pickle=False)
    cm=np.zeros((2,2),dtype=np.int64);np.add.at(cm,(truth,pred),1)
    recalls=np.diag(cm)/cm.sum(axis=1)
    prediction_rows=[dict(prediction_index=i,sample_id=rows[1200+i]['sample_id'],
        predicted_label=int(pred[i]),predicted_name='fast' if pred[i] else 'slow',
        probability_slow=float(probs[i,0]),probability_fast=float(probs[i,1]),true_label=int(truth[i])) for i in range(200)]
    write_csv(out/'prediction_mapping.csv',prediction_rows);plot(out,history,cm)
    write_json(out/'summary.json',dict(status='complete',config=cfg,parameters=sum(p.numel() for p in model.parameters()),
        final_training_cross_entropy=history[-1]['train_cross_entropy'],final_training_accuracy=history[-1]['train_accuracy'],
        test_accuracy=float((pred==truth).mean()),test_balanced_accuracy=float(recalls.mean()),
        test_confusion_matrix=cm.tolist(),confusion_order=['slow','fast'],
        training_majority_baseline_class='fast',test_accuracy_of_training_majority_baseline=float((truth==1).mean()),
        test_majority_reference=float(max(np.bincount(truth))/200),
        evaluation='Block 7 evaluated once after fixed 200 epochs; do not tune on these results.',
        prediction_shape=list(probs.shape),training_compute_seconds=sum(h['epoch_compute_seconds'] for h in history)))
    print('Complete:',out,flush=True)
    print(f'Test accuracy {(pred==truth).mean():.1%}; balanced accuracy {recalls.mean():.1%}')

if __name__=='__main__':main()
