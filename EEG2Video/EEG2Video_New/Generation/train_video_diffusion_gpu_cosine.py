"""Tune upstream temporal UNet on training videos only. Epoch-based cosine schedule; resumable bounded GPU sessions."""
import argparse,time,random,math
from pathlib import Path
import numpy as np
from common import manifest,targets,base_model,write_json,read_json,sha,save_csv,COMMIT

def cosine_lr(completed_updates,total_updates):
    return 3e-5*0.5*(1+math.cos(math.pi*completed_updates/total_updates))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('D:/Capstone/work'))
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--session-steps',type=int,default=5,help='Updates this invocation; does not change the full cosine schedule')
    p.add_argument('--save-every',type=int,default=5)
    p.add_argument('--training-set',choices=['paper_all','readme_80'],default='paper_all')
    p.add_argument('--accumulate',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--attention',choices=['max','auto','xformers'],default='xformers')
    p.add_argument('--resume',action='store_true');p.add_argument('--out',type=Path)
    a=p.parse_args()
    if min(a.epochs,a.accumulate,a.session_steps,a.save_every)<1:p.error('Positive arguments required')
    import torch
    from diffusers import AutoencoderKL,DDPMScheduler
    from tqdm import tqdm
    from vendor.unet import UNet3DConditionModel
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for training')
    root=a.root.resolve();rr=manifest(root)[:1200];td,s=targets(root);model_path=base_model(root,full=True)
    from common import rows
    target_rows=rows(td/'sample_mapping.csv')
    if [r['sample_id'] for r in target_rows[:1200]] != [r['sample_id'] for r in rr]:
        raise ValueError('Training target order differs from video manifest')
    chosen=list(range(1200))
    if a.training_set=='readme_80':
        # Deterministic two clips per concept from distinct training blocks.
        chosen=[]
        for concept in sorted({int(r['concept_id']) for r in rr}):
            chosen.extend(next(i for i,r in enumerate(rr) if int(r['concept_id'])==concept and int(r['block'])==b) for b in [1,2])
    if len(chosen)%a.accumulate:raise ValueError('Accumulation must divide the training set size for exact epochs')
    updates_per_epoch=len(chosen)//a.accumulate
    a.steps=a.epochs*updates_per_epoch
    out=a.out or root/'outputs/video_diffusion_gpu_cosine';out=out.resolve()
    cfg={'trainer_version':'gpu_cosine_1','epochs':a.epochs,'updates_per_epoch':updates_per_epoch,
         'lr_schedule':'cosine_no_warmup','attention':a.attention,'source_commit':COMMIT,'steps':a.steps,'training_set':a.training_set,'indices':chosen,
         'sample_ids':[rr[i]['sample_id'] for i in chosen],'effective_batch_size':a.accumulate,'microbatch':1,
         'seed':a.seed,'learning_rate':3e-5,'optimizer':'AdamW','weight_decay':0.01,
         'trainable_modules':['attn1.to_q','attn2.to_q','attn_temp'],
         'model_revision':s.get('resolved_revision'),'model_source':s['model_source'],
         'manifest_sha256':sha(root/'outputs/prepare_video_clips/paired_samples.csv'),
         'text_sha256':sha(td/'train_text_embeddings.npy'),'latent_scale':s['latent_scaling_factor'],
         'vae_latents':'posterior samples from training RGB; frozen VAE on CPU',
         'limitations':['Paper B.3 specifies 200 epochs and cosine; zero warmup is an implementation choice.',
                        'microbatch 1, gradient accumulation and frozen-half/trainable-float32 storage are hardware adaptations.',
                        'readme_80 selection, if used, is deterministic but original 80 IDs are unreleased.']}
    if out.exists() and any(out.iterdir()):
        if not a.resume:raise ValueError('Output exists: use --resume with unchanged settings')
        if read_json(out/'run_config.json')!=cfg:raise ValueError('Resume settings or data differ')
    else:
        out.mkdir(parents=True,exist_ok=True);write_json(out/'run_config.json',cfg)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    ck=out/'latest.pt'
    if a.resume and not ck.exists():raise ValueError('No completed checkpoint to resume')
    print('Loading model; full budget:',a.steps,'updates. This session:',a.session_steps,flush=True)
    net=UNet3DConditionModel.from_pretrained_2d(str(model_path),subfolder='unet')
    net.requires_grad_(False)
    for name,module in net.named_modules():
        if name.endswith(tuple(cfg['trainable_modules'])):module.requires_grad_(True)
    net.to(device='cuda',dtype=torch.float16)
    for param in net.parameters():
        if param.requires_grad:param.data=param.data.float()
    trainable=[x for x in net.parameters() if x.requires_grad]
    if not trainable:raise ValueError('No trainable parameters')
    net.enable_gradient_checkpointing()
    if a.attention=='xformers':net.enable_xformers_memory_efficient_attention()
    else:net.set_attention_slice(a.attention)
    net.train()
    optimizer=torch.optim.AdamW(trainable,lr=3e-5,betas=(.9,.999),weight_decay=.01,eps=1e-8,foreach=False)
    scaler=torch.cuda.amp.GradScaler()
    scheduler=DDPMScheduler.from_pretrained(str(model_path),subfolder='scheduler')
    if scheduler.config.prediction_type!='epsilon':raise ValueError('Expected SD1.4 epsilon prediction')
    vae=AutoencoderKL.from_pretrained(str(model_path),subfolder='vae').eval().requires_grad_(False)
    embeddings=np.load(td/'train_text_embeddings.npy',mmap_mode='r',allow_pickle=False)
    if embeddings.shape!=(1200,77,768):raise ValueError('Wrong training text shape')
    start=0;history=[];order=[];cursor=0
    if a.resume:
        state=torch.load(ck,map_location='cpu')
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start=state['step'];history=state['history'];order=state['order'];cursor=state['cursor']
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']);del state
    end=min(a.steps,start+a.session_steps)
    for step in tqdm(range(start,end),initial=start,total=a.steps,desc='Total training updates'):
        lr=cosine_lr(step,a.steps)
        for group in optimizer.param_groups:group['lr']=lr
        torch.cuda.reset_peak_memory_stats()
        tqdm.write(f'Epoch {step//updates_per_epoch+1}/{a.epochs}; update {step+1}/{a.steps}; LR {lr:.8g}')
        began=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for micro in tqdm(range(a.accumulate),desc='Clips in update',leave=False):
            if cursor>=len(order):order=random.sample(chosen,len(chosen));cursor=0
            index=order[cursor];cursor+=1;r=rr[index]
            rgb=np.load(root/r['rgb_array'],allow_pickle=False)
            if rgb.shape!=(6,288,512,3) or rgb.dtype!=np.uint8:raise ValueError('Invalid training frames')
            frames=torch.from_numpy(rgb.copy()).permute(0,3,1,2).float()/127.5-1
            tqdm.write(f'Clip {micro+1}/{a.accumulate}: VAE encoding')
            with torch.no_grad():
                z=torch.cat([vae.encode(f[None]).latent_dist.sample() for f in tqdm(frames,desc='VAE frames',leave=False)])*s['latent_scaling_factor']
            z=z.permute(1,0,2,3)[None].to('cuda',torch.float16)
            noise=torch.randn_like(z);t=torch.randint(0,scheduler.config.num_train_timesteps,(1,),device='cuda')
            noisy=scheduler.add_noise(z,noise,t).requires_grad_(True)
            cond=torch.from_numpy(np.array(embeddings[index:index+1])).to('cuda',torch.float16)
            tqdm.write(f'Clip {micro+1}/{a.accumulate}: UNet forward')
            with torch.autocast('cuda',dtype=torch.float16):
                pred=net(noisy,t,encoder_hidden_states=cond).sample
                loss=torch.nn.functional.mse_loss(pred.float(),noise.float())
            if not torch.isfinite(loss):raise RuntimeError('Non-finite diffusion loss')
            tqdm.write(f'Clip {micro+1}/{a.accumulate}: backward')
            scaler.scale(loss/a.accumulate).backward();loss_sum+=loss.item()/a.accumulate
            del loss,pred,noisy,z,cond,noise
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        history.append({'step':step+1,'epoch':(step+1)/updates_per_epoch,'learning_rate':lr,'loss':loss_sum,'seconds':time.perf_counter()-began,
                        'cuda_peak_gib':torch.cuda.max_memory_allocated()/2**30})
        save_csv(out/'training_log.csv',history)
        if (step+1)%a.save_every==0 or step+1==end:
            tqdm.write('Saving resumable checkpoint...')
            state={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                   'step':step+1,'history':history,'order':order,'cursor':cursor,
                   'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            torch.save(state,out/'latest.tmp');(out/'latest.tmp').replace(ck)
            del state
            write_json(out/'progress.json',{'completed_steps':step+1,'total_steps':a.steps,'epochs_completed':(step+1)/updates_per_epoch})
            import matplotlib;matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            axes[0].plot([x['step'] for x in history],[x['loss'] for x in history]);axes[0].set(xlabel='Optimizer step',ylabel='Noise prediction MSE',title='Training loss (Blocks 1–6)')
            axes[1].plot([x['step'] for x in history],[x['learning_rate'] for x in history]);axes[1].set(xlabel='Optimizer step',ylabel='Learning rate',title='Cosine schedule')
            fig.tight_layout();fig.savefig(out/'training_curve.png',dpi=180);plt.close(fig)
    if end==a.steps:
        print('Exporting final model...',flush=True)
        net.save_pretrained(out/'unet');scheduler.save_pretrained(out/'scheduler')
    write_json(out/'provenance.json',dict(cfg,status='complete' if end==a.steps else 'incomplete',completed_steps=end,test_block_used=False))
    print('Session saved:',end,'of',a.steps,'updates at',out,flush=True)
    print('Continue with the same settings and --resume; session-steps may change.',flush=True)

if __name__=='__main__':main()
