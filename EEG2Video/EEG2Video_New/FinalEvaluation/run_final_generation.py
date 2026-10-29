"""Final single-subject run. Paper beta + released500 DANA + 100 DDIM. Documented adaptation."""
import argparse,time
from pathlib import Path
import numpy as np
from final_common import predictions,manifest,targets,base_model,read_json,write_json,sha,COMMIT
from final_media import decode,export_clip,gallery

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('D:/Capstone/work'))
    p.add_argument('--conditioning',choices=['PP'],default='PP')
    p.add_argument('--session-clips',type=int,default=200)
    p.add_argument('--resume',action='store_true')

    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--attention',choices=['auto','max','xformers'],default='auto')
    p.add_argument('--out',type=Path)
    p.add_argument('--negative-embedding',type=Path,default=Path(__file__).with_name('final_negative.npy'))

    p.add_argument('--checkpoint',type=Path)

    a=p.parse_args()
    if not 1<=a.session_clips<=200:p.error('session-clips must be 1..200')
    a.seed=42;a.mode='full';a.dana_schedule='released500';a.case_label='original_mlp'
    indices=list(range(200))
    from final_metrics import evaluate
    import imageio_ffmpeg
    imageio_ffmpeg.get_ffmpeg_exe()
    import torch
    from diffusers import AutoencoderKL,DDIMScheduler
    from tqdm import tqdm
    import sys
    generation=a.root.resolve()/'EEG2Video/EEG2Video_New/Generation'
    sys.path.insert(0,str(generation))
    from vendor.unet import UNet3DConditionModel
    if a.threads<1:p.error('threads must be positive')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use --device cpu')
    if a.device=='cpu' and a.attention=='xformers':p.error('xformers requires CUDA here')
    torch.set_num_threads(a.threads)
    device=a.device;dtype=torch.float32 if device=='cpu' else torch.float16
    root=a.root.resolve();checkpoint=(a.checkpoint or root/'outputs/video_diffusion_gpu_cosine').resolve()
    prov=read_json(checkpoint/'provenance.json')
    if prov.get('status')!='complete' or prov.get('test_block_used') is not False:
        raise ValueError('Require completed checkpoint with verified training-only provenance')
    if prov.get('completed_steps')!=24000 or prov.get('steps')!=24000 or prov.get('epochs')!=200:
        raise ValueError('Expected completed 200-epoch, 24000-update checkpoint')
    print('Training provenance reports 24000/24000 updates. Checking prediction inputs...',flush=True)
    td,s=targets(root)
    if sha(root/'outputs/prepare_video_clips/paired_samples.csv')!=prov['manifest_sha256']:
        raise ValueError('Manifest changed since diffusion training')
    if sha(td/'train_text_embeddings.npy')!=prov['text_sha256']:
        raise ValueError('Training text embeddings changed since diffusion training')
    if float(prov['latent_scale'])!=float(s['latent_scaling_factor']):
        raise ValueError('Training/target latent scale mismatch')
    if prov['model_revision']!=s.get('resolved_revision'):raise ValueError('Diffusion and target base model revisions differ')
    arr,hashes=predictions(root);rr=manifest(root)[1200:]
    from final_common import rows
    mapping=rows(td/'sample_mapping.csv')
    if [r['sample_id'] for r in mapping]!=[r['sample_id'] for r in manifest(root)]:
        raise ValueError('Reference target manifest mismatch')
    out=(a.out or root/'outputs/final_evaluation/subject01_session01_block07/paper_beta_released500_v1').resolve()
    permitted=(root/'outputs/final_evaluation').resolve()
    if permitted not in out.parents:raise ValueError('Outputs must be beneath outputs/final_evaluation')
    if out.exists() and any(out.iterdir()) and not a.resume:raise ValueError('Output exists. Use --resume with unchanged settings.')
    out.mkdir(parents=True,exist_ok=True)
    negative_path=a.negative_embedding.resolve()
    if not negative_path.is_file():raise FileNotFoundError(negative_path)
    print('Hashing final UNet weights (may take a moment)...',flush=True)
    weights=list((checkpoint/'unet').glob('*.bin'))+list((checkpoint/'unet').glob('*.safetensors'))
    if not weights:raise ValueError('No fine-tuned UNet weights')
    config={'device':device,'dtype':str(dtype),'attention':a.attention,'source_commit':COMMIT,'mode':a.mode,'seed':a.seed,'steps':100,'guidance_scale':12.5,'fps':3,
            'conditioning':a.case_label,'indices':indices,'vendor_hashes':{p.relative_to(generation).as_posix():sha(p) for p in sorted((generation/'vendor').rglob('*.py'))},'diagnostic_only':False,'planned_clips':200,'manifest_sha256':sha(root/'outputs/prepare_video_clips/paired_samples.csv'),'prediction_hashes':hashes,'weight_hashes':{x.name:sha(x) for x in weights},
            'negative_hash':sha(negative_path),'beta':{'slow':.3,'fast':.2},
            'generator_sha256':sha(Path(__file__)),'unet_config_sha256':sha(checkpoint/'unet/config.json'),'scheduler_config_sha256':sha(checkpoint/'scheduler/scheduler_config.json'),
            'protocol_version':'final_paper_beta_released500_v1','dana_schedule':a.dana_schedule,
            'dana_timestep':'Uniform integer 0..499 on separate linear schedule' if a.dana_schedule=='released500' else 'First DDIM timestep on denoising scheduler',
            'deviation':'Paper fast=.2/slow=.3 mapping overrides reversed newer add_noise.py mapping. Released500 uses released noise schedule independently of DDIM. Per-clip seed42+i, sequential CFG and CPU float32 VAE decoding are documented adaptations. No exact-output equivalence claimed.',
            'model_revision':s.get('resolved_revision')}
    cp=out/'run_config.json'
    config['support_code_hashes']={f.name:sha(f) for f in Path(__file__).parent.glob('final_*.py')}
    if cp.exists() and read_json(cp)!=config:raise ValueError('Configuration changed; use separate output folder')
    write_json(cp,config)
    completed=[]
    for i in indices:
        folder=out/'clips'/rr[i]['sample_id'];done=folder/'complete.json'
        if done.exists():
            d=read_json(done)
            if d.get('generated_sha256')!=sha(folder/'generated_rgb.npy'):raise ValueError('Completed output hash mismatch')
            completed.append(i)
    pending=[i for i in indices if i not in completed][:a.session_clips]
    print(f'Already complete: {len(completed)}/200; this session: {len(pending)} clips',flush=True)
    if not pending:
        evaluate(root,out);return
    print('Loading trained UNet...',flush=True)
    net=UNet3DConditionModel.from_pretrained(str(checkpoint),subfolder='unet',torch_dtype=dtype).to(device).eval()
    if a.attention=='xformers':net.enable_xformers_memory_efficient_attention()
    else:net.set_attention_slice(a.attention)
    if not all(torch.isfinite(p).all().item() for p in net.parameters()):
        raise ValueError('Exported UNet contains non-finite weights')
    print('UNet loaded; weights are finite. Preparing DDIM scheduler and CPU VAE...',flush=True)
    scheduler=DDIMScheduler.from_pretrained(str(checkpoint),subfolder='scheduler',clip_sample=False,steps_offset=1,set_alpha_to_one=False)
    vae=AutoencoderKL.from_pretrained(str(base_model(root)),subfolder='vae').eval() # CPU framewise decoding saves VRAM
    neg=np.load(negative_path,allow_pickle=False)
    if neg.shape!=(1,77,768) or not np.isfinite(neg).all():raise ValueError('Invalid negative embedding')
    negative=torch.from_numpy(neg).to(device,dtype)
    released_alphas=torch.cumprod(1-torch.linspace(.0001,.02,500,dtype=torch.float32),dim=0)
    with torch.no_grad():
        for i in tqdm(pending,desc='Generate remaining clips',unit='clip'):
            r=rr[i];folder=out/'clips'/r['sample_id'];done=folder/'complete.json'
            if done.exists():
                previous=read_json(done)
                if previous.get('generated_sha256')!=sha(folder/'generated_rgb.npy'):
                    raise ValueError('Completed RGB integrity mismatch: '+str(folder))
                continue
            folder.mkdir(parents=True,exist_ok=True)
            began=time.perf_counter();scheduler.set_timesteps(100,device=device)
            g=torch.Generator(device=device).manual_seed(a.seed+i)
            z=torch.from_numpy(np.array(arr['visual'][i])).permute(1,0,2,3)[None].to(device,torch.float32)
            label=int(arr['dynamic'][i]);beta=.2 if label else .3
            # DANA uses frame-major float32 noises before casting to UNet dtype.
            # Released implementation samples its own 500-step noise timestep;
            # it then runs the full 100-step denoising schedule independently.
            dana_t=None;alpha=None
            if a.mode=='full':
                if a.dana_schedule=='released500':
                    dana_t=int(torch.randint(0,500,(1,),generator=g,device=device).item())
                    alpha=float(released_alphas[dana_t])
                else:
                    dana_t=int(scheduler.timesteps[0]);alpha=float(scheduler.alphas_cumprod[dana_t])
                diverse=torch.randn((1,6,4,36,64),generator=g,device=device,dtype=torch.float32).permute(0,2,1,3,4)
                shared=torch.randn((1,1,4,36,64),generator=g,device=device,dtype=torch.float32).permute(0,2,1,3,4)
                noise_array=torch.cat([shared.flatten(),diverse.flatten()]).cpu().numpy()
                import hashlib
                noise_hash=hashlib.sha256(noise_array.tobytes()).hexdigest()
                np.save(folder/'noise_shared_diverse.npy',noise_array)
                z=(alpha**.5*z.float()+(1-alpha)**.5*(beta**.5*shared+(1-beta)**.5*diverse)).to(dtype)
            # no_dana retains the released direct-latent ablation; not pure noise.
            z=z.to(dtype)*scheduler.init_noise_sigma
            initial_hash=__import__('hashlib').sha256(z.float().cpu().numpy().tobytes()).hexdigest()
            t0=int(scheduler.timesteps[0])
            tqdm.write(f"{r['sample_id']}: mode={a.mode}, beta={beta}, DANA timestep={dana_t}, alpha={alpha}, first DDIM timestep={t0}")
            cond=torch.from_numpy(np.array(arr['semantic'][i:i+1])).to(device,dtype)
            for t in tqdm(scheduler.timesteps,desc=r['sample_id'],leave=False):
                model_input=scheduler.scale_model_input(z,t)
                # Sequential CFG halves peak activation storage without changing the formula.
                uncond=net(model_input,t,encoder_hidden_states=negative).sample
                conditional=net(model_input,t,encoder_hidden_states=cond).sample
                predicted=uncond+12.5*(conditional-uncond)
                z=scheduler.step(predicted,t,z,eta=0.,generator=g).prev_sample
                if not torch.isfinite(z).all():raise RuntimeError('Non-finite denoising latents')
            tqdm.write(r['sample_id']+': decoding six frames on CPU')
            rgb=decode(vae,z[0].permute(1,0,2,3).float().cpu().numpy(),s['latent_scaling_factor'])
            # Ground truth is loaded AFTER denoising, solely for evidence.
            original=np.load(root/r['rgb_array'].replace('\\','/'),allow_pickle=False)
            export_clip(folder,r['sample_id'],{'Original video':original,a.case_label:rgb})
            np.save(folder/'generated_rgb.npy',rgb)
            write_json(done,{'sample_id':r['sample_id'],'seed':a.seed+i,'dynamic_label':label,'beta':beta,
                'first_ddim_timestep':t0,'dana_timestep':dana_t,'dana_alpha_cumprod':alpha,'protocol':a.dana_schedule,'seconds':time.perf_counter()-began,
                'reference_sha256':sha(root/r['rgb_array'].replace('\\','/')),'generated_sha256':sha(folder/'generated_rgb.npy'),'conditioning':a.case_label,'ground_truth_conditioning':a.case_label=='reference_oracle','noise_sha256':noise_hash,'initial_latent_sha256':initial_hash})
            gallery(out,'EEG2Video: '+a.mode+' reconstruction gallery')
    gallery(out,'EEG2Video audited generation: '+a.dana_schedule+' / '+a.mode)
    print('Generation session finished:',out/'index.html',flush=True)
    del net,vae
    if device=='cuda':torch.cuda.empty_cache()
    evaluate(root,out)

if __name__=='__main__':main()
