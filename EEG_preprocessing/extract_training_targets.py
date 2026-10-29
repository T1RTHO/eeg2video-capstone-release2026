"""Extract frozen Stable Diffusion 1.4 VAE and CLIP targets for Subject 1.
Run: python EEG_preprocessing/extract_training_targets.py
Optional: --model D:\\models\\stable-diffusion-v1-4 (Diffusers-format directory).
No EEG model is trained here. Block 7 targets are reference-only for evaluation.
Requires torch, diffusers, transformers, huggingface_hub, numpy, matplotlib.
"""
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import shutil
import time
import numpy as np


def to_image_tensor(frames, torch, device):
    if frames.shape != (6, 288, 512, 3) or frames.dtype != np.uint8:
        raise ValueError(f'Expected uint8 RGB (6,288,512,3), got {frames.shape} {frames.dtype}')
    return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 127.5 - 1.0


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--model', default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--revision', default='main', help='Hugging Face revision, resolved to an immutable commit')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    import torch
    import diffusers
    import transformers
    from diffusers import AutoencoderKL
    from transformers import CLIPTokenizer, CLIPTextModel
    from huggingface_hub import snapshot_download
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this environment. Check PyTorch or explicitly use --device cpu (slow).')
    root = args.root.resolve()
    src = root/'outputs'/'prepare_video_clips'/'paired_samples.csv'
    out = root/'outputs'/Path(__file__).stem
    prep_summary = json.loads((src.parent/'summary.json').read_text(encoding='utf-8'))
    if prep_summary.get('status') != 'complete':
        raise ValueError('Video preparation must finish before target extraction.')
    with src.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1400 or len({r['sample_id'] for r in rows}) != 1400:
        raise ValueError('Expected 1,400 unique paired samples.')
    for i, row in enumerate(rows):
        expected_split = 'train' if i < 1200 else 'test'
        expected_index = i if i < 1200 else i-1200
        if row['split'] != expected_split or int(row['split_index']) != expected_index:
            raise ValueError(f'Split/order mismatch at row {i}.')
        if not (root/row['rgb_array']).is_file() or not row['caption'].strip():
            raise ValueError(f'Missing RGB input or caption for {row["sample_id"]}.')
    run = dict(manifest_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), model=args.model,
               revision=args.revision, latent_method='posterior_mode', precision='float32', version=1)
    state_path = out/'progress.json'
    if out.exists() and any(out.iterdir()):
        if not state_path.exists():
            raise RuntimeError(f'Incomplete output without progress metadata: {out}. Rename it before retrying.')
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['run'] != run:
            raise ValueError('Inputs/settings differ from saved run. Rename the output folder before changing settings.')
        if state.get('complete'):
            print(f'Target extraction already complete: {out}'); return
        print(f'Resuming from clip {state["done"]}/1400.', flush=True)
    else:
        if shutil.disk_usage(root).free < 1.5*1024**3:
            raise RuntimeError('Allow at least 1.5 GiB output space, plus space for downloaded model components.')
        out.mkdir(parents=True, exist_ok=True)
        state = dict(run=run, done=0, text_complete=False, complete=False)
        atomic_json(state_path, state)
    print('Loading frozen Stable Diffusion 1.4 components (downloads on first use).', flush=True)
    if Path(args.model).is_dir():
        model_path = str(Path(args.model).resolve())
    else:
        # On resume use the exact cached snapshot originally selected.
        model_path = state.get('resolved_model')
        if not model_path or not Path(model_path).is_dir():
            model_path = snapshot_download(args.model, revision=state.get('resolved_revision') or args.revision,
                allow_patterns=['vae/*', 'text_encoder/*', 'tokenizer/*', 'model_index.json'])
    for component in ['vae', 'text_encoder', 'tokenizer']:
        if not (Path(model_path)/component).is_dir():
            raise FileNotFoundError(f'Model directory lacks {component}: {model_path}')
    state['resolved_model'] = model_path
    if not Path(args.model).is_dir():
        state['resolved_revision'] = Path(model_path).name
    atomic_json(state_path, state)
    device = torch.device(args.device)
    print('Device:', torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU', flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder='tokenizer', local_files_only=True)
    if not state['text_complete']:
        encoder = CLIPTextModel.from_pretrained(model_path, subfolder='text_encoder', local_files_only=True)
        encoder = encoder.to(device=device, dtype=torch.float32).eval()
        encoder.requires_grad_(False)
        if encoder.config.hidden_size != 768:
            raise ValueError('Expected SD 1.4 CLIP hidden size 768. Do not substitute an incompatible model.')
        def text_embeddings(texts):
            tokens = tokenizer(texts, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
            kwargs = {'input_ids': tokens.input_ids.to(device)}
            if getattr(encoder.config, 'use_attention_mask', False):
                kwargs['attention_mask'] = tokens.attention_mask.to(device)
            return encoder(**kwargs)[0].float().cpu().numpy()
        with torch.no_grad():
            for split, selected in [('train', rows[:1200]), ('test', rows[1200:])]:
                target = np.lib.format.open_memmap(out/f'{split}_text_embeddings.npy', mode='w+',
                                                  dtype='float32', shape=(len(selected),77,768))
                for start in range(0,len(selected),16):
                    target[start:start+16] = text_embeddings([r['caption'] for r in selected[start:start+16]])
                if not np.isfinite(target).all():
                    raise ValueError('Non-finite text embeddings.')
                target.flush(); del target
            negative = text_embeddings([''])
            np.save(out/'negative_text_embedding.npy', negative)
        del encoder
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
        state['text_complete'] = True
        atomic_json(state_path, state)
        print('Caption targets saved.', flush=True)
    # Reusable mapping makes test references visibly distinct from training targets.
    shutil.copyfile(src, out/'sample_mapping.csv')
    vae = AutoencoderKL.from_pretrained(model_path, subfolder='vae', local_files_only=True)
    vae = vae.to(device=device, dtype=torch.float32).eval()
    vae.requires_grad_(False)
    scale = float(getattr(vae.config, 'scaling_factor', 0.18215))
    if int(vae.config.latent_channels)!=4 or abs(scale-0.18215)>1e-6:
        raise ValueError(f'Unexpected VAE configuration: latent channels={vae.config.latent_channels}, scale={scale}')
    state['latent_scaling_factor'] = scale
    atomic_json(state_path, state)
    arrays = {}
    for split,count in [('train',1200),('test',200)]:
        path = out/f'{split}_visual_latents.npy'
        shape = (count,6,4,36,64)
        if path.exists():
            array = np.load(path,mmap_mode='r+')
            if array.shape != shape or array.dtype!=np.float32: raise ValueError(f'Invalid target array: {path}')
        else:
            if state['done']>0: raise FileNotFoundError(f'Resume target missing: {path}')
            array = np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=shape)
        arrays[split] = array
    started = time.perf_counter()
    initial = state['done']
    with torch.no_grad():
        for i in range(initial,1400):
            row = rows[i]
            frames = np.load(root/row['rgb_array'],allow_pickle=False)
            images = to_image_tensor(frames,torch,device)
            latents = []
            # One frame at a time limits GPU memory use; output order stays unchanged.
            for frame in images.split(1):
                latent = vae.encode(frame).latent_dist.mode() * scale
                latents.append(latent.float().cpu().numpy())
            z = np.concatenate(latents,axis=0)
            if z.shape!=(6,4,36,64) or not np.isfinite(z).all():
                raise ValueError(f'Invalid VAE output for {row["sample_id"]}.')
            arrays[row['split']][int(row['split_index'])] = z
            done = i+1
            if done%10==0 or done==1400:
                for array in arrays.values(): array.flush()
                state['done'] = done
                atomic_json(state_path,state)
                elapsed = time.perf_counter()-started
                eta = elapsed/(done-initial)*(1400-done)
                print(f'\rVAE clips {done}/1400 | {done/14:.1f}% | elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min',end='',flush=True)
        # Diagnostic only: frozen VAE encode/decode of a training clip, not EEG prediction.
        originals = np.load(root/rows[0]['rgb_array'],allow_pickle=False)
        reconstructed = []
        for z in arrays['train'][0]:
            decoded = vae.decode(torch.from_numpy(z.copy())[None].to(device)/scale).sample
            rgb = (decoded.clamp(-1,1)+1)/2
            reconstructed.append(rgb[0].permute(1,2,0).cpu().numpy())
    fig, axes = plt.subplots(2,6,figsize=(16,5),constrained_layout=True)
    for j in range(6):
        axes[0,j].imshow(originals[j]); axes[1,j].imshow(reconstructed[j])
        axes[0,j].set_title(f'Frame {j+1}',fontsize=10)
        for ax in axes[:,j]: ax.set_xticks([]); ax.set_yticks([])
    axes[0,0].set_ylabel('Original'); axes[1,0].set_ylabel('VAE decoded')
    fig.suptitle('Frozen VAE round trip: Block 1, clip 1 (not EEG reconstruction)')
    fig.savefig(out/'vae_roundtrip_preview.png',dpi=150); plt.close(fig)
    summary = dict(status='complete', model_source=args.model, resolved_model=model_path,
        resolved_revision=state.get('resolved_revision','local directory'),
        versions=dict(torch=torch.__version__,diffusers=diffusers.__version__,transformers=transformers.__version__),
        latent_scaling_factor=scale, latent_method='Posterior mode (mean): deterministic target extraction; paper does not specify mean versus sampling.',
        rgb_preprocessing='uint8 RGB -> float32 NCHW in [-1,1]; no crop or augmentation.',
        train_visual_shape=[1200,6,4,36,64],test_visual_shape=[200,6,4,36,64],
        train_text_shape=[1200,77,768],test_text_shape=[200,77,768],dtype='float32',
        text_target='Last hidden states of frozen SD 1.4 CLIP encoder, maximum 77 tokens, original per-clip BLIP captions.',
        negative_conditioning='Empty caption embedding; explicit implementation choice.',
        test_policy='Block 7 targets are evaluation references only. Never supply them as EEG-predicted conditioning.',
        training='No models trained. VAE and text encoder frozen; no statistics fitted to the test set.',
        future_consistency='Use the same resolved SD 1.4 components, latent scale and sample mapping downstream.')
    atomic_json(out/'summary.json',summary)
    state['complete'] = True
    atomic_json(state_path,state)
    print(f'\nComplete: {out}\nVisual and semantic targets ready. EEG predictors have not yet been trained.')


if __name__ == '__main__':
    main()
