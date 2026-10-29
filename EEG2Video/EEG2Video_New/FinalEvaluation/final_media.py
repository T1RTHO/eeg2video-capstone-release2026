"""Exports actual arrays, frame strips, GIFs, MP4s and a local HTML gallery."""
import html
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

def export_clip(folder, sample_id, named_frames):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    width,height=256,144
    panel=Image.new('RGB',(6*width,len(named_frames)*(height+28)), 'white')
    draw=ImageDraw.Draw(panel)
    for row,(name,frames) in enumerate(named_frames.items()):
        frames=np.asarray(frames)
        if frames.shape!=(6,288,512,3) or frames.dtype!=np.uint8:
            raise ValueError('Expected uint8 RGB 6x288x512x3: '+name)
        y=row*(height+28)
        draw.text((5,y+5),name+' | '+sample_id,fill='black')
        for j,frame in enumerate(frames):
            panel.paste(Image.fromarray(frame).resize((width,height)),(j*width,y+28))
    panel.save(folder/'comparison.png')
    # Side by side for each frame; label each row rather than overlaying content.
    movie=[]
    for j in range(6):
        frame=Image.new('RGB',(512,len(named_frames)*320),'white');d=ImageDraw.Draw(frame)
        for k,(name,arr) in enumerate(named_frames.items()):
            d.text((5,k*320+8),name+' | '+sample_id,fill='black')
            frame.paste(Image.fromarray(arr[j]),(0,k*320+32))
        movie.append(np.asarray(frame))
    imageio.mimsave(folder/'comparison.gif',movie,duration=333,loop=0)
    imageio.mimwrite(folder/'comparison.mp4',movie,fps=3,codec='libx264',quality=8,macro_block_size=1)

def gallery(out,title):
    out=Path(out)
    cards=[]
    for p in sorted(out.glob('clips/*/comparison.png')):
        rel=p.parent.relative_to(out).as_posix(); sid=html.escape(p.parent.name)
        cards.append(f'<article><h2>{sid}</h2><a href="{rel}/comparison.png"><img loading="lazy" src="{rel}/comparison.png"></a><p><a href="{rel}/comparison.mp4">MP4</a> | <a href="{rel}/comparison.gif">GIF</a></p></article>')
    page='<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:16px system-ui;background:#f1f5f9;color:#172838;padding:24px;max-width:1500px;margin:auto}article{background:white;padding:16px;margin:20px 0}img{width:100%}h2{font-size:17px}</style><h1>'+html.escape(title)+'</h1><p>All completed clips in sample order; no selection by output quality. GIF/MP4 are for viewing. Metrics use lossless RGB arrays.</p>'+''.join(cards)
    (out/'index.html').write_text(page,encoding='utf-8')

def decode(vae,z,scale,device='cpu'):
    import torch
    result=[]
    with torch.no_grad():
        for frame in z:
            x=torch.as_tensor(np.array(frame),dtype=next(vae.parameters()).dtype,device=device)[None]/scale
            y=vae.decode(x).sample[0].float().cpu().permute(1,2,0).numpy()
            if not np.isfinite(y).all(): raise ValueError('Non-finite decoded image')
            result.append(np.rint(np.clip(y/2+0.5,0,1)*255).astype(np.uint8))
    return np.stack(result)
