"""Prepare SEED-DV videos and captions for the single-subject EEG2Video pipeline.
Run from the project root. Requires numpy, opencv-python, matplotlib.
Outputs: lossless RGB arrays for model input, MP4 previews, paired manifest,
and a six-frame contact sheet for the first clip of each block.
Uses nominal 24-FPS frame boundaries rather than seeking by seconds.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time
import numpy as np
import cv2

NAMES = ['1st_10min', '2nd_10min', '3rd_10min', '4th_10min',
         '5th_10min', '6th_10min', '7th_10min']
OFFSETS = (0, 8, 16, 24, 32, 40)


def read_clip(cap, next_frame, start):
    """Sequential decode; return six RGB frames and next unread frame index."""
    if start < next_frame:
        raise ValueError('Clip boundaries must be in ascending order.')
    frames = []
    wanted = {start + i for i in OFFSETS}
    while next_frame < start + 48:
        if not cap.grab():
            raise RuntimeError(f'Video ended or failed to decode at frame {next_frame}.')
        if next_frame in wanted:
            ok, bgr = cap.retrieve()
            if not ok or bgr is None:
                raise RuntimeError(f'Cannot retrieve frame {next_frame}.')
            bgr = cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        next_frame += 1
    if len(frames) != 6:
        raise RuntimeError('Expected six frames per clip.')
    return np.stack(frames), next_frame


def save_mp4(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 3.0, (512, 288))
    if not writer.isOpened():
        raise RuntimeError(f'MP4 writer unavailable: {path}')
    try:
        for rgb in frames:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f'MP4 creation failed: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    video_dir = root / 'SEED-DV' / 'Video'
    mapping = root / 'outputs' / 'segment_raw_signals_200Hz' / 'sample_mapping.csv'
    out = root / 'outputs' / Path(__file__).stem
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with mapping.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1400 or len({r['sample_id'] for r in rows}) != 1400:
        raise ValueError('Expected 1,400 unique rows from EEG segmentation.')
    captions, metadata = {}, []
    # Validate every input before starting expensive decoding.
    for b, name in enumerate(NAMES, 1):
        candidates = [p for p in (video_dir / 'BLIP-caption').rglob('*.txt')
                      if p.name == name + '.txt']
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one BLIP-caption/{name}.txt; found {len(candidates)}.')
        lines = candidates[0].read_text(encoding='utf-8-sig').splitlines()
        if len(lines) != 200 or any(not line.strip() for line in lines):
            raise ValueError(f'{candidates[0]} must contain exactly 200 nonempty caption lines.')
        captions[b] = [line.strip() for line in lines]
        cap = cv2.VideoCapture(str(video_dir / (name + '.mp4')))
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or count != 12480 or not 23.9 <= fps <= 24.1:
                raise ValueError(f'Unexpected video metadata for {name}: frames={count}, fps={fps}')
            metadata.append(dict(block=b, filename=name+'.mp4', frames=count, fps=fps, seconds=count/fps))
        finally:
            cap.release()
    for index, r in enumerate(rows):
        b, clip = divmod(index, 200)
        start = (clip // 5 * 13 + 3 + clip % 5 * 2) * 24
        expected_id = f'sub01_session01_block{b+1:02d}_clip{clip+1:03d}'
        if (int(r['block']) != b+1 or int(r['clip_in_block']) != clip+1
                or r['sample_id'] != expected_id or int(r['caption_line_1based']) != clip+1
                or int(r['expected_video_start_frame']) != start
                or int(r['expected_video_end_frame_exclusive']) != start+48
                or r['split'] != ('train' if b < 6 else 'test')):
            raise ValueError(f'Unexpected sample order or boundary in row {index}.')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Results already exist: {out}. Rename that folder before rerunning.')
    if shutil.disk_usage(root).free < 4.5 * 1024**3:
        raise RuntimeError('At least 4.5 GiB free disk space is required.')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'previews').mkdir()
    started = time.perf_counter()
    paired = []
    print('Preparing 1,400 clips. RGB arrays are canonical model inputs; MP4s are viewing copies.', flush=True)
    for b, name in enumerate(NAMES, 1):
        frame_dir = out / 'frames' / f'block{b:02d}'
        movie_dir = out / 'videos' / f'block{b:02d}'
        frame_dir.mkdir(parents=True); movie_dir.mkdir(parents=True)
        cap = cv2.VideoCapture(str(video_dir / (name+'.mp4')))
        next_frame = 0
        try:
            for local_index, r in enumerate(rows[(b-1)*200:b*200]):
                start = int(r['expected_video_start_frame'])
                frames, next_frame = read_clip(cap, next_frame, start)
                stem = r['sample_id']
                array_path, movie_path = frame_dir/(stem+'.npy'), movie_dir/(stem+'.mp4')
                np.save(array_path, frames, allow_pickle=False)
                save_mp4(movie_path, frames)
                entry = dict(r)
                entry.update(caption=captions[b][local_index], source_video=(video_dir/(name+'.mp4')).relative_to(root).as_posix(),
                    rgb_array=array_path.relative_to(root).as_posix(), video_preview=movie_path.relative_to(root).as_posix(),
                    sampled_source_frames=';'.join(str(start+i) for i in OFFSETS))
                paired.append(entry)
                if local_index == 0:
                    fig, axes = plt.subplots(2, 3, figsize=(12, 5), constrained_layout=True)
                    for j, ax in enumerate(axes.flat):
                        ax.imshow(frames[j]); ax.axis('off'); ax.set_title(f'Source frame {start+OFFSETS[j]}', fontsize=9)
                    fig.suptitle(f'Block {b}, clip 1 | '+captions[b][0], fontsize=11, wrap=True)
                    fig.savefig(out/'previews'/f'block{b:02d}_clip001.png', dpi=140)
                    plt.close(fig)
                done = len(paired)
                if done % 10 == 0:
                    elapsed = time.perf_counter()-started
                    print(f'\rClips {done}/1400 | {done/14:.1f}% | elapsed {elapsed/60:.1f} min | ETA {elapsed/done*(1400-done)/60:.1f} min', end='', flush=True)
        finally:
            cap.release()
    with (out/'paired_samples.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    summary = dict(status='complete', samples=1400, train=1200, test=200,
        rgb_array_shape=[6,288,512,3], dtype='uint8', colour='RGB', output_fps=3,
        source_videos=metadata, sample_offsets=list(OFFSETS),
        method='Skip 72 instruction frames per group; five clips of 48 source frames; sample every eighth frame.',
        timing_note='Nominal 24-FPS layout follows 12,480 source frames. Reported source FPS is retained above; this does not independently prove physiological synchronisation.',
        captions='Released BLIP captions, paired in original presentation order; no class sorting.',
        model_input='Use RGB .npy arrays for target extraction and generator training to avoid MP4 recompression.',
        runtime_seconds=round(time.perf_counter()-started,2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'\nComplete: {out}\n1,200 training clips + 200 test clips. Paired manifest: paired_samples.csv')


if __name__ == '__main__':
    main()
