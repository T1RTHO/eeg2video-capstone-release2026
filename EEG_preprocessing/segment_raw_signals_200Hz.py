"""Subject 1 EEG segmentation adapted from XuanhaoLiu/EEG2Video.
Run from the project root: python EEG_preprocessing/segment_raw_signals_200Hz.py
Dependencies: numpy, matplotlib. No filtering, normalisation or video extraction.
Official timing: skip 3 s, retain five 2 s clips, repeat 40 times per block.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np


def segment_block(block):
    """Preserve presentation order: (40 groups, 5 clips, 62 channels, 400 samples)."""
    if block.shape != (62, 104000):
        raise ValueError(f'Unexpected block shape: {block.shape}')
    result = np.empty((40, 5, 62, 400), dtype=block.dtype)
    for group in range(40):
        for clip in range(5):
            start = group * 2600 + 600 + clip * 400
            result[group, clip] = block[:, start:start + 400]
    if not np.isfinite(result).all():
        raise ValueError('EEG contains NaN or infinity in retained clips.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Project folder containing SEED-DV')
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'SEED-DV' / 'EEG' / 'sub1.npy'
    label_path = root / 'SEED-DV' / 'Video' / 'meta-info' / 'All_video_label.npy'
    out = root / 'outputs' / Path(__file__).stem
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eeg = np.load(source, mmap_mode='r', allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False)
    if eeg.shape != (7, 62, 104000) or not np.issubdtype(eeg.dtype, np.floating):
        raise ValueError(f'Unexpected EEG shape/type: {eeg.shape}, {eeg.dtype}')
    if labels.shape != (7, 40) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'Unexpected label shape/type: {labels.shape}, {labels.dtype}')
    if not all(np.array_equal(np.sort(row), np.arange(1, 41)) for row in labels):
        raise ValueError('Each block must contain concept IDs 1 through 40 exactly once.')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Results already exist: {out}. Rename that result folder before rerunning.')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Preserve source dtype and the official repository output shape.
    segmented = np.lib.format.open_memmap(out / 'sub1.npy', mode='w+', dtype=eeg.dtype,
                                         shape=(7, 40, 5, 62, 400))
    rows = []
    for block in range(7):
        segmented[block] = segment_block(eeg[block])
        for group in range(40):
            for clip in range(5):
                ordinal = group * 5 + clip
                seconds = group * 13 + 3 + clip * 2
                rows.append(dict(sample_id=f'sub01_session01_block{block+1:02d}_clip{ordinal+1:03d}',
                    block=block+1, clip_in_block=ordinal+1, concept_position=group+1,
                    clip_in_concept=clip+1, concept_id=int(labels[block, group]),
                    split='train' if block < 6 else 'test', split_index=block*200+ordinal if block<6 else ordinal,
                    eeg_start_sample=seconds*200, eeg_end_sample_exclusive=(seconds+2)*200,
                    nominal_start_seconds=seconds, nominal_end_seconds=seconds+2,
                    expected_video_start_frame=seconds*24, expected_video_end_frame_exclusive=(seconds+2)*24,
                    caption_line_1based=ordinal+1))
        elapsed = time.perf_counter() - started
        done = block+1
        print(f'[{"#"*done}{"."*(7-done)}] Block {done}/7 | {elapsed:.1f}s elapsed | '
              f'{elapsed/done*(7-done):.1f}s estimated remaining', flush=True)
    segmented.flush()
    np.save(out / 'train_eeg.npy', segmented[:6].reshape(1200, 62, 400))
    np.save(out / 'test_eeg.npy', segmented[6].reshape(200, 62, 400))
    clip_labels = np.repeat(labels, 5, axis=1)
    np.save(out / 'train_labels.npy', clip_labels[:6].reshape(-1))
    np.save(out / 'test_labels.npy', clip_labels[6])
    with (out / 'sample_mapping.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    for channel in range(3):
        axes[0].plot(np.arange(2600)/200, eeg[0, channel, :2600], lw=.7, label=f'Channel index {channel}')
        axes[1].plot(np.arange(400)/200, segmented[0, 0, 0, channel], lw=.7)
    axes[0].axvspan(0, 3, color='grey', alpha=.2, label='Excluded instruction')
    for boundary in [3, 5, 7, 9, 11, 13]:
        axes[0].axvline(boundary, color='black', alpha=.25, lw=.8)
    axes[0].set_title('Subject 1, Block 1: first instruction and five clips')
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title('Extracted first clip: original block samples [600:1000]')
    for ax in axes:
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude (source units)')
        ax.grid(alpha=.15)
    fig.savefig(out / 'eeg_segmentation_preview.png', dpi=160)
    plt.close(fig)
    summary = dict(subject=1, session=1, input=str(source), input_shape=list(eeg.shape),
        dtype=str(eeg.dtype), sampling_rate_hz=200, segmented_shape=list(segmented.shape),
        train_shape=[1200,62,400], test_shape=[200,62,400], train_blocks=list(range(1,7)), test_blocks=[7],
        label_encoding='Original concept IDs 1..40; no sorting by class',
        validation='No validation split here. Reserve training data for model selection; do not tune on Block 7.',
        processing='Segmentation only; no additional filtering, downsampling, standardisation or channel reordering.',
        video_timing='Frame indices are expected boundaries under nominal 24 FPS, not measured EEG/video synchronisation. Source videos report 23.976 FPS.',
        provenance='Adapted timing from https://github.com/XuanhaoLiu/EEG2Video/blob/main/EEG_preprocessing/segment_raw_signals_200Hz.py',
        runtime_seconds=round(time.perf_counter()-started,2))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Complete: {out}\nTraining EEG: (1200, 62, 400)\nTest EEG: (200, 62, 400)')
    print('Video files and captions will be prepared in the next stage.')


if __name__ == '__main__':
    main()
