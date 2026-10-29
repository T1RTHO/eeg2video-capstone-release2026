# External data and metadata requirements

No EEG or source video data is included. Obtain access to the relevant SEED-DV material through its authorized source and comply with its access and redistribution terms. No download URL is supplied here because the permitted source and terms have not been verified for redistribution.

## Inputs required

- Subject 01 EEG array and concept labels for all seven blocks.
- Seven source video files and their caption text files.
- `channel_62_pos.locs` with the channel ordering recorded by the final predictor runs.
- Dynamic predictor metadata arrays corresponding to the official labels and optical-flow scores. They are referenced by `EEG2Video/EEG2Video_New/Dynamic/source_manifest.json` but intentionally not bundled pending redistribution review.
- The generated sample mapping/paired manifest (`outputs/prepare_video_clips/paired_samples.csv`), which ties EEG sample IDs to captions, source frames and RGB arrays. The final run records this manifest SHA-256:
  `a5a258cb228247d29aed8afd38bbf4780aa51df70b3a8b9233baf525e7f4f6b8`.

The scripts expect a project root containing the local `SEED-DV/` tree and write intermediates under `outputs/`. Those large or sensitive paths are excluded by `.gitignore`.

## Data lineage and cautions

EEG segmentation uses 200 Hz signals and presentation order. Video pairing samples six frames at nominal 24 fps, while source videos report 23.976 fps. This does not independently establish physiological synchronization. Never use Block 7 target latents/text as generation conditions; they are references for evaluation only.

The dynamic label metadata has source checksums in `source_manifest.json`. That manifest is retained, but its source arrays are not. Restore only copies obtained from an authorized source and verify the checksums before running dynamic training.
