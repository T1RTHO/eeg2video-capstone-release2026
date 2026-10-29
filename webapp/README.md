# Local EEG2Video Gradio application

This app runs on the local machine and binds only to `127.0.0.1`. It has two deliberately distinct paths:

- **View Existing Result** strictly matches an uploaded EEG sample to the canonical Block 7 array and retrieves its completed generation. No predictor or diffusion model runs in this mode.
- **Generate Live Video** runs the three saved EEG predictors, verifies known-sample predictions against the saved Block 7 predictions, and then performs a new single-sample generation. It does not call any training entry point.

The app does not download models, change the research pipeline, write into the configured read-only asset directory, or publish a public endpoint. There is no default to the original development directory.
The interface presents EEG controls beside the ground-truth/generated video comparison and four recorded evaluation metrics. The 200 Hz input is fixed and still validated by the backend; the unmatched-sample seed is under Advanced Settings. Readiness reflects the startup preflight, and detailed asset, stage, run and evaluation data is available in the collapsed Advanced Details / Diagnostics section.


## Requirements and separate environments

Use separate Python 3.11 environments. The tested UI was Python 3.11.16, Gradio 5.50.0, and NumPy 2.4.6. The tested worker was Python 3.11.16, PyTorch 2.2.2+cu118, CUDA 11.8, Diffusers 0.11.1, Transformers 4.26.1, xFormers 0.0.25.post1+cu118, NumPy 1.26.4, and imageio-ffmpeg 0.5.1. These exact builds are a tested configuration, not a guarantee for other Windows, Python, driver, or GPU combinations.

1. Create the frontend environment:

   ```powershell
   py -3.11 -m venv webapp\.venv
   .\webapp\.venv\Scripts\Activate.ps1
   python -m pip install -r webapp\requirements.txt
   ```

2. Create a separate worker environment. Install CUDA-matched PyTorch and xFormers first, then the pinned packages in [`worker-requirements.txt`](worker-requirements.txt). Do not combine this with the older preserved research requirements; those requirements describe different software generations.
3. Configure the worker interpreter before launching. Use the actual path on your machine, for example:

   ```powershell
   $env:EEG2VIDEO_WORKER_PYTHON = "C:\path\to\worker-env\python.exe"
   ```

   The UI probes required worker modules without importing PyTorch. The launcher requires this variable. The app disables a mode if its required files or packages are missing; file presence does not replace the worker's hash, mapping, provenance and runtime checks.

Do not install the webapp worker requirements into the original training environment in place. If you create a new worker environment, install a CUDA-matched PyTorch/xFormers combination first and validate it with a small local preflight.

## Private asset directory and configuration

Set `EEG2VIDEO_ASSETS_DIR` to the private asset directory on your computer. The app requires this variable and the `asset_config.json` created there by the preparation utility. All external paths in the configuration are relative to that directory, are checked against path traversal, and cannot resolve outside it. There is no fallback to another folder.

The public template [`asset_config.example.json`](asset_config.example.json) contains only portable relative paths. For an unusual layout, edit a private copy in the asset directory or set `EEG2VIDEO_ASSET_CONFIG` to a local JSON configuration file; keep its values relative to the configured asset root. The preparation utility is dry-run by default:

```powershell
python tools\prepare_webapp_assets.py --source-root <authorized-source-folder> --assets-dir <private-assets-folder> --sd14-source <local-sd14-snapshot>
python tools\prepare_webapp_assets.py --source-root <authorized-source-folder> --assets-dir <private-assets-folder> --sd14-source <local-sd14-snapshot> --copy
```

Only the second command copies. It copies an explicit allowlist, verifies hashes, reports omissions/mismatches, never deletes or moves source files, and never overwrites a destination. The resulting asset manifest and copied model/data artifacts belong outside Git. If redistribution permission is unclear, retain them privately and resolve permission before publication.

| Variable | Default | Purpose |
|---|---|---|
| `EEG2VIDEO_ASSETS_DIR` | **Required** | Private root for all external runtime files, including checkpoints, data, generated arrays, the verified negative embedding, and the local VAE snapshot. |
| `EEG2VIDEO_ASSET_CONFIG` | `<assets-dir>\asset_config.json` | Optional path to a private JSON config; every asset path remains relative to `EEG2VIDEO_ASSETS_DIR`. |
| `EEG2VIDEO_WORKER_PYTHON` | UI interpreter when running directly; required by launcher | Worker Python executable with compatible inference packages and CUDA support. |
| `EEG2VIDEO_CACHE_DIR` | `webapp/.cache` | Per-run uploads, predictions, MP4s, logs and metadata. Must not overlap the private read-only asset directory. |
| `HF_HOME` | Hugging Face default | Optional cache location for separately acquired, authorized model snapshots. Webapp inference uses local paths and offline loading; it does not fetch models. |
| `EEG2VIDEO_CPU_THREADS` | `4` | Worker CPU threads for prediction. |
| `EEG2VIDEO_PORT` | `7860` | Local browser port. |

Uploads and new outputs are copied to a per-run `.cache/runs/<UUID>/` folder, which is ignored by Git and kept outside the configured asset root. UUID run folders older than seven days are cleaned on app startup.

Precomputed mode needs canonical Block 7 EEG and channel metadata, both mappings, all 200 Block 7 reference RGB arrays, all existing generated RGB arrays and completion records, and the completed per-clip metrics. Live mode additionally needs the three exact predictor checkpoints and normalizers, their Block 7 prediction arrays/mappings, the final temporal UNet, scheduler and provenance, the exact `FinalEvaluation/final_negative.npy`, and a local Stable Diffusion 1.4 VAE snapshot at the recorded revision. The app verifies recorded hashes and provenance at runtime; it never substitutes the alternate float32 empty-caption embedding.

## Start the application (Windows)

From PowerShell at the release root, set paths for your own machine and run the checked launcher:

```powershell
$env:EEG2VIDEO_ASSETS_DIR = "<private-assets-folder>"
$env:EEG2VIDEO_WORKER_PYTHON = "<worker-environment>\python.exe"
.\run_webapp.bat
```

Create the frontend environment and configure the asset directory before launch. The launcher checks both environments and the asset config, installs nothing, and downloads nothing. Open `http://127.0.0.1:7860`. The app always binds to loopback, disables Gradio sharing, and does not expose a public endpoint.

## Input format and validation

Upload one non-pickled `.npy` file, no larger than 64 MiB, containing:

- one sample `(62, 400)`;
- an optional singleton batch `(1, 62, 400)`; or
- the exact canonical test array `(200, 62, 400)`, with the selected index.

The signal is 62 channels over 400 consecutive samples (2 seconds at 200 Hz). State `200 Hz` and confirm that the channel order matches the project's recorded `channel_62_pos.locs`. The app checks that local file against its verified SHA-256. It rejects nonnumeric, complex, non-finite, object, wrong-shape, wrong-rate and reordered/unconfirmed input. It never silently reshapes or resamples.

For single-sample uploads, identification compares the actual array values to all 200 canonical Block 7 signals; the filename is not used. A full-array upload is accepted only if its values exactly equal the verified canonical test array. Precomputed mode fails on unmatched signals rather than showing an unrelated example. Live mode can continue for an unmatched *single* sample, but uses the user-entered seed and shows no ground-truth video or metrics.

## Execution modes

### View Existing Result

Requires the canonical test EEG, EEG and paired-video mappings, channel-order file, final per-clip metrics, completed RGB outputs and paired ground-truth frames. For an exact match, it:

1. resolves the sample ID from the mapping;
2. checks completion provenance and generated/reference SHA-256 hashes;
3. reads the matching stored RGB arrays;
4. verifies the saved metric row against the arrays; and
5. writes two separate six-frame, 512×288, 3-fps MP4s in that run's cache.

The metrics are retrieved from the completed evaluation record. **No model inference time is reported** and no model inference is performed.

### Generate Live Video

Requires all three exact checkpoints and training normalisations, saved Block 7 prediction arrays/mappings for known-sample validation, final diffusion UNet and scheduler, local SD 1.4 VAE, verified final negative embedding, and the CUDA worker dependencies. Known Block 7 inputs use seed `42 + test_index`; the app reports numerical prediction differences against the saved predictions and stops before diffusion if any fail the recorded tolerances. It also checks the released500 timestep, noise hash, alpha and scheduler timestep against the completed run. The initial-latent hash may differ because live inference uses freshly recomputed predictions; the app reports the numeric difference from the historical initial latent without replacing live predictions with saved values. An unmatched single input uses the explicit seed field and is never assigned an invented Block 7 index.

The implementation reuses the release's predictor definitions and feature function:

- visual: seven 100-sample windows at stride 50; original training-fitted normalisation; output `(6, 4, 36, 64)`;
- semantic: original `full_window_400fft` DE features and normalisation; CPU float32 MLP; output `(77, 768)`;
- dynamic: raw-signal, training-fitted normalisation and original GLMNet; `0=slow`, `1=fast`.

Predictions run sequentially on CPU; the large semantic network is not moved to GPU. Diffusion runs separately on CUDA using the completed experiment's six frames, 512×288 resolution, 3 fps, 100 DDIM steps, guidance 12.5, released500 DANA noise, beta mapping (fast 0.2 / slow 0.3), float16 UNet, xFormers attention and CPU framewise float32 VAE decode. It does not silently reduce settings or switch attention implementations. If CUDA memory or a required artifact is unavailable, the job reports a failure instead.

Ground truth is only loaded after predictions, diffusion and VAE decoding are complete. For a known sample, metrics are computed for the **new output**, and a separate pixel-difference summary compares it with the existing generated array. Historical metrics are never presented as live metrics. For unmatched samples, reference metrics are unavailable.

## Hardware and limitations

The tested worker uses an NVIDIA RTX 3060 Laptop GPU with about 6 GiB VRAM and a 32 GiB host. Live generation requires the verified xFormers path and at least 4.5 GiB free GPU memory before starting. CPU semantic inference temporarily loads a roughly 3.58 GB checkpoint; allow several GiB of free RAM. No guarantees are made for other GPUs, drivers, model caches or library builds.

This is a one-subject/session capstone adaptation. Block 7 has been inspected previously and is not a pristine blind test. The negative embedding is verified by its hash against the final generation record, but its derivation provenance remains unresolved; the app refuses any different embedding and the repository documentation retains that caveat. The application does not assert redistribution rights for EEG, video, model weights or third-party code.

## Troubleshooting

- **Mode disabled:** review the readiness list in the app. Set `EEG2VIDEO_ASSETS_DIR` and `EEG2VIDEO_WORKER_PYTHON`, then confirm `asset_config.json` and the required files exist in the private asset folder.
- **Hash mismatch:** restore the exact verified artifact; do not substitute a similarly named checkpoint, manifest, channel file or embedding.
- **CUDA/xFormers failure:** use the worker environment matching the verified CUDA/PyTorch/xFormers builds, close other GPU workloads and confirm free VRAM. The app will not switch to a different generation configuration.
- **Precomputed sample not found:** upload the exact canonical EEG sample values. A filename or approximate signal match is not sufficient.
- **Unmatched live sample:** provide a real 62×400 sample in the recorded channel order. It may be generated without reference metrics; the UI will not claim Block 7 identity.
- **Port in use:** set `EEG2VIDEO_PORT` to another local port.
