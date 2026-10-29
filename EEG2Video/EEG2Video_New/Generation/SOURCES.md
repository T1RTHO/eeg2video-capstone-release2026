# Source provenance

Official repository: https://github.com/XuanhaoLiu/EEG2Video
Reference commit: 18af805753bc89e796625cdaaca1d0844bee22a8

vendor/unet.py, unet_blocks.py, attention.py and resnet.py retain the upstream EEG2Video/models
architecture source without architecture changes (line endings normalized).
Original files credit adaptations from Hugging Face Diffusers; preserve their notices.
Consult upstream repositories for applicable license terms; no new license is asserted here.

Training integration follows train_finetune_videodiffusion.py's trainable modules,
posterior sampling, DDPM noise objective and optimizer settings. Dataset access and
execution/checkpoint handling are new integration code.

Generation follows inference_eeg2video.py's dimensions, 100 steps and guidance 12.5.
The DANA beta mapping follows the paper rather than newer add_noise.py.
Scheduler consistency choices and incomplete exact-replication details are in README.md.

Evaluation follows 40_class_run_metrics.py's SSIM, ViT and VideoMAE top-3 pseudo-label
N-way methods. The implementation loads classifiers once, pairs clips by sample ID and
uses lossless arrays rather than class-sorted GIF filenames.

Paper: https://papers.nips.cc/paper/2024/file/84bad835faaf48f24d990072bb5b80ee-Paper-Conference.pdf
Legacy scheduler API: https://huggingface.co/docs/diffusers/v0.11.0/api/schedulers/ddim
Torch installation: https://pytorch.org/get-started/previous-versions/
