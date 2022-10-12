# Video Noise Reduction

A simple and fast network for Deep Video Denoising Algorithm.

## Code User Guide

### Testing
```
test_vnr.py \
	--in_video <path_to_input_video> \
	--out_video <path_to_output_video> \
	--noise_sigma 40 \
	--model <model_path>
```

### Training

```
train_vnr.py \
	--trainset_dir <path_to_input_mp4s> \
	--valset_dir <path_to_val_sequences>
```