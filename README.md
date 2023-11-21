# Diffusers DDIM Inversion example
A simple example of using `DDIMInverseScheduler` for inverting an input image to StableDiffusion's latent space

## Usage
Just run
```
python ddim_inversion.py
```
The code loads the `poike.png` example image, and uses `DDIMInverseScheduler` to invert it to the noisy latent and reconstruct it back.

### Notes
Reconstruction is not perfect -- you can add more diffusion steps.
