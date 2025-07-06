Failures of cyclegan:
- With limited monet paintings to train on (300), overfitting was a big issue and excessive tuning had depreciating results. Would be very hard to get a top score with this architecture.

New architecture:
1. Generator pretrained on larger dataset. Fine tuned with Monet paintings. (transfer learning)
2. Pretrained/shallow discriminator with low LR.
3. Subsequent GANs to further stylize images.
4. Diffusion process to discriminator's input as a regularization step.
5. Training loop: Transfer learning w/ pretrained weights, train GANs sequentially.

Issue: Complete mode collapse. All images look identical with a pixel difference of <0.0001
Solved: U-net which utilizes skip connection to better maintain information from origin photo.
https://arxiv.org/abs/1701.07875

Issue: Grainy/gridlike textures.
fix:

Issue: generations are too similar to the image, lacking monet style. Also, black hole artifacts appearing.
fix: ReLU -> LeakyReLU, reduce content preservation metric, raise monet style incentive
