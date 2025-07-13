<img width="1481" height="761" alt="image" src="https://github.com/user-attachments/assets/c178b3da-1408-4c73-8373-c643ccc527d6" />

GAN for unpaired image style transfer

The goal for this project was to build a Generative Adversarial Network to turn unpaired photos into Monet paintings. \
Included are 2 different architectures and approaches to solving this problem. \
1. gen2.py: This model was an implementation of CycleGAN architecture.
CycleGAN is a type of GAN architecture that is characterized by using 2 generators and discriminators instead of a standard GAN's 1. Generator A->B translates domain A (photos) to domain B (Monet). 
Generator B->A translates domain B to domain A. Then, discriminators A and B learn to distinguish generated images from the real ones, respectively. 
This architecture is strung together with <b>cycle-consistency loss</b>, which enforces the idea that if we translate an image to the alternate domain, and back again, it should be as similar as possible. 
Thus, the adversarial nature of GAN architecture; generators A & B learn to produce realistic generations that fool the discriminators, while discriminators A & B learn to better distinguish generations from the real domain. This loop drives improvement, while cycle-consistency loss ensures the nature of the original domain is maintained. \

From this explanation, the loss calculations are intuitive:
<img width="512" height="110" alt="image" src="https://github.com/user-attachments/assets/d8fe05ec-e87a-47b3-9c9e-b8fc9f2cdd6a" />
via ([https://arxiv.org/abs/1703.10593](CycleGAN (Zhu et al., 2017))
Adversarial loss for $G_(x->y)$



   
3. gen3.py: This model used a pretrained text-to-image model CLIP to provide inherent semantic guidance to style the images. 



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


Conclusion for iteration 2.
This iteration's architecture was comprised of using a U-Net generator paired with the text-to-image translator CLIP as a feature extractor for the source photos.
Incorporated loss from CLIP monet style loss + CLIP content loss of original photo, and adversarial loss from the PatchGAN discriminator.
We fell short on this iteration. I found it very challenging to balance style and content loss, reduce artifacts in the generations, maintaing monet brush style.


