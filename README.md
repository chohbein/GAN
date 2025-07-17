<img width="1481" height="761" alt="image" src="https://github.com/user-attachments/assets/c178b3da-1408-4c73-8373-c643ccc527d6" />

# GAN for unpaired image style transfer

The goal for this project was to build a Generative Adversarial Network to turn unpaired photos into Monet paintings.
Included are 2 different architectures and approaches to solving this problem.  
1. gen2.py: This model was an implementation of CycleGAN architecture.  
CycleGAN is a type of GAN architecture that is characterized by using 2 generators and discriminators instead of a standard GAN's 1. Generator A->B translates domain A (photos) to domain B (Monet). 
Generator B->A translates domain B to domain A. Then, discriminators A and B learn to distinguish generated images from the real ones, respectively. 
This architecture is strung together with <b>cycle-consistency loss</b>, which enforces the idea that if we translate an image to the alternate domain, and back again, it should be as similar as possible. 
Thus, the adversarial nature of GAN architecture; generators A & B learn to produce realistic generations that fool the discriminators, while discriminators A & B learn to better distinguish generations from the real domain. This loop drives improvement, while cycle-consistency loss ensures the nature of the original domain is maintained. 
<br>
## From this explanation, the loss calculations are intuitive: <br>
<img width="512" height="110" alt="image" src="https://github.com/user-attachments/assets/d8fe05ec-e87a-47b3-9c9e-b8fc9f2cdd6a" />  <br>
via https://arxiv.org/abs/1703.10593  <br>
Adversarial loss for $G_{x,y}$, G is trying to generate images that look like domain B, while D tries to distinguish between fake and real images of domain B.  
The same would be for $G_{y,x}$.  
<br>
<img width="397" height="77" alt="image" src="https://github.com/user-attachments/assets/d4fcdf82-143c-48a6-9323-99263b407568" />  
Cycle-consistency loss here is measured by a forward and backward term. The forward term takes an image from domain A and translates it to domain B, then back to A. Then, the loss is measured by how similar the reconstructed image is to its original. This uses L1 normalizaton to focus on pixel similarity.  
The backward term does the same but this time from domain B, to A, back to B, then calculates the difference.  
<br><br>
Full loss function for CycleGAN: <img width="382" height="116" alt="image" src="https://github.com/user-attachments/assets/15afe4a3-c8bc-42ea-b252-f4cf1b374549" />
I also used identity loss, something that was brought up further in the CycleGAN paper. The authors suggested this as a way to better retain color composition

## Changes made
1. Added spectral normalization to initial layers of the discriminators.
- Quickly into training, discriminator losses were unusually low while the MiFID remained high. This suggested the discriminators were greatly overpowering the generators, potentially due to exploding gradients. We applied spectral normalization to the initial layers to restrict the gradients and stabilize learning.

2. Regularization: Added label smoothing.
- This was done to further stress the discriminators so it wouldn't overpower the generators. The generations were beginning to lose their uniqueness which was indicative of overwhelming discriminators. Label smoothing helps this by creating slight uncertainty in the discriminators' classifications of real and fake.

3. Regularization: Added noise
- Added a small amount of random noise to the real and fake images before feeding them to the discriminators. Just another step to make them work harder and stick to generalizations.

4. Delayed discriminator updates
- Discriminator weights are updated once for every 2 generator updates. Helped early in training to limit discriminators.

5. Mixed precision
- Mixed precision was very helpful and was used in every iteration of this project. Automatic Mixed Precision (AMP) was vital in reducing stress on my laptop GPU memory. It works by performing calculations with less precision when possible.

6. LR scheduling
- Found success in setting a decay on the learning rate over epochs. Additionally, initial learning rate for discriminators were 5e-5 while generates were 2e-4. This helped to nerf the discriminators.



4. gen3.py: This model used a pretrained text-to-image model CLIP to provide inherent semantic guidance to style the images. 



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


