# GAN for unpaired image style transfer
<br><br>
<img width="1481" height="761" alt="image" src="https://github.com/user-attachments/assets/c178b3da-1408-4c73-8373-c643ccc527d6" />

The goal for this project was to build a Generative Adversarial Network to turn unpaired photos into Monet paintings.
Included are 2 different architectures and approaches to solving this problem.  
## gen2.py
This model was an implementation of CycleGAN architecture.  
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
<img width="397" height="77" alt="image" src="https://github.com/user-attachments/assets/d4fcdf82-143c-48a6-9323-99263b407568" />  <br>
Cycle-consistency loss here is measured by a forward and backward term. The forward term takes an image from domain A and translates it to domain B, then back to A. Then, the loss is measured by how similar the reconstructed image is to its original. This uses L1 normalizaton to focus on pixel similarity.  
The backward term does the same but this time from domain B, to A, back to B, then calculates the difference.  
<br><br>
Full loss function for CycleGAN: <br>
<img width="382" height="116" alt="image" src="https://github.com/user-attachments/assets/15afe4a3-c8bc-42ea-b252-f4cf1b374549" /> <br>
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

6. Learning Rate Management
- Found success in setting different initial learning rates to balance generators and discriminators. Discriminators used a lower rate of 5e-5 with generators at 2e-4.
- Additionally, decaying both learning rates over epochs was helpful. Implemented resumed decay when returning to training from a checkpoint.


Failures of cyclegan:
- With limited monet paintings to train on (300), overfitting was a big issue and excessive tuning had diminishing returns. Would be very hard to get a top score with this architecture.
- ResNet struggled to capture accurate brush strokes and textures of the Monet style.

## gen3.py
U-net with CLIP-feature extraction
This approach used a U-Net generator. This change was made because I learned it is better at generating details within the images because it uses skip-connections to pass high-resolution feature maps through the encoder. This will ensure we maintain the sharp edges and detailed brush strokes of the Monet style. 
Furthermore, I used Wasserstein GAN with Gradient Penalty for the discriminator. Replacing our CycleGAN discriminators with a single WGAN-GP critic seemed to help my previous issue of discriminators overpowering the generators. 
I also used CLIP's pretrained weights to extract the features of the image; mountains, rivers, etc. However, I froze the weights as to not train it further, just using CLIP to extract feature vectors. This was helpful; instead of forcing the generator to learn to recognize these features, we just supply it with that information. Additionally, I used CLIP to calculate perceptual loss. This guided the generator to not just create visually similar paintings, but ones that maintain the descriptive nature of the original photo.



Loss <br>
<img width="447" height="168" alt="image" src="https://github.com/user-attachments/assets/6ab06dcc-ffdc-4c5d-bce9-00df89a6e969" /> <br>
g_clip_style: Style loss. Compares CLIP embeddings of generated image to that of a Monet painting.
g_clip_content: Content loss. compares generated image to original photo, ensuring we maintain the nature of the photo.
g_adv: Adversarial loss. The primary metric for learning to create good paintings. Calculated by passing the fake Monet through the critic. MSE measures how far the critic's output is from the standard for real Monets.
loss_identity: Identity loss. Calculated with the MAE of a real Monet and it's translated version after being passed through the generator.


## Changes made
1. Complete mode collapse. All images look identical with a pixel difference of <0.0001
- Implemented WGAN-GP, replacing the discriminator with a "critic" that scores the realness instead of classifying it as real or fake. Also adds a gradient penalty to the critic's loss that stabilizes training.
- I asymetrically updated the critic, once for every 5 generator steps, to ensure the critic is giving strong and reliable updates.
- Inserted noise into the generator to give some slight disparity for all generations.
https://arxiv.org/abs/1701.07875

2. Gridlike/checkerboard artifacts.
- Added L1 identity loss to the loss function.
- Tweaked upsampling; replaced ConvTranspose2d with nn.Upsample + nn.Conv2d. This helps because ConvTranspose2d applies filters in a way that unevenly overlaps pixels.

2. generations are too similar to the image, lacking monet style. Also, black hole artifacts appearing (dying neurons).
- ReLU -> LeakyReLU in discriminator and generator downsampling.
- reduce content preservation hyperperameter, raise monet style incentive

3. Exploding gradients/NAN losses
- Gradient clipping. Drastically helped the issue by setting a threshold for the gradients that scales them down proportionally when exceeded.
- Spectral Normalization to all layers of critic. Helps to constrain weights.
- GradScaler for Mixed Precision may also have helped as the range of numbers for half precision is much smaller.


Conclusion for iteration 2.
For this iteration, I found it very challenging to balance style and content loss, reduce artifacts in the generations, maintaing monet brush style. Despite all this, we saw a relatively similar score to iteration 1 in less time. Training for longer may give better results.


