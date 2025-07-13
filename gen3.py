import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
import random
from torch.utils.data import Dataset
from itertools import cycle

from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance

from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
import clip

import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

clip_normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711]
)

class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.lower().endswith(('.jpg'))
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def split_indices(n, val_split=0.2):
    idxs = list(range(n))
    random.shuffle(idxs)
    split = int(n * (1 - val_split))
    return idxs[:split], idxs[split:]

def get_loaders(batch_size):
    # Load + Preprocessing
    img_h, img_w = 256, 256

    train_transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])



    MONET_PATH = r"C:\Users\Chohb\OneDrive\Documents\CUBoulder\Intro_to_DL\GAN\monet_jpg"
    PHOTO_PATH = r"C:\Users\Chohb\OneDrive\Documents\CUBoulder\Intro_to_DL\GAN\photo_jpg"

    # Monet
    monet_dataset = FlatImageDataset(MONET_PATH, transform=None)  # No transform yet
    monet_train_idx, monet_val_idx = split_indices(len(monet_dataset))
    monet_train = Subset(FlatImageDataset(MONET_PATH, transform=train_transform), monet_train_idx)
    monet_val = Subset(FlatImageDataset(MONET_PATH, transform=val_transform), monet_val_idx)

    # Real
    real_dataset = FlatImageDataset(PHOTO_PATH, transform=None)
    real_train_idx, real_val_idx = split_indices(len(real_dataset))
    real_train = Subset(FlatImageDataset(PHOTO_PATH, transform=train_transform), real_train_idx)
    real_val = Subset(FlatImageDataset(PHOTO_PATH, transform=val_transform), real_val_idx)

    # Take first 20 images from each train subset for overfit test
    overfit_real_train = Subset(real_train.dataset, indices=real_train.indices[:20])
    overfit_monet_train = Subset(monet_train.dataset, indices=monet_train.indices[:20])

    overfit_real_loader = DataLoader(overfit_real_train, batch_size=batch_size, shuffle=True)
    overfit_monet_loader = DataLoader(overfit_monet_train, batch_size=batch_size, shuffle=True)



    #   Loaders
    monet_train_loader = DataLoader(monet_train, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True,persistent_workers=True)
    monet_val_loader = DataLoader(monet_val, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True,persistent_workers=True)
    real_train_loader = DataLoader(real_train, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True,persistent_workers=True)
    real_val_loader = DataLoader(real_val, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True,persistent_workers=True)

    print("Here: (train,val)")
    print('photo: ', len(real_train_loader.dataset),len(real_val_loader.dataset))
    print('Monet: ', len(monet_train_loader.dataset),len(monet_val_loader.dataset))

    return monet_train_loader,monet_val_loader,real_train_loader,real_val_loader

# ===== 3. Visualization/EDA =====
def display_img(monet_loader, real_loader, num_samples=4):
    monet_batch = next(iter(monet_loader))[0][:num_samples]
    real_batch = next(iter(real_loader))[0][:num_samples]
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 4))
    for i in range(num_samples):
        img = monet_batch[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5) + 0.5
        axes[0, i].imshow(img)
        axes[0, i].set_title('Monet')
        axes[0, i].axis('off')
        img = real_batch[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5) + 0.5
        axes[1, i].imshow(img)
        axes[1, i].set_title('Real')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()



# Model Building 
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)


#   Because the monet dataset is much smaller than the photo dataset, we will use all of them then shuffle and reuse once out.
def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch

#   Gaussian noise for discriminators
def add_noise(x, noise_std=0.05):
    return x + torch.randn_like(x) * noise_std if noise_std > 0 else x

#   Label smoothing to make the discriminator's job harder.
def smooth_positive_labels(t):
    return torch.empty_like(t).uniform_(0.8, 0.9)

def smooth_negative_labels(t):
    return torch.empty_like(t).uniform_(0.0, 0.1)


#       ============CLIP weights============
def preprocess_for_clip(x):
    # x: (batch, 3, H, W) in [-1, 1]
    x = (x + 1) / 2  # to [0, 1]
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    x = clip_normalize(x)
    return x

class CLIPEncoderDecoder(nn.Module):
    def __init__(self, clip_encoder, out_channels=3, latent_dim = 100):
        super().__init__()
        self.clip_encoder = clip_encoder
        self.latent_dim = latent_dim
        self.fc = nn.Linear(512 + latent_dim, 8*8*256)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32x32
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 64x64
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # 128x128
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(16, out_channels, 4, stride=2, padding=1),  # 256x256
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        x_clip = preprocess_for_clip(x)

        with torch.no_grad():   #   freeze clip during gen
            features = self.clip_encoder.encode_image(x_clip)
        
        features = features.float() # This line converts to float32 if CLIP outputs bfloat16/float16
        #   add noise. all generations are the same, suggesting generator is avoiding diversity to fool discriminator
        z = torch.randn(batch_size,self.latent_dim).to(device)
        combined_features = torch.cat((features, z), dim=1)
        
        x = self.fc(combined_features)
        x = x.view(-1, 256, 8, 8)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x


# Helper modules for the U-Net
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(DownBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UpBlock, self).__init__()
        layers = [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

# Main U-Net Generator Class
class UNetGenerator(nn.Module):
    def __init__(self, clip_encoder, latent_dim=100):
        super(UNetGenerator, self).__init__()
        self.clip_encoder = clip_encoder
        self.latent_dim = latent_dim

        # U-Net Encoder Path
        self.down1 = DownBlock(3, 64, normalize=False) # Input: 3x256x256 -> Output: 64x128x128
        self.down2 = DownBlock(64, 128)               # -> 128x64x64
        self.down3 = DownBlock(128, 256)              # -> 256x32x32
        self.down4 = DownBlock(256, 512)              # -> 512x16x16
        self.down5 = DownBlock(512, 512)              # -> 512x8x8

        # Bottleneck where we inject CLIP and latent features
        self.bottleneck_fc = nn.Linear(512 + self.latent_dim, 512 * 8 * 8)

        # U-Net Decoder Path
        # Note the input channels are doubled because of the skip connections
        self.up1 = UpBlock(512, 512, dropout=0.5)     # Input: 512x8x8 -> Output: 512x16x16
        self.up2 = UpBlock(1024, 256, dropout=0.5)    # Input: 512+512=1024 -> 256x32x32
        self.up3 = UpBlock(512, 128)                  # Input: 256+256=512 -> 128x64x64
        self.up4 = UpBlock(256, 64)                   # Input: 128+128=256 -> 64x128x128

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 3, 3, 1, 1), # Input: 64+64=128
            nn.Tanh(),
        )

    def forward(self, x):
        # --- Encoder Path ---
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # --- Bottleneck ---
        with torch.no_grad():
            clip_features = self.clip_encoder.encode_image(preprocess_for_clip(x)).float()
        
        z = torch.randn(x.size(0), self.latent_dim).to(x.device)
        combined_bottleneck_features = torch.cat((clip_features, z), dim=1)
        
        bottleneck = self.bottleneck_fc(combined_bottleneck_features)
        bottleneck = bottleneck.view(x.size(0), 512, 8, 8)
        
        # blend the image bottleneck with the semantic bottleneck
        #   adding weight to stress importance of input image
        bottleneck = (0.2 * bottleneck + 0.8 * d5) / 2 

        # --- Decoder Path with Skip Connections ---
        u1 = self.up1(bottleneck, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.final_up(u4)


#   ==============PatchGAN=======================
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        layers = [
            spectral_norm(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        curr_dim = base_channels
        for _ in range(3):
            layers += [
                spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            curr_dim *= 2
        layers += [
            spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=4, padding=1))
            # Optionally, add nn.Sigmoid() here if using BCE loss
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def clip_loss(generated, target, clip_model, device):
    generated_features = clip_model.encode_image(generated)
    target_features = clip_model.encode_image(target)
    gen_norm = F.normalize(generated_features, p=2, dim=-1)
    tgt_norm = F.normalize(target_features, p=2, dim=-1)
    cos_sim = (gen_norm * tgt_norm).sum(dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-8, 1.0 - 1e-8)
    return 1 - cos_sim.mean()


#   ============Finalizations============
#   Score with MiFID
def compute_mifid(generator, monet_val_loader, real_val_loader, device,batch_size):
    # Generate Monet's from real photo val set
    fake_images = []
    generator.eval()
    with torch.no_grad():
        for real_batch, _ in real_val_loader:
            real_batch = real_batch.to(device)
            fake = generator(real_batch)
            fake_images.append(fake.cpu())
    fake_images = torch.cat(fake_images)
    generator.train()

    # Prep Monet val images
    monet_images = []
    for batch, _ in monet_val_loader:
        monet_images.append(batch)
    monet_images = torch.cat(monet_images)

    # Compute MiFID
    mifid = MemorizationInformedFrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for i in range(0, len(monet_images), batch_size):
        monet_batch = monet_images[i:i+batch_size].to(device)
        fake_batch = fake_images[i:i+batch_size].to(device)
        #   Convert to [0,1] before every update
        mifid.update((monet_batch + 1) / 2, real=True)
        mifid.update((fake_batch  + 1) / 2, real=False)
    return mifid.compute().item()


#   Display sample generations.
def show_generated_samples(generator, input_images, device, num_samples=4):
    generator.eval()
    with torch.no_grad():
        input_images  = input_images.to(device)
        fake_images   = generator(input_images)

        if fake_images.size(0) > 1:
            diff = torch.mean(torch.abs(fake_images[0] - fake_images[1]))
            print(f"DEBUG: Average pixel difference between first two generated images: {diff.item()}")

    # bring both tensors to CPU and denormalise from [-1,1] → [0,1]
    input_images = ((input_images.cpu() + 1) / 2).clamp(0, 1)
    fake_images  = ((fake_images.cpu()  + 1) / 2).clamp(0, 1)

    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))

    for i in range(num_samples):
        # original photo
        axes[0, i].imshow(input_images[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title("Photo")
        axes[0, i].axis("off")

        # generated Monet
        axes[1, i].imshow(fake_images[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title("Monet")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()
    generator.train()   # optional: return G to training mode



#   =========================Train===============================
def train_gan(generator, discriminator, real_loader, monet_loader, clip_model, device, g_scheduler, d_scheduler, scaler,
              num_epochs=50, lr=1e-4, lambda_clip_style=5.0, lambda_clip_content=2.0,
              g_opt=None, d_opt=None, start_epoch=0, best_mifid=float('inf'),
              batch_size=4, monet_val_loader=None, real_val_loader=None, n_critics = 5):

    adversarial_criterion = nn.MSELoss()
    generator.train()
    discriminator.train()
    best_mifid_so_far = best_mifid

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(zip(real_loader, cycle_loader(monet_loader)), total=len(real_loader))
        for i,((real_imgs, _), (monet_imgs, _)) in enumerate(pbar):
            monet_imgs = monet_imgs[:real_imgs.size(0)] #   Ensure matching
            real_imgs = real_imgs.to(device)
            monet_imgs = monet_imgs.to(device)

            # --- Train Discriminator ---
            with autocast():
                fake_monet = generator(real_imgs).detach()
                d_real = discriminator(monet_imgs)
                real_loss = adversarial_criterion(d_real, torch.ones_like(d_real))
                d_fake = discriminator(fake_monet)
                fake_loss = adversarial_criterion(d_fake, torch.zeros_like(d_fake))
                
                # Gradient Penalty
                alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to(device)
                interpolates = (alpha * monet_imgs + (1 - alpha) * fake_monet).requires_grad_(True)
                d_interpolates = discriminator(interpolates)
                gradients = torch.autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(d_interpolates),
                    create_graph=True,
                    retain_graph=True
                )[0]
                grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1) + 1e-8
                gradient_penalty = ((grad_norm - 1) ** 2).mean()
                
                d_loss = real_loss + fake_loss + 10.0 * gradient_penalty

            d_opt.zero_grad()
            #   scale + backprop
            scaler.scale(d_loss).backward()
            scaler.unscale_(d_opt)  #   Unscale before clipping! 
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)    #   gradient clipping
            scaler.step(d_opt)
            scaler.update()

            # --- Train Generator (only n_critics)---
            g_loss_val = 'skipped'
            if (i+1) % n_critics == 0:
                with autocast():
                    fake_monet = generator(real_imgs)
                    d_fake = discriminator(fake_monet)
                    g_adv = adversarial_criterion(d_fake, torch.ones_like(d_fake))

                    # CLIP losses
                    fake_monet_clip = preprocess_for_clip(fake_monet)
                    monet_imgs_clip = preprocess_for_clip(monet_imgs)
                    real_imgs_clip = preprocess_for_clip(real_imgs)
                    g_clip_style = clip_loss(fake_monet_clip, monet_imgs_clip, clip_model, device)
                    g_clip_content = clip_loss(fake_monet_clip, real_imgs_clip, clip_model, device)

                    #   Identity loss (to fix gridlike artifacts)
                    same_monet = generator(monet_imgs)
                    loss_identity = F.l1_loss(same_monet,monet_imgs)    #   mae
                    lambda_idenity = 0.2    #   artifact penalty

                    g_loss = (
                        g_adv + 
                        lambda_clip_style * g_clip_style + 
                        lambda_clip_content * g_clip_content +
                        lambda_idenity * loss_identity
                    )

                g_opt.zero_grad()
                # Scale + backpropagate
                scaler.scale(g_loss).backward()
                scaler.unscale_(g_opt)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)    #   gradient clipping
                scaler.step(g_opt)
                scaler.update()
                g_loss_val = g_loss.item()

            pbar.set_description(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss_val if isinstance(g_loss_val, str) else f'{g_loss_val:.4f}'}")

        g_scheduler.step()
        d_scheduler.step()
        current_lr = g_scheduler.get_last_lr()[0]
        print(f"End of epoch {epoch}. Current LR: {current_lr:.6f}")

        # Visualization and checkpointing
        if epoch % 5 == 0:
            sample_batch, _ = next(iter(real_val_loader))
            show_generated_samples(generator, sample_batch[:4], device)
        
        current_mifid = compute_mifid(generator, monet_val_loader, real_val_loader, device, batch_size)
        print(f"Epoch {epoch}/{num_epochs} MiFID: {current_mifid:.4f}")
        if current_mifid < best_mifid_so_far:
            best_mifid_so_far = current_mifid
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'best_mifid': best_mifid_so_far,
                'g_opt': g_opt.state_dict(),
                'd_opt': d_opt.state_dict(),
                'g_scheduler': g_scheduler.state_dict(),
                'd_scheduler': d_scheduler.state_dict()
            }, 'best_model.pth')
            print(f"Saved new best model with MiFID: {best_mifid_so_far:.4f}")
            
    return generator


if __name__ == "__main__":
        # GPU Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    scaler = GradScaler()

    #   Hyperparameters
    #   lambda_clip_style: monet style incentive
    #   lambda_clip_content: content preservation pentalty
    batch_size = 4
    num_epochs = 50
    initial_lr = 2e-4
    lambda_clip_style = 40.0
    lambda_clip_content = 5.0
    n_critics = 5

    monet_train_loader,monet_val_loader,real_train_loader,real_val_loader = get_loaders(batch_size)
    #   Pretrained "CLIP" weights
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_encoder = clip_model.float()
    clip_encoder.eval()

    #generator = CLIPEncoderDecoder(clip_encoder).to(device)
    generator = UNetGenerator(clip_encoder, latent_dim=100).to(device)
    discriminator = PatchDiscriminator().to(device)

    g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))


    g_scheduler = torch.optim.lr_scheduler.LinearLR(g_opt, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
    d_scheduler = torch.optim.lr_scheduler.LinearLR(d_opt, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
    start_epoch = 0
    best_mifid = float('inf')

    #   If resuming from checkpoint
    if os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth')
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        g_opt.load_state_dict(checkpoint['g_opt'])
        d_opt.load_state_dict(checkpoint['d_opt'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_mifid = checkpoint.get('best_mifid', float('inf'))

        if 'g_scheduler' in checkpoint and 'd_scheduler' in checkpoint:
            g_scheduler.load_state_dict(checkpoint['g_scheduler'])
            d_scheduler.load_state_dict(checkpoint['d_scheduler'])
            print("Resumed scheduler states.")

        print(f"Resumed training from epoch {start_epoch}, best MiFID: {best_mifid:.4f}")
        sample_batch, _ = next(iter(real_val_loader))
        show_generated_samples(generator, sample_batch[:4], device)

    trained_generator = train_gan(
        generator, discriminator, real_train_loader, monet_train_loader, 
        clip_model, device, 
        g_scheduler, d_scheduler, scaler,
        num_epochs=num_epochs, 
        g_opt=g_opt, d_opt=d_opt, start_epoch=start_epoch, 
        best_mifid=best_mifid, batch_size=batch_size, 
        monet_val_loader=monet_val_loader, real_val_loader=real_val_loader,
        lambda_clip_style=lambda_clip_style, 
        lambda_clip_content=lambda_clip_content, 
        n_critics=n_critics
    )
    display_img(monet_train_loader, real_train_loader)