import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from torch.utils.data import Dataset
from itertools import cycle

from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torch.nn.utils import spectral_norm

from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler

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

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residuals=9, features=64):
        super().__init__()

        #   Initial Conv. Block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, kernel_size=7),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True)
        ]

        #   Downsampling
        curr_dim = features
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim*2),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2

        #   Residual block (between down/upsample)
        for _ in range(n_residuals):
            model += [ResidualBlock(curr_dim)]

        #   Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(curr_dim//2),
                nn.ReLU(inplace=True)
            ]
            curr_dim //= 2
        
        #   Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_dim, out_channels, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        #   Spectral_norm on first 2 layers
        layers = [
            spectral_norm(nn.Conv2d(in_channels, features[0], 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        layers += [
            spectral_norm(nn.Conv2d(features[0], features[1], 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_feat = features[1]
        for feat in features[2:]:
            stride = 1 if feat == features[-1] else 2
            layers += [
                nn.Conv2d(in_feat, feat, 4, stride, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_feat = feat

        #   Final layer (no spectral norm)
        layers += [
            nn.Conv2d(in_feat, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

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

def save_ckpt(path, epoch, G_AB, G_BA, D_A, D_B,
              g_opt, d_a_opt, d_b_opt, best_mifid):
    torch.save({
        'epoch': epoch,
        'best_mifid': best_mifid,
        'G_AB': G_AB.state_dict(),
        'G_BA': G_BA.state_dict(),
        'D_A':  D_A.state_dict(),
        'D_B':  D_B.state_dict(),
        'g_opt': g_opt.state_dict(),
        'd_a_opt': d_a_opt.state_dict(),
        'd_b_opt': d_b_opt.state_dict()
    }, path)


def train(monet_train_loader,monet_val_loader,real_train_loader,real_val_loader,device,batch_size,resume_path = None):
    #   Initialize losses to save for visualizing
    loss_path = "cycleGAN_losses7.npz"
    if os.path.isfile(loss_path):
        buf = np.load(loss_path)
        g_losses         = buf["g_losses"].tolist()
        d_a_losses       = buf["d_a_losses"].tolist()
        d_b_losses       = buf["d_b_losses"].tolist()
        cycle_losses     = buf["cycle_losses"].tolist()
        identity_losses  = buf["identity_losses"].tolist()
    else:
        g_losses, d_a_losses, d_b_losses, cycle_losses, identity_losses = ([] for _ in range(5))


    best_mifid = 1000
    start_epoch = 0

    # ===== 5. Losses and Optimizers =====
    adv_criterion = nn.MSELoss()
    cycle_criterion = nn.L1Loss()
    identity_criterion = nn.L1Loss()

    G_AB = Generator().to(device)  # Photo → Monet
    G_BA = Generator().to(device)  # Monet → Photo
    D_A = Discriminator().to(device)  # Photo disc
    D_B = Discriminator().to(device)  # Monet disc

    lr = 2e-4
    beta1 = 0.5
    G_optimizer = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(beta1, 0.999))
    D_A_optimizer = optim.Adam(D_A.parameters(), lr=5e-5, betas=(beta1, 0.999))
    D_B_optimizer = optim.Adam(D_B.parameters(), lr=5e-5, betas=(beta1, 0.999))

    #   If resuming from checkpoint
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)

        #  -- restore weights --
        G_AB.load_state_dict(ckpt['G_AB'])
        G_BA.load_state_dict(ckpt['G_BA'])
        D_A.load_state_dict(ckpt['D_A'])
        D_B.load_state_dict(ckpt['D_B'])

        #  -- restore optimisers --
        G_optimizer.load_state_dict(ckpt['g_opt'])
        for p in G_optimizer.param_groups:  #   Half the resuming G lr
            p['lr'] *= 0.5
        D_A_optimizer.load_state_dict(ckpt['d_a_opt'])
        D_B_optimizer.load_state_dict(ckpt['d_b_opt'])
        #   Half discriminator lr
        for opt in (D_A_optimizer, D_B_optimizer):
            for g in opt.param_groups:
                g['lr'] *= 0.5


        #  -- bookkeeping --
        start_epoch = ckpt['epoch'] + 1
        best_mifid  = ckpt['best_mifid']
        print(f"Resumed from {resume_path}  (epoch {start_epoch}, best MiFID {best_mifid:.2f})")

    #   mixed precision
    scaler = GradScaler()

    # ===== 6. Training Loop =====
    lambda_cycle = 10
    lambda_identity = 0.0
    epochs = 100
    '''   
    g_losses = []
    d_a_losses = []
    d_b_losses = []
    cycle_losses = []
    identity_losses = []
    '''
    print("Beginning Training...")
    step = 0
    monet_iter = cycle_loader(monet_train_loader)

    for epoch in range(start_epoch,epochs):
        epoch_g_loss = 0
        epoch_d_a_loss = 0
        epoch_d_b_loss = 0
        epoch_cycle_loss = 0
        epoch_identity_loss = 0

        loop = tqdm(real_train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                unit="batch", ncols=100)
        
        #   Iterate through loop for progress bar, or photo_iter
        for real_batch,_ in loop:
            monet_batch, _ = next(monet_iter)

            real_A = real_batch.to(device)
            real_B = monet_batch.to(device)

            # ────────── GENERATOR STEP (AMP) ──────────
            G_optimizer.zero_grad()
            with autocast():
                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)
                recov_A = G_BA(fake_B)
                recov_B = G_AB(fake_A)
                idt_A   = G_BA(real_A)
                idt_B   = G_AB(real_B)

                D_B_fake = D_B(fake_B)
                D_A_fake = D_A(fake_A)

                loss_GAN_AB = adv_criterion(D_B_fake, torch.ones_like(D_B_fake, device=device))
                loss_GAN_BA = adv_criterion(D_A_fake, torch.ones_like(D_A_fake,  device=device))

                loss_cycle_A = cycle_criterion(recov_A, real_A)
                loss_cycle_B = cycle_criterion(recov_B, real_B)

                loss_idt_A = identity_criterion(idt_A, real_A) * lambda_identity
                loss_idt_B = identity_criterion(idt_B, real_B) * lambda_identity

                loss_G = (
                    loss_GAN_AB + loss_GAN_BA +
                    lambda_cycle * (loss_cycle_A + loss_cycle_B) +
                    loss_idt_A + loss_idt_B
                )

            scaler.scale(loss_G).backward()
            scaler.step(G_optimizer)
            scaler.update()

            #   Discriminators
            if step % 2 == 0:
                # ---- D_A ----
                D_A_optimizer.zero_grad()
                with autocast():
                    D_A_real = D_A(add_noise(real_A, 0.05))     
                    D_A_fake = D_A(add_noise(fake_A.detach(), 0.05))
                    loss_D_A_real = adv_criterion(D_A_real, smooth_positive_labels(D_A_real))
                    loss_D_A_fake = adv_criterion(D_A_fake, smooth_negative_labels(D_A_fake))
                    loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
                scaler.scale(loss_D_A).backward()
                scaler.step(D_A_optimizer)

                # ---- D_B ----
                D_B_optimizer.zero_grad()
                with autocast():
                    D_B_real = D_B(add_noise(real_B, 0.05))
                    D_B_fake = D_B(add_noise(fake_B.detach(), 0.05))
                    loss_D_B_real = adv_criterion(D_B_real, smooth_positive_labels(D_B_real))
                    loss_D_B_fake = adv_criterion(D_B_fake, smooth_negative_labels(D_B_fake))
                    loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
                scaler.scale(loss_D_B).backward()
                scaler.step(D_B_optimizer)

                scaler.update()
           
            else:
                # keep last loss values for logging when D is skipped
                loss_D_A = loss_D_A.detach()
                loss_D_B = loss_D_B.detach()

            step += 1

            loop.set_postfix(G=f"{loss_G.item():.3f}",
                        D_A=f"{loss_D_A.item():.3f}",
                        D_B=f"{loss_D_B.item():.3f}")

            # Track losses
            epoch_g_loss += loss_G.item()
            epoch_d_a_loss += loss_D_A.item()
            epoch_d_b_loss += loss_D_B.item()
            epoch_cycle_loss += (loss_cycle_A + loss_cycle_B).item()
            epoch_identity_loss += (loss_idt_A + loss_idt_B).item()

        # Save avg per epoch
        g_losses.append(epoch_g_loss / len(real_train_loader))
        d_a_losses.append(epoch_d_a_loss / len(real_train_loader))
        d_b_losses.append(epoch_d_b_loss / len(real_train_loader))
        cycle_losses.append(epoch_cycle_loss / len(real_train_loader))
        identity_losses.append(epoch_identity_loss / len(real_train_loader))
        mifid_score = compute_mifid(G_AB, monet_val_loader, real_val_loader, device,batch_size)

        print(f"Epoch [{epoch+1}/{epochs}] | G Loss: {g_losses[-1]:.4f} | D_A Loss: {d_a_losses[-1]:.4f} | D_B Loss: {d_b_losses[-1]:.4f} | MiFID: {mifid_score}")
        if mifid_score < best_mifid:
            best_mifid = mifid_score
            save_ckpt('ckpt_best.pt', epoch,G_AB,G_BA,D_A,D_B,G_optimizer,D_A_optimizer,D_B_optimizer,best_mifid)
        save_ckpt('ckpt_latest.pt', epoch,G_AB,G_BA,D_A,D_B,G_optimizer,D_A_optimizer,D_B_optimizer,best_mifid)


        #   Save losses to plot later
        np.savez(loss_path,
                g_losses=np.array(g_losses),
                d_a_losses=np.array(d_a_losses),
                d_b_losses=np.array(d_b_losses),
                cycle_losses=np.array(cycle_losses),
                identity_losses=np.array(identity_losses))

        print("Losses saved.")

    return G_AB

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



if __name__ == "__main__":
        # GPU Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    

    batch_size = 4
    
    monet_train_loader,monet_val_loader,real_train_loader,real_val_loader = get_loaders(batch_size)
    print("len(real_train_loader) =", len(real_train_loader))


    display_img(monet_train_loader, real_train_loader)

    #   Train
    G_AB = train(monet_train_loader,monet_val_loader,real_train_loader,real_val_loader,device,batch_size,resume_path="ckpt_best.pt")

    # Run eval
    mifid_score = compute_mifid(G_AB, monet_val_loader, real_val_loader, device,batch_size)
    print(f"Validation MiFID Score: {mifid_score:.2f}")

    real_batch, _ = next(iter(real_val_loader))
    show_generated_samples(G_AB, real_batch, device, num_samples=4)