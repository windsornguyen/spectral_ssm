import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_video
import os
import numpy as np
from tqdm import tqdm
from ignite.metrics import SSIM

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=in_channels) # depth-wise separable convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)   # point-wise convolution
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()   # Swish activation function
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.silu(out + identity)


class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.softplus = nn.Softplus()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            ResBlock(32, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            ResBlock(64, 64),
            nn.Conv2d(64, 128, 4, 2, 1),
            ResBlock(128, 128),
            nn.Conv2d(128, 256, 4, 2, 1),
            ResBlock(256, 256),
            nn.Flatten(),
            nn.Linear(256 * 30 * 30, 512),  # Updated for 480x480 input
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(
            latent_dim, 256 * 30 * 30
        )  # Updated for 480x480 input
        self.decoder = nn.Sequential(
            ResBlock(256, 256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            ResBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            ResBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            ResBlock(32, 32),
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = self.softplus(logvar) # Use softplus to ensure positive std
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 30, 30)  # Updated for 480x480 input
        return self.decoder(h)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        x_recon = x_recon.view(b, t, c, h, w)
        return x_recon, mu, logvar


class VideoDataset(Dataset):
    def __init__(self, video_dirs, sequence_length, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform
        self.video_files = []
        for video_dir in video_dirs:
            self.video_files += [
                os.path.join(video_dir, f)
                for f in os.listdir(video_dir)
                if f.endswith(".mp4")
            ]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video, _, info = read_video(video_path, pts_unit="sec")

        if video.shape[0] < self.sequence_length:
            raise ValueError(
                f"Video {self.video_files[idx]} has fewer frames than required"
            )

        start_idx = np.random.randint(0, video.shape[0] - self.sequence_length + 1)
        sequence = video[start_idx : start_idx + self.sequence_length]
        sequence = sequence.permute(0, 3, 1, 2).float() / 255.0

        if self.transform:
            sequence = torch.stack([self.transform(frame) for frame in sequence])

        return sequence


def validate_dataset(dataset):
    print(f"Total number of videos: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample min value: {sample.min()}")
    print(f"Sample max value: {sample.max()}")

    # Validate first 5 videos
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        assert sample.shape == (
            30,
            3,
            480,
            480,
        ), f"Unexpected sample shape: {sample.shape}"
        assert (
            sample.min() >= -1 and sample.max() <= 1
        ), f"Sample values out of expected range: [{sample.min()}, {sample.max()}]"

    print("Dataset validation passed!")


def train_vae(model, train_loader, val_loader, num_epochs, learning_rate, beta, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def loss_function(recon_x, x, mu, logvar):
        ssim = SSIM(data_range=1.0)
        ssim_values = []
        for i in range(recon_x.size(1)):
            ssim.update((recon_x[:, i], x[:, i]))
            ssim_value = ssim.compute()
            ssim_values.append(torch.tensor(ssim_value))  # Convert to tensor
            ssim.reset()  # Reset for the next computation
        ssim_per_frame = torch.stack(ssim_values)
        recon_loss = 1 - ssim_per_frame.mean()
        # recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return recon_loss + beta * kld_loss

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                val_loss += loss_function(recon_batch, data, mu, logvar).item()

        avg_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    return model


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")

    # Hyperparameters
    latent_dim = 128
    num_epochs = 3
    batch_size = 1
    learning_rate = 1e-4
    beta = 4.0  # Beta-VAE hyperparameter
    sequence_length = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((480, 480)),  # Updated to match input size
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Create datasets
    train_dataset = VideoDataset(
        video_dirs=["../../data/vae/Ant-v1/testing", "../../data/vae/HalfCheetah-v1/testing", "../../data/vae/Walker2D-v1/testing"],
        sequence_length=sequence_length,
        transform=transform,
    )
    val_dataset = VideoDataset(
        video_dirs=["../../data/vae/Ant-v1/testing_val", "../../data/vae/HalfCheetah-v1/testing_val", "../../data/vae/Walker2D-v1/testing_val"],
        sequence_length=sequence_length,
        transform=transform,
    )

    # Validate datasets
    print("Validating training dataset:")
    validate_dataset(train_dataset)
    print("\nValidating validation dataset:")
    validate_dataset(val_dataset)

    # Create DataLoaders
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
    )

    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4
    )

    # Initialize and train the model
    model = VAE(input_channels=3, latent_dim=latent_dim)
    model.to(device)
    model = DistributedDataParallel(model)

    trained_model = train_vae(
        model, train_loader, val_loader, num_epochs, learning_rate, beta, device
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), "sota_vae_mujoco.pth")

    print("Training completed and model saved.")

    dist.destroy_process_group()
