import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.transforms import Resize, CenterCrop, Compose
from torchvision.io import read_video
import os
import random


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=in_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        print(f"ResBlock input shape: {x.shape}")
        identity = self.downsample(x)
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        result = self.silu(out + identity)
        print(f"ResBlock output shape: {result.shape}")
        return result


class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

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
            nn.Linear(256 * 30 * 30, 512),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 256 * 30 * 30)
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
        print(f"Encode input shape: {x.shape}")
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            print(f"Encoder layer {i} output shape: {x.shape}")
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        print(f"Encode output shapes - mu: {mu.shape}, logvar: {logvar.shape}")
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        print(f"Decode input shape: {z.shape}")
        h = self.decoder_input(z)
        h = h.view(-1, 256, 30, 30)
        print(f"Decoder input reshape: {h.shape}")
        for i, layer in enumerate(self.decoder):
            h = layer(h)
            print(f"Decoder layer {i} output shape: {h.shape}")
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def load_video_frame(video_dir, size=(480, 480)):
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    video_path = os.path.join(video_dir, random.choice(video_files))
    print(f"Selected video: {video_path}")

    video, _, _ = read_video(video_path, pts_unit="sec")
    print(f"Video shape: {video.shape}")
    frame_index = random.randint(0, len(video) - 1)
    frame = video[frame_index].float() / 255.0
    print(f"Selected frame index: {frame_index}, Frame shape: {frame.shape}")

    transform = Compose(
        [
            Resize(size),
            CenterCrop(size),
        ]
    )
    frame = frame.permute(2, 0, 1)
    frame = transform(frame)
    print(f"Frame shape after transform: {frame.shape}")

    for c in range(3):
        frame[c] = (frame[c] - frame[c].mean()) / frame[c].std()

    frame = frame.unsqueeze(0)
    print(f"Final frame shape: {frame.shape}")
    return frame


if __name__ == "__main__":
    print("Starting script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_channels = 3
    latent_dim = 128
    model = VAE(input_channels=input_channels, latent_dim=latent_dim)
    print("Model architecture:")
    print(model)

    print("Loading model weights...")
    model.load_state_dict(torch.load("sota_vae_mujoco.pth", map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded and set to eval mode")

    video_dir = "../../data/vae/Ant-v1/testing_val"
    print(f"Loading frame from: {video_dir}")
    frame = load_video_frame(video_dir).to(device)
    print(f"Frame loaded, shape: {frame.shape}, device: {frame.device}")

    print("Starting inference...")
    with torch.no_grad():
        try:
            print("Encoding frame...")
            mu, logvar = model.encode(frame)
            print("Frame encoded successfully")

            print("Reparameterizing...")
            latent = model.reparameterize(mu, logvar)
            print(f"Latent vector shape: {latent.shape}")

            print("Decoding latent vector...")
            decoded_frame = model.decode(latent)
            print("Frame decoded successfully")
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback

            print(traceback.format_exc())

    print("Saving images...")
    save_image(frame, "original_frame.png")
    save_image(decoded_frame, "decoded_frame.png")

    print("Generating new frame...")
    with torch.no_grad():
        try:
            sampled_latent = torch.randn(1, latent_dim).to(device)
            generated_frame = model.decode(sampled_latent)
            save_image(generated_frame, "generated_frame.png")
            print("New frame generated and saved")
        except Exception as e:
            print(f"Error during frame generation: {str(e)}")
            import traceback

            print(traceback.format_exc())

    print("Script completed")
