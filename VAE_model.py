import torch
from torch import nn
from torch.nn import functional as F

class ConvVAE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=128):
        super(ConvVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(256 * 32 * 32, self.latent_dim)
        self.fc_logvar = nn.Linear(256 * 32 * 32, self.latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(self.latent_dim, 256 * 32 * 32)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.shape[0], 256, 32, 32)  # Un-flatten the tensor
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        return torch.sigmoid(self.conv_out(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # generate images
    def generate(self, num_samples=1, device='cpu'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
