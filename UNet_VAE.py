import torch
from torch.nn import functional as F
from torch import nn
class UNetVAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128):
        super(UNetVAE, self).__init__()
        self.latent_dim = latent_dim

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        def conv_block_dec(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc_conv1 = conv_block(input_channels, 64)
        self.enc_conv2 = conv_block(64, 128)
        self.enc_conv3 = conv_block(128, 256)
        self.enc_conv4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc_mu = nn.Linear(512 * 16 * 16, self.latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 16, self.latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(self.latent_dim, 512 * 16 * 16)
        self.dec_upconv4 = upconv_block(1024, 256)
        self.dec_conv4 = conv_block(256, 256)
        self.dec_upconv3 = upconv_block(512, 128)
        self.dec_conv3 = conv_block(128, 128)
        self.dec_upconv2 = upconv_block(256, 64)
        self.dec_conv2 = conv_block(64, 64)
        self.dec_upconv1 = upconv_block(128, 64)
        self.dec_conv1 = conv_block(64, 32)
        self.final_conv = nn.Conv2d(32, input_channels, kernel_size=1)

    def encode(self, x):
        x1 = self.enc_conv1(x)
        x2 = self.pool(x1)
        x2 = self.enc_conv2(x2)
        x3 = self.pool(x2)
        x3 = self.enc_conv3(x3)
        x4 = self.pool(x3)
        x4 = self.enc_conv4(x4)
        x4 = self.dropout(x4)
        x5 = self.pool(x4)
        x5 = x5.view(x5.size(0), -1)
        print(x1.shape,x2.shape,x3.shape,x4.shape)
        return self.fc_mu(x5), self.fc_logvar(x5), x1, x2, x3, x4

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
   
    def decode(self, z, x1, x2, x3, x4):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 512, 16, 16)

        # start decoder process (with Upsampling and concatenation)
        z = nn.functional.interpolate(z, size=(x4.size(2), x4.size(3)))
        z = torch.cat([x4, z], dim=1)
        z = self.dec_upconv4(z)
        z = self.dec_conv4(z)
        z = nn.functional.interpolate(z, size=(x3.size(2), x3.size(3)))
        z = torch.cat([x3, z], dim=1)
        z = self.dec_upconv3(z)
        z = self.dec_conv3(z)

        z = nn.functional.interpolate(z, size=(x2.size(2), x2.size(3)))
        z = torch.cat([x2, z], dim=1)
        z = self.dec_upconv2(z)
        z = self.dec_conv2(z)
        z = nn.functional.interpolate(z, size=(x1.size(2), x1.size(3)))
        z = torch.cat([x1, z], dim=1)
        z = self.dec_upconv1(z)
        z = self.dec_conv1(z)
        # final layer uses sigmoid activation function like the UNet model
        out = torch.sigmoid(self.final_conv(z))
        return out

    def forward(self, x):
        mu, logvar, x1, x2, x3, x4 = self.encode(x)
        # print(f'Shape of mu: {mu.shape}')
        # print(f'Shape of logvar: {logvar.shape}')
        z = self.reparameterize(mu, logvar)
        # print(f'Shape of z: {z.shape}')
        return self.decode(z, x1, x2, x3, x4), mu, logvar



