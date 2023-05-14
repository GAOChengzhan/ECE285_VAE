import torch
import os
from tqdm import tqdm
from UNet_VAE import UNetVAE
from VAE_model import ConvVAE
from scipy.stats import entropy
import torch.nn.functional as F
import torchvision.transforms.functional as tF
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from utils import RescaleToUnitRange,generate_from_data,unnormalize



n_sample=100
input_dim=256
latent_dim=128
batch_size=24
root="./chest_xray"
train_dir, val_dir, test_dir = root+'/train', root+'/val', root+'/test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: "+str(device))

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((input_dim,input_dim)),
    transforms.ToTensor(),
    transforms.Normalize(0.4823,0.2363),
    RescaleToUnitRange(),
])
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# Load your trained VAE model
# vae_model = UNetVAE(1, latent_dim=256).to(device)
# vae_model.load_state_dict(torch.load('./modelsbest_model.pt'))

vae_model = ConvVAE(input_dim=256, latent_dim=128).to(device)
vae_model.load_state_dict(torch.load('./res/1/modelsbest_model.pt'))
x_dir='./x_mean/'
x1_mean,x2_mean, x3_mean,x4_mean= torch.load(x_dir+'x1_mean.pt'),\
                                  torch.load(x_dir+'x2_mean.pt'),\
                                  torch.load(x_dir+'x3_mean.pt'),\
                                  torch.load(x_dir+'x4_mean.pt')

for batch_idx, (data, _) in tqdm(enumerate(test_dataloader)):
    data = data.to(device)
    with torch.no_grad():
        # noise = torch.randn(batch_size, latent_dim, device=device)
        vae_model.eval()
        # _, _, x1, x2, x3, x4 = vae_model.encode(data)
        # images = vae_model.decode(noise, x1_mean, x2_mean, x3_mean, x4_mean).cpu()
        images, _, _ = vae_model(data)
        os.makedirs(f"images/ConvVAE/batch{batch_idx}", exist_ok=True)
        for i, image in enumerate(images):
            image = tF.to_pil_image(image)  # Convert to PIL Image
            image.save(f"images/ConvVAE/batch{batch_idx}/image{i}.png")

