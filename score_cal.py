import torch
from tqdm import tqdm
import PIL.Image as Image
from VAE_model import ConvVAE
from UNet_VAE import UNetVAE
from scipy.stats import entropy
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import FID, InceptionScore
from torchvision import transforms, datasets
from utils import RescaleToUnitRange,generate_from_data,unnormalize

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
train_dataset,val_dataset,test_dataset = datasets.ImageFolder(root=train_dir, transform=transform),\
                                         datasets.ImageFolder(root=val_dir, transform=transform),\
                                         datasets.ImageFolder(root=test_dir, transform=transform)
train_dataloader,val_dataloader,test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True),\
                                                  DataLoader(val_dataset, batch_size=batch_size, shuffle=False),\
                                                  DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Load your trained VAE model
# vae_model = UNetVAE(1, latent_dim=256).to(device)
# vae_model.load_state_dict(torch.load('./modelsbest_model.pt'))

vae_model = ConvVAE(input_dim=256, latent_dim=128).to(device)
vae_model.load_state_dict(torch.load('./res/1/modelsbest_model.pt'))
vae_model.eval()
x_dir='./x_mean/'
x1_mean,x2_mean, x3_mean,x4_mean= torch.load(x_dir+'x1_mean.pt'),\
                                  torch.load(x_dir+'x2_mean.pt'),\
                                  torch.load(x_dir+'x3_mean.pt'),\
                                  torch.load(x_dir+'x4_mean.pt')

fid_metric = FID(device=device)
is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])


def interpolate(batch,input_dim=input_dim):
    arr = []
    for img in batch:
        img = img.repeat((3,1,1))  # Repeat the single channel 3 times
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((input_dim,input_dim), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)


def evaluation_step(engine, batch,batch_size=batch_size, latent_dim=latent_dim):
    with torch.no_grad():
        vae_model.eval()
        fake_batch, _, _ = vae_model(batch[0].to(device))  # Generate images
        # noise = torch.randn(batch_size, latent_dim, device=device)
        
        # sample_inputs = torch.randn(batch_size, 1, input_dim, input_dim).to(device)
        # _, _, x1, x2, x3, x4 = vae_model.encode(sample_inputs)
        # Now you can use these in the decode method
        # fake_batch = vae_model.decode(noise, x1_mean, x2_mean, x3_mean, x4_mean)
        # fake_batch = vae_model.decode(noise)
        fake = interpolate(unnormalize(fake_batch.cpu()))  # Interpolate generated images
        real = interpolate(batch[0])  # Interpolate real images
        return fake, real

evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")

# Run the evaluator
evaluator.run(test_dataloader)

metrics = evaluator.state.metrics
fid_score = metrics['fid']
is_score = metrics['is']

print(f"FID score: {fid_score}")
print(f"Inception score: {is_score}")









# # Load pre-trained Inception-v3 model
# inception_model = InceptionV3().to(device)


# # Create a dataloader for real images
# real_dataloader = DataLoader(test_dataset, batch_size=24, shuffle=False)

# # Generate a list of real images
# real_imgs = []
# generated_imgs = []
# for batch_idx, (data, _) in tqdm(enumerate(real_dataloader)):
#     real_imgs.extend(data)
#     data = data.to(device)
# # Generate a list of generated images using VAE
#     with torch.no_grad():
#         generated_img = generate_from_data(vae_model, data,device)
#         generated_img = unnormalize(generated_img.cpu())
#         generated_imgs.append(generated_img)

# # Calculate inception score
# mean, std = inception_score(generated_imgs, inception_model)
# print(f"Inception score: {mean} +/- {std}")

# # # Calculate FID score
# # fid = compute_fid_score(real_imgs, generated_imgs, inception_model
