import os
import torch
from PIL import Image
from VAE_model import ConvVAE
import matplotlib.pyplot as plt
from loss_func import loss_function
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.transforms.functional as F

#train function
def train(epoch, model, dataloader, device, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader),
                loss.item() / len(data)))
        # Generate and save images every 100 batches
        if batch_idx % 100 == 0:
            with torch.no_grad():
                sample = model.generate(num_samples=8, device=device)
                sample = sample.cpu()  # Move to CPU
                os.makedirs(f"images/epoch{epoch}", exist_ok=True)
                for i, image in enumerate(sample):
                    image = F.to_pil_image(image)  # Convert to PIL Image
                    image.save(f"images/epoch{epoch}/batch{batch_idx}_image{i}.png")
    avg_loss = train_loss / len(dataloader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(dataloader.dataset)))
    return avg_loss

def validate(epoch, model, dataloader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            val_loss += loss.item()

    avg_loss = val_loss / len(dataloader.dataset)
    print('====> Validation set loss: {:.4f}'.format(avg_loss))
    return avg_loss

def main():
    # train the VAE
    num_epoch=10
    lr=1e-5
    input_dim=256
    latent_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define the data loader
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((input_dim,input_dim)),
        transforms.ToTensor(),
    ])
    root="./chest_xray"
    train_dir, val_dir, test_dir = root+'/train', root+'/val', root+'./test'

    train_dataset,val_dataset,test_dataset = datasets.ImageFolder(root=train_dir, transform=transform),\
                                             datasets.ImageFolder(root=val_dir, transform=transform),\
                                             datasets.ImageFolder(root=test_dir, transform=transform)
    train_dataloader,val_dataloader,test_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True),\
                                                      DataLoader(val_dataset, batch_size=32, shuffle=False),\
                                                      DataLoader(test_dataset, batch_size=32, shuffle=False)

    # initialize the VAE
    model = ConvVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.00001)
    best_val_loss = float('inf')
    train_loss_lst = []
    for epoch in range(1, num_epoch+1):
        train_loss = train(epoch, model, train_dataloader, device, optimizer)
        val_loss = validate(epoch, model, val_dataloader, device)
        train_loss_lst.append(train_loss)
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            print("Saving model, Validation loss has decreased from: {:.4f} to: {:.4f}".format(best_val_loss, val_loss))
            best_val_loss = val_loss
            torch.save(model.state_dict(), './modelsbest_model.pt')    
    # After training, plot the train losses
    plt.figure(figsize=(15, 7))
    plt.plot(train_loss_lst)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('train_loss_plot.png')
    plt.show()
if __name__=="__main__":
    main()