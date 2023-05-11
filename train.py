import torch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the VAE
model = VAE(input_dim=64*64, hidden_dim=256, latent_dim=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train the VAE
for epoch in range(10):
    for i, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(dataloader.dataset),
                100. * i / len
