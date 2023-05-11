from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# define a transform to normalize the data
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

# download and load the data
dataset = datasets.ImageFolder(root='./chest_xray', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
