import torch
import torch
from torchvision import datasets, transforms
import torchvision

dataset = datasets.ImageFolder(r'../../../../ssd/hypersim_sample/rgb/hypersim_sample', transform=transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(256),
                             transforms.ToTensor()]))

loader = torch.utils.data.DataLoader(dataset,
                         batch_size=10,
                         num_workers=0,
                         shuffle=False)

mean = 0.0
print("1")
for images, _ in loader:
    batch_samples = images.size(0) 
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean = mean / len(loader.dataset)
print("mean")
print(mean)

print("2")
var = 0.0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])
std = torch.sqrt(var / (len(loader.dataset)*224*224))
print("std")
print(std)