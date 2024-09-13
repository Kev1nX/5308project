import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


train_transform = transforms.Compose([transforms.ToTensor()])  # Divide all numbers by 255 and normalize the data to [0, 1]
trainset = ImageFolder('../training_set', transform=train_transform)  # Load the training dataset

print('Number of images are', len(trainset))  # Print the total number of images in the training dataset

# Create a data loader for the training dataset
data_loader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Initialize variables to store the mean and standard deviation for each channel
mean = torch.zeros(3)
std = torch.zeros(3)

for X, _ in data_loader:
    for d in range(3):  # Iterate over the three RGB channels (0 for Red, 1 for Green, 2 for Blue)
        mean[d] += X[:, d, :, :].mean()  # Calculate the mean for the current channel
        std[d] += X[:, d, :, :].std()  # Calculate the standard deviation for the current channel

# Divide the accumulated mean and standard deviation values by the total number of images in the training dataset
mean.div_(len(trainset))
std.div_(len(trainset))

print(f'Mean: {list(mean.numpy())}')
print(f'Std: {list(std.numpy())}')