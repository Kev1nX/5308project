import argparse
import torch
from network import GoogLeNet # the network you used
from network import ResNet18
from torchvision.datasets import ImageFolder
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
args = parser.parse_args()

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

train_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize((0.4323223, 0.42033324, 0.4274624), (0.19253562, 0.18891077, 0.19183522))
])

ValSet = ImageFolder('../classification_test', transform=train_transform)
valloader = torch.utils.data.DataLoader(ValSet, batch_size=8,
                                         shuffle=False, num_workers=2)


def eval_net(net, loader, path):
    net = net.eval()
    if args.cuda:
        net = net.cuda()
        
    if args.cuda:
        net.load_state_dict(torch.load(path + '.pth', map_location='cuda'))
    else:
        net.load_state_dict(torch.load(path + '.pth', map_location='cpu'))
        
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if args.cuda:
            outputs = outputs.cuda()
            labels = labels.cuda()
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)

if __name__ == '__main__':     # this is used for running in Windows
    network = GoogLeNet()
    if args.cuda:
        network = network.cuda()

    eval_net(network,valloader,"test")
