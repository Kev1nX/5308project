import argparse
import torch
from network import GoogLeNet # the network you used
from network import ResNet18
from torchvision.datasets import ImageFolder
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

parser = argparse.ArgumentParser(description= \
                                     'scipt for classification part of project 2')
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
    transforms.Resize(500),
    transforms.CenterCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.564108, 0.50346, 0.427237), (0.20597, 0.206595, 0.21542))
])
# change the folder path to the parent folder of the folder storing the images
# the folder must look like :
#                            parent_folder_name
#                             -- image_folder_name
#                               -- image_1
#                               -- image_2 ...
# the parent folder should also not have any other folders besides the image folder.


def eval_net(net, loader):
    output = []
    try:
        for data in loader:
            images, labels = data
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            if args.cuda:
                outputs = outputs.cuda()
                labels = labels.cuda()
            _, predicted = torch.max(outputs.data, 1)
            out = predicted.tolist()
            for i in out:
                output.append(i)
                
            break
    except FileNotFoundError:
        print("File moved, retrying")
            
    return output

if __name__ == '__main__':     # this is used for running in Windows
    # make sure to change this to use the correct network for the .pth file
    network = ResNet18()
    if args.cuda:
        network = network.cuda()

    net = network.eval()
    if args.cuda:
        net = net.cuda()

    path = "test"
        
    if args.cuda:
        net.load_state_dict(torch.load(path + '.pth', map_location='cuda'))
    else:
        net.load_state_dict(torch.load(path + '.pth', map_location='cpu'))
    
    while True:
        ValSet = ImageFolder('classification_test', transform=train_transform)
        valloader = torch.utils.data.DataLoader(ValSet, batch_size=8,
                                         shuffle=False, num_workers=2)
        
        results = eval_net(net,valloader)
        print(results)
        del ValSet
        del valloader

# after installing all dependencies
# run the script by moving the terminal to this scipt's directory and running : python .\classify.py or python .\classify.py --cuda
# the script will print out in a list, the image classifications
# the results list object will contain all of the classificaitons
