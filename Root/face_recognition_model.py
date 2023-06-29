import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import random
import numpy as np
import PIL

class FaceRecognitionModel(nn.Module):
    def __init__(self, weights_path):
        super(FaceRecognitionModel, self).__init__()
        self.weights_path = weights_path
        self.siamese_net = SiameseNetwork()

    def load_weights(self):
        self.siamese_net.load_state_dict(torch.load(self.weights_path))

    def get_face_embedding(self, face):
        transform = transforms.Compose([transforms.Resize((100, 100)),
                                        transforms.ToTensor()])
        face = transform(face).unsqueeze(0)
        embedding = self.siamese_net.forward_once(face)
        return embedding

    def match_face(self, face_embedding, database_path):
        folder_dataset = dset.ImageFolder(root=database_path)
        siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                                transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                            transforms.ToTensor()]),
                                                should_invert=False)
        dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)

        for i, (x0, _, _) in enumerate(dataloader):
            concatenated = torch.cat((x0, face_embedding), 0)
            output1, output2 = self.siamese_net(Variable(x0), Variable(face_embedding))
            euclidean_distance = F.pairwise_distance(output1, output2)
            
            if euclidean_distance > 0.27:
                return "Незнакомец"
            else:
                return "Есть совпадение"

class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2