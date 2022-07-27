import torch
import torchvision.datasets as dsets
import torchvision
from models import VAE
from training import Trainer
from torch.optim import Adam
from visualizations import Visualizer as Viz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import os

torch.manual_seed(3407)

class GetDataset(Dataset):
    def __init__(self, directory, images):
        self.directory = directory
        self.images = images
        self.entries = self._create_entries()


    def _create_entries(self):
        entries = []
        transform = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            #torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((64, 64))
        ])
        names = []
        for filename in self.images:
            f = os.path.join(self.directory, filename)
            image = Image.open(f)
            x = transform(image).to(torch.float)
            # y = int(filename[filename.index('_') + 1])
            y = 1
            entries.append({'x': x, 'y': y})
            names.append(filename)
        return entries

    def __getitem__(self, index):
        entry = self.entries[index]
        return entry['x'], entry['y']

    def __len__(self):
        return len(self.entries)





def plot_kld(kl):
    plt.plot(kl)
    plt.title('KL Divergence Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title(f'KLD Loss Per Epoch')
    plt.legend(['kl'])
    plt.show()
    plt.savefig('kld_loss.jpeg')
    


def plot_rec(rec):
    plt.plot(rec)
    plt.title('Reconstruction Loss Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['loss'])
    plt.savefig('rec_loss.jpeg')
    plt.show()




def reproduce_hw3():
    latent_space = {'cont': 100, 'disc': [50]}

    model = VAE(img_size=(3, 64, 64), latent_space=latent_space)
    model.load_state_dict(torch.load('model.pkl'))
    model = model.cuda()
    viz = Viz(model)
    model.samples()
    plt.clf()
    viz.vis_latent()


def main():
  #transform = torchvision.transforms.Compose([
  #    torchvision.transforms.ToTensor(),
  #    torchvision.transforms.Resize((64, 64)),
  #    torchvision.transforms.Normalize((0.51, 0.42, 0.38), (0.30, 0.28, 0.282))
  #])
  #train_dataset = dsets.CelebA(root="data_path",
  #                             split='train',
  #                             target_type='attr',
  #                             transform=transform,
  #                             download=False)
  #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
  #
  #latent_space = {'cont': 100, 'disc': [50]}
  #model = VAE(img_size=(3, 64, 64), latent_space=latent_space)
  #
  #adam = Adam(model.parameters())
  #trainer = Trainer(model, adam, cont_capacity=[0., 5., 25000, 40.],
  #                  disc_capacity=[0., 5., 25000, 40.])
  #kl, recon = trainer.train(train_loader, epochs_number=15)
  #torch.save(model.state_dict(), 'model.pkl')
  #plt.clf()
  #plot_kld(kl)
  #plt.clf()
  #plot_rec(recon)
  reproduce_hw3()

if __name__ == '__main__':
    main()