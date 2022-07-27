from torch.autograd import Variable
from scipy import stats
import torch
import pandas as pd
import numpy as np
from torchvision.utils import make_grid, save_image
import data_set as prs
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
np.random.seed(0)

class Visualizer():
    def __init__(self, model):
        self.model = model
            
    def get_dataset(self,key,labels,minus):
        dataset = list(labels[labels[key] == 1*minus]['image_id'])
        return prs.GetDataset("data_path/celeba/img_align_celeba/", dataset)
        
    def vis_latent(self):
        labels = pd.read_csv(r"list_attr.txt", delimiter=",")
        hair_color = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Bald", "Gray_Hair"]
        gender = ['Male','Female']
        
        hair_datasets = [self.get_dataset(key,labels,1) for key in hair_color]
        gender_datasets = [self.get_dataset('Male',labels,1),self.get_dataset('Male',labels,-1)]
        
        for dataset in hair_datasets:
            self._get_latent(dataset)
        plt.legend(hair_color)
        plt.title('Hair')
        
        plt.show()
        plt.savefig('hairs.png')
        plt.clf()
        
        for dataset in gender_datasets:
            self._get_latent(dataset)
        plt.legend(gender)
        plt.title('Gender')
        
        plt.show()
        plt.savefig('gender.png')
        
    
    def _get_latent(self, ds):
        ds_loader = DataLoader(dataset=ds, batch_size=64, shuffle=False)
        means = []
        vars = []
        for i, (x, y) in enumerate(ds_loader):
            z = self.model.encode(x.cuda())
            mean, logvar = z['cont']
            v1 = torch.mean(mean).to('cpu').detach().numpy()
            v2 = torch.mean(logvar).to('cpu').detach().numpy()
            means.append(v1)
            vars.append(v2)
        plt.scatter(means, vars)

