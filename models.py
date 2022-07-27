import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
EPS = 1e-12


class VAE(nn.Module):
    def __init__(self, img_size, latent_space, temperature=.6, cuda=True):
    
        super(VAE, self).__init__()
        self.use_cuda = cuda
        self.img_size = img_size
        self.is_continuous = 'cont' in latent_space
        self.is_discrete = 'disc' in latent_space
        self.latent_space = latent_space
        self.num_pixels = img_size[1] * img_size[2]
        self.temperature = temperature
        self.hidden_dim = 256  
        self.reshape = (64, 4, 4)  
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_space['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_space['disc']])
            self.num_disc_latents = len(self.latent_space['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        encoderingLayers = [
            nn.Conv2d(self.img_size[0], 32, (4, 4), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, (4, 4), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.LeakyReLU()
        ]

        self.encoder_conv = nn.Sequential(*encoderingLayers)

        self.to_hidden = nn.Sequential(
            nn.Linear(64 * 4 * 4, self.hidden_dim),
            nn.LeakyReLU()
        )

        self.fc_mean = nn.Linear(self.hidden_dim, self.latent_cont_dim)
        self.fc_log_var = nn.Linear(self.hidden_dim, self.latent_cont_dim)

        alphas = []
        for disc_dim in self.latent_space['disc']:
            alphas.append(nn.Linear(self.hidden_dim, disc_dim))
        self.alphas = nn.ModuleList(alphas)

        self.from_hidden = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 64 * 4 * 4),
            nn.LeakyReLU()
        )

        decoder_layers = [
            nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, self.img_size[0], (4, 4), stride=2, padding=1),
            nn.Sigmoid()
        ]

        self.decoder_uncov = nn.Sequential(*decoder_layers)

    def encode(self, x):
        batch_size = x.size()[0]

        features = self.encoder_conv(x)
        hidden = self.to_hidden(features.view(batch_size, -1))

        latent_dist = {}

        if self.is_continuous:
            latent_dist['cont'] = [self.fc_mean(hidden), self.fc_log_var(hidden)]

        if self.is_discrete:
            latent_dist['disc'] = []
            for fc_alpha in self.alphas:
                latent_dist['disc'].append(F.softmax(fc_alpha(hidden), dim=1))

        return latent_dist

    def reparameterize(self, latent_dist):
        latent_sample = []
        if self.is_continuous:
            mean, logvar = latent_dist['cont']
            cont_sample = self.sample_normal(mean, logvar)
            latent_sample.append(cont_sample)

        if self.is_discrete:
            for alpha in latent_dist['disc']:
                disc_sample = self.sample_gumbel_softmax(alpha)
                latent_sample.append(disc_sample)

        return torch.cat(latent_sample, dim=1)
    

    def traverse_discrete_grid(self, dim, axis, traverse, size):
        num_samples = size[0] * size[1]
        samples = np.zeros((num_samples, dim))

        if traverse:
            disc_traversal = [i % dim for i in range(size[axis])]
            for i in range(size[0]):
                for j in range(size[1]):
                    if axis == 0:
                        samples[i * size[1] + j, disc_traversal[i]] = 1.
                    else:
                        samples[i * size[1] + j, disc_traversal[j]] = 1.
        else:
            
            samples[np.arange(num_samples), np.random.randint(0, dim, num_samples)] = 1.
            

        return torch.Tensor(samples)
        
    def samples(self, size=(3, 3), filename='sampling',epoch = None):
        samples = []
        if self.is_continuous:
            samples.append(torch.Tensor(np.random.normal(size=(size[0]*size[1],self.latent_cont_dim))))
          
        if self.is_discrete:
            for i, disc_dim in enumerate(self.latent_space['disc']):
                    samples.append(self.traverse_discrete_grid(dim=disc_dim,
                                                                axis=0,
                                                                traverse=False,
                                                                size=size))
        prior_samples = torch.cat(samples, dim=1)
        latent_samples = Variable(prior_samples)
        if self.use_cuda:
            latent_samples = prior_samples.cuda()
        generated = self.decode(latent_samples).cpu()
        
        fig = plt.gcf()
        plt.imshow(make_grid(generated.data, nrow=size[1]).permute(1,2,0))
        plt.axis('off')
            
        
        if epoch is not None:
          plt.title('epoch ' + str(epoch))
          plt.savefig(filename+'_'+str(epoch)+'.png')
        else:
          plt.savefig(filename+'.png')
          

    def sample_normal(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_()
            if self.use_cuda:
                eps = eps.cuda()
                std = std.cuda()
                mean = mean.cuda()
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def sample_gumbel_softmax(self, alpha):
        
        if self.training:
            unif = torch.rand(alpha.size())
            if self.use_cuda:
                unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            if self.use_cuda:
                one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples

    def decode(self, latent_sample):
        
        features = self.from_hidden(latent_sample)
        return self.decoder_uncov(features.view(-1, *self.reshape))

    def forward(self, x):
        
        latent_dist = self.encode(x)
        latent_sample = self.reparameterize(latent_dist)
        return self.decode(latent_sample), latent_dist
