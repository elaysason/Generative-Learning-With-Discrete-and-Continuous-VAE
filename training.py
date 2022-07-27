import imageio
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
from visualizations import Visualizer as Viz
import tqdm
torch.manual_seed(3407)
EPS = 1e-12


class Trainer():
    def __init__(self, model, optimizer, cont_capacity=None,
                 disc_capacity=None, print_loss_every=50, record_loss_every=5,
                 use_cuda=True):
        self.model = model
        self.viz = Viz(self.model)
        self.optimizer = optimizer
        self.cont_capacity = cont_capacity
        self.disc_capacity = disc_capacity
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.model.cuda()

        # Initialize attributes
        self.num_steps = 0
        self.batch_size = None
        self.losses = {'loss': [],
                       'recon_loss': [],
                       'kl_loss': []}

        # Keep track of divergence values for each latent variable
        if self.model.is_continuous:
            self.losses['kl_loss_cont'] = []
            # For every dimension of continuous latent variables
            for i in range(self.model.latent_space['cont']):
                self.losses['kl_loss_cont_' + str(i)] = []

        if self.model.is_discrete:
            self.losses['kl_loss_disc'] = []
            # For every discrete latent variable
            for i in range(len(self.model.latent_space['disc'])):
                self.losses['kl_loss_disc_' + str(i)] = []

    
    
              
    def train(self, data_loader, epochs_number=10):

        self.batch_size = data_loader.batch_size
        self.model.train()
        kl = []
        recon = []
        epochs = [0,4,14,29]
        for epoch in range(epochs_number):
            mean_loss,mean_kl, mean_recon = self._train_epoch(data_loader)
            kl.append(mean_kl)
            recon.append(mean_recon)
            print('Epoch: {} Average loss: {:.2f}'.format(epoch + 1,
                                                          self.batch_size * self.model.num_pixels * mean_loss))
            if epoch in epochs:
              self.viz = Viz(self.model)
              if epoch == 5:
                self.model.samples(epoch = 10)
              else:
                self.model.samples(epoch = epoch+1)
        return kl,recon
    def _train_epoch(self, data_loader):
        epoch_loss = 0.
        epoch_kl = 0.
        epoch_recon = 0.
        print_every_loss = 0. 
        for batch_idx, (data, label) in enumerate(data_loader):
            iter_loss,kl_loss,recon_loss = self._train_iteration(data)
            epoch_loss += iter_loss
            epoch_kl += kl_loss
            epoch_recon += recon_loss
            print_every_loss += iter_loss
            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                print('{}/{}\tLoss: {:.3f}'.format(batch_idx * len(data),
                                                  len(data_loader.dataset),
                                                  self.model.num_pixels * mean_loss))
                print_every_loss = 0.
        return epoch_loss / len(data_loader.dataset),epoch_kl/len(data_loader.dataset),epoch_recon/len(data_loader.dataset)

    def _train_iteration(self, data):
        self.num_steps += 1
        if self.use_cuda:
            data = data.cuda()

        self.optimizer.zero_grad()
        recon_batch, latent_dist = self.model(data)
        loss,kl,recon = self._loss_function(data, recon_batch, latent_dist)
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        return train_loss,kl,recon

    def _loss_function(self, data, recon_data, latent_dist):
        
        recon_loss = F.binary_cross_entropy(recon_data,
                                            data)
        recon_loss *= self.model.num_pixels

        
        kl_cont_loss = 0  
        kl_disc_loss = 0
        cont_capacity_loss = 0
        disc_capacity_loss = 0

        if self.model.is_continuous:
            mean, logvar = latent_dist['cont']
            kl_cont_loss = self._kl_normal_loss(mean, logvar)
            cont_min, cont_max, cont_num_iters, cont_gamma = \
                self.cont_capacity
            cont_cap_current = (cont_max - cont_min) * self.num_steps / float(cont_num_iters) + cont_min
            cont_cap_current = min(cont_cap_current, cont_max)
            # Calculate continuous capacity loss
            cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)

        if self.model.is_discrete:
            kl_disc_loss = self._kl_multiple_discrete_loss(latent_dist['disc'])
            disc_min, disc_max, disc_num_iters, disc_gamma = \
                self.disc_capacity
            disc_cap_current = (disc_max - disc_min) * self.num_steps / float(disc_num_iters) + disc_min
            disc_cap_current = min(disc_cap_current, disc_max)
            disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.model.latent_space['disc']])
            disc_cap_current = min(disc_cap_current, disc_theoretical_max)
            disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

        kl_loss = kl_cont_loss + kl_disc_loss

        # Calculate total loss
        total_loss = recon_loss + cont_capacity_loss + disc_capacity_loss

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['recon_loss'].append(recon_loss.item())
            self.losses['kl_loss'].append(kl_loss.item())
            self.losses['loss'].append(total_loss.item())

        # To avoid large losses normalise by number of pixels
        return total_loss / self.model.num_pixels,recon_loss.item(),kl_loss.item()

    def _kl_normal_loss(self, mean, logvar):
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl_means = torch.mean(kl_values, dim=0)
        kl_loss = torch.sum(kl_means)

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_cont'].append(kl_loss.item())
            for i in range(self.model.latent_space['cont']):
                self.losses['kl_loss_cont_' + str(i)].append(kl_means[i].item())

        return kl_loss

    def _kl_multiple_discrete_loss(self, alphas):
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]

        # Total loss is sum of kl loss for each discrete latent
        kl_loss = torch.sum(torch.cat(kl_losses))

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_disc'].append(kl_loss.item())
            for i in range(len(alphas)):
                self.losses['kl_loss_disc_' + str(i)].append(kl_losses[i].item())

        return kl_loss

    def _kl_discrete_loss(self, alpha):
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)])
        if self.use_cuda:
            log_dim = log_dim.cuda()
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss
