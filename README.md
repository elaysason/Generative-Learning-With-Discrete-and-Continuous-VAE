# Generative-Learning-With-Discrete-and-Continuous-VAE
Using VAE based netural network in order to learn the disiturbtion space which the photos are getting sampled and generate new photos.

1. [General](#General)
    - [Background](#background)
3. [Program Structure](#Program-Structure)
    - [Network Structure](#Network-Structure)
5. [Installation](#Installation)

## General
The goal is to bulid deep learning based generative model. In order to that the network learn the disiturbtion space of the photos and sample from it.

### Background
VAE is an autoencoder which provides a probabilistic manner for describing an observation in latent space. Thus, rather than building an encoder which outputs a single value to describe each latent state attribute, it will formulate encoder to describe a probability distribution for each latent attribute.

<img src="https://i.imgur.com/mDgus7e.png" width = 50% height=50%>

## Program Structure
* models.py - Creation of the VAE netowrk.
* main.py - loads a existing model or create one and plot visualizations of it
* visualizations.py - Responible for the visualizations of the model.
* training.py - trains the model

### Network-Structure
The VAE network is described as follows: 

<img src="[https://i.imgur.com/mDgus7e.pn](https://i.imgur.com/sKmNDhV.png)" width = 50% height=50%>
