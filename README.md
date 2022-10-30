# Generative-Learning-With-Discrete-and-Continuous-VAE
Using VAE based netural network in order to learn some discrete and continuous latent state representation of that data and generate new photos.

1. [General](#General)
    - [Background](#background)
3. [Program Structure](#Program-Structure)
    - [Network Structure](#Network-Structure)
    - [Visualtions](#Visualtions)
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

<img src="https://i.imgur.com/JzaBhBb.png" width = 50% height=50%>

The dims of the convolution are 3-32-64, padding is 1 and stride is 2 and every convolution layer also includes Leaky relu function in both encoder and decoder. 

The linear layers dims encoder: 
* The linear layer dims are 64 * 4 * 4 to hidden dim. 
* The linear discrete layers dims are hidden dim to discrete dim. 
* The linear continuous layers dims are hidden dim to continuous dim. 

The linear layers dims decoder: 
* The linear layers dims are latent dim to hidden dim and hidden dim to 64 * 4 * 4 .
* The linear layers also include leaky relu function. 
* We used reparameterization trick for continuous variables and Gumbel SoftMax for the discrete variables.  

### Visualtions
Having a look about the Visualtions of the latent spaces:
* Gender:
    <img src="https://i.imgur.com/JQWq0KF.png" width = 50% height=50%>

* Hair Style:
    <img src="https://i.imgur.com/2xnlPmn.png" width = 50% height=50%>



## Installation
1. Open the terminal

2. Clone the project by:
```
    $ git clone https://github.com/elaysason/Generative-Learning-With-Discrete-and-Continuous-VAE.git
```
3. Run the main.py file by:
```
    $ python main.py
```
