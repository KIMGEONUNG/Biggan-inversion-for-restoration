from models_dgp import Generator, Discriminator
import torch
import pickle


path_config = './checkpoints/config.pickle'
path_ckpt_D = 'checkpoints/D_256.pth'

with open(path_config, 'rb') as f:
    config = pickle.load(f)

G = Generator(**config)
D = Discriminator(**config)

D.load_state_dict(
    torch.load(path_ckpt_D))
