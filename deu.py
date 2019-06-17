
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import math
import random

from deu_gravitation_activation import deu_activation
import pdb

class DeuHorse(torch.autograd.Function):

    def forward(self, x, deu_coeffs):

        self.save_for_backward(x, deu_coeffs)

        output = torch.zeros(x.size())
        for i, k in enumerate(deu_coeffs):
            output[:, i] = deu_activation(x[:, i], *k)
        return output.type_as(x)

    def backward(self, grad_output):
        x, deu_coeffs = self.saved_tensors
        grad_in = torch.zeros(x.size()).type_as(x)
        grad_coeff_a = torch.zeros(x.size()).type_as(x)
        grad_coeff_b = torch.zeros(x.size()).type_as(x)
        grad_coeff_c = torch.zeros(x.size()).type_as(x)
        grad_coeff_c1 = torch.zeros(x.size()).type_as(x)
        grad_coeff_c2 = torch.zeros(x.size()).type_as(x)
        grad_coeff = torch.zeros(x.size()[1], 5).type_as(x)
        for i, k in enumerate(deu_coeffs):
            fp, fa, fb, fc, fc1, fc2 = deu_activation(x[:, i], *k, derivative=True)
            grad_in[:, i] = fp
            grad_coeff_a[:, i] = fa            
            grad_coeff_b[:, i] = fb 
            grad_coeff_c[:, i] = fc 
            grad_coeff_c1[:, i] = fc1 
            grad_coeff_c2[:, i] = fc2 
        grad_in *= grad_output
        grad_coeff[:, 0] =  torch.sum(grad_output * grad_coeff_a, 0)
        grad_coeff[:, 1] =  torch.sum(grad_output * grad_coeff_b, 0)
        grad_coeff[:, 2] =  torch.sum(grad_output * grad_coeff_c, 0)
        grad_coeff[:, 3] =  torch.sum(grad_output * grad_coeff_c1, 0)
        grad_coeff[:, 4] =  torch.sum(grad_output * grad_coeff_c2, 0)
        
        return grad_in, grad_coeff


class Deu(torch.nn.Module):
    def __init__(self, n_neurons):
        super(Deu, self).__init__()
        self.n_neurons = n_neurons
        self.coeffs = torch.nn.Parameter(self.init_coeffs(self.n_neurons))

    def forward(self, x):
        deu_horse = DeuHorse()
        return deu_horse(x, self.coeffs)

    @staticmethod
    def init_coeffs(n_neurons):
        coeffs = torch.empty(n_neurons, 5).uniform_(0, 1)
        coeffs[:,3] = 0
        coeffs[:,4] = 0
        return coeffs

     
