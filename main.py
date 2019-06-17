
from deu import Deu
import numpy as np
import torch
import sklearn.datasets
from torch import nn
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

noisy_circles = sklearn.datasets.make_circles(n_samples=500, factor=.5, noise=.05)
d = noisy_circles[0]
target = noisy_circles[1]
t_train = torch.FloatTensor(d[:100])
y_train = torch.LongTensor(target[:100])

batch_size = 1
lr = 0.001


n_epoch = 600
learning_rate = 1e-1
n_in = np.shape(t_train)[-1]
n_out = 2


f = Deu(4)

net = nn.Sequential(
        nn.Linear(n_in, 4),
        f,
        nn.Linear(4, 4),
        nn.Linear(4, n_out),
)

params = list(net.named_parameters())


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(n_epoch):
    
    running_loss = 0.0
    inputs, labels = t_train, y_train
    inputs, labels = Variable(inputs), Variable(labels)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    
    if epoch % 100 == 0:
        print('%i loss: %.4f' % (epoch + 1, loss.data))
        for p in params:
            if 'coeff' in p[0]:
                a =  p[1][0,0].data
                b =  p[1][0,1].data
                c =  p[1][0,2].data
                print "DEU_0: eq: {}y'' + {}y' + {}y = UnitStep(t)".format(a,b,c)
                
        



