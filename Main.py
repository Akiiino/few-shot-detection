
# coding: utf-8

# In[20]:


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable

import os
import os.path

import pandas as pd
import numpy as np

from PIL import Image
from collections import namedtuple

from tqdm import trange as trange, tqdm as tqdm

np.random.seed(1337)


# In[2]:


import dataset


# In[3]:


USE_CUDA = False
n_classes = 5


# In[205]:


class HeatmapNet(nn.Module):
    def __init__(self, reg_coef=0.5, verbose=False):
        super(HeatmapNet, self).__init__()
        self.extractor = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(30, 45, kernel_size=3, padding=1),
            nn.BatchNorm2d(45),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(45, 65, kernel_size=3, padding=1),
            nn.BatchNorm2d(65),
            nn.ReLU(),
            nn.Conv2d(65, n_classes, kernel_size=1, padding=0),
        )
        
        self.verbose = verbose
        self.reg_coef = reg_coef


    def get_heatmap(self, x):
        heatmap = self.extractor(x)
        
        return heatmap

    
    def forward(self, x):
        heatmap = self.extractor(x)
        
        x = F.avg_pool2d(heatmap, heatmap.shape[-2:])
        x = x.view(x.shape[0], -1)
        
        return x
    
    
    def _get_loss(self, pred, target, heatmap=None):
        loss = F.cross_entropy(pred, target)
        
        if self.training:
            reg_loss = self.reg_coef * (heatmap.abs()).mean()
            loss += reg_loss
            
        return loss

    
    def _get_acc(self, pred, target):
        pred = pred.data.max(1, keepdim=True)[1]
        acc = pred.eq(target.data.view_as(pred)).float().mean()

        return acc


    def get_metrics(self, input_, target):
        input_ = Variable(input_)
        target = Variable(target)
        
        heatmap = self.get_heatmap(input_)
        pred = F.avg_pool2d(heatmap, heatmap.shape[-2:]).view(heatmap.shape[0], -1)
        
        return self._get_loss(pred, target, heatmap), self._get_acc(pred, target)


    def train_epoch(self, batch_iter, optimizer):
        self.train()

        smooth_loss = None
        smooth_acc = None

        def update_smooth(smooth, val, gamma=0.99):
            if smooth is not None:
                return (smooth * gamma + val * (1-gamma))
            else:
                return val

        if self.verbose:
            batch_iter = tqdm(batch_iter)
        else:
            batch_iter = batch_iter

        for data, target in batch_iter:
            optimizer.zero_grad()

            if USE_CUDA:
                data = data.cuda()
                target = target.cuda()

            loss, accuracy = self.get_metrics(data, target)        

            loss.backward()
            optimizer.step()

            smooth_loss = update_smooth(smooth_loss, loss.data.mean())
            smooth_acc = update_smooth(smooth_acc, accuracy)

            if self.verbose:
                batch_iter.set_postfix_str('loss: {:.6f} | acc: {:.6f}'.format(smooth_loss, smooth_acc))

        return smooth_loss, smooth_acc

    def test_epoch(self, test_batch_iter):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_batch_iter:
            if USE_CUDA:
                data = data.cuda()
                target = target.cuda()

            loss, accuracy = model.get_metrics(data, target)

            test_loss += loss.data[0] * len(data)
            correct += accuracy * len(data)

        test_loss /= len(test_batch_iter.dataset)
        correct /= len(test_batch_iter.dataset)
        
        return test_loss, correct


    def train_epochs(self, batch_iter, test_batch_iter, optimizer, epochs=10, verbose=False):
        model.train()

        for epoch in range(epochs):
            self.train_epoch(batch_iter, optimizer)
            loss, acc = self.test_epoch(test_batch_iter)
            if self.verbose:
                print('epoch {} | loss: {:.6f} | accuracy: {:.2f}%\n'.format(epoch, loss, 100*acc))


# In[206]:


t = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

train_loader = data.DataLoader(datasets.ImageFolder("imagenet/train/", transform=t), 32, shuffle=True)
test_loader = data.DataLoader(datasets.ImageFolder("imagenet/val/", transform=t), 32, shuffle=True)


# In[207]:


n_classes = len(train_loader.dataset.classes)


# In[209]:


# train_loader, test_loader = dataset.make_split("proc_data", n_classes=n_classes, examples_per_class=3)


# In[210]:


model = HeatmapNet(0, verbose=True)

if USE_CUDA:
    model.cuda()

optimizer = optim.Adam(model.parameters())


# In[211]:


model.train_epochs(train_loader, test_loader, optimizer, verbose=True, epochs=100)


# In[212]:


import sys
sys.exit()


# In[13]:


def train(loader, epoch, k=10, verbose=True):
    model.train()
    
    smooth_loss = None
    smooth_acc = None
    
    def update_smooth(smooth, val, gamma=0.99):
        if smooth is not None:
            return (smooth * gamma + val * (1-gamma))
        else:
            return val
    
    if verbose:
        loader = tqdm(loader)
    else:
        loader = loader
    
    phi = model.state_dict()
    
    for tr_data in loader:
        for i in range(k):
            for batch in tr_data[:-1]:
                optimizer.zero_grad()

                target = torch.arange(batch.shape[0]).long()

                if USE_CUDA:
                    batch = batch.cuda()
                    target = target.cuda()
                
                loss, accuracy = model.get_metrics(batch, target)        

                loss.backward()
                optimizer.step()

        loss, accuracy = model.get_metrics(tr_data[-1], target)
                
        smooth_loss = update_smooth(smooth_loss, loss.data.mean())
        smooth_acc = update_smooth(smooth_acc, accuracy)
        
        W = model.state_dict()
        
        phi = {
            m: phi[m] + (W[m] - phi[m]) / k
            for m in phi
        }

        model.load_state_dict(phi)
        

        if verbose:
            loader.set_postfix_str('epoch: {} | loss: {:.6f} | acc: {:.6f}'.format(epoch, smooth_loss, smooth_acc))
        
    return smooth_loss, smooth_acc


# In[14]:


def mb_train(loader, epoch, k=10, meta_batch_size=2, verbose=True):
    model.train()
    
    smooth_loss = None
    smooth_acc = None
    
    def update_smooth(smooth, val, gamma=0.99):
        if smooth is not None:
            return (smooth * gamma + val * (1-gamma))
        else:
            return val
    
    if verbose:
        loader = tqdm(loader)
    else:
        loader = loader
    
    phi = model.state_dict()
    
    for meta_batch, tr_data in enumerate(loader):
        for i in range(k):
            for batch in tr_data[:-1]:
                optimizer.zero_grad()

                target = torch.arange(batch.shape[0]).long()

                if USE_CUDA:
                    batch = batch.cuda()
                    target = target.cuda()
                
                loss, accuracy = model.get_metrics(batch, target)        

                loss.backward()
                optimizer.step()

        loss, accuracy = model.get_metrics(tr_data[-1], target)
                
        smooth_loss = update_smooth(smooth_loss, loss.data.mean())
        smooth_acc = update_smooth(smooth_acc, accuracy)
        
        W = model.state_dict()
        
        if meta_batch % meta_batch_size == 0:
            weight_diff = {
                m: (W[m] - phi[m]) / k
                for m in phi
            }
        else:
            weight_diff = {
                m: weight_diff[m] + (W[m] - phi[m]) / k
                for m in phi
            }

        if (meta_batch + 1) % meta_batch_size == 0:
            phi = {
                m: phi[m] + weight_diff[m] / meta_batch_size
                for m in phi
            }

            model.load_state_dict(phi)
        

        if verbose:
            loader.set_postfix_str('epoch: {} | loss: {:.6f} | acc: {:.6f}'.format(epoch, smooth_loss, smooth_acc))
        
    return smooth_loss, smooth_acc


# In[15]:


def test(loader, k=10, name="Test set"):
    model.eval()
    test_loss = 0
    correct = 0
    for test_task in loader:
        state = model.state_dict()
        
        for i in range(k):
            for batch in test_task[:-1]:
                optimizer.zero_grad()

                target = torch.arange(batch.shape[0]).long()

                if USE_CUDA:
                    batch = batch.cuda()
                    target = target.cuda()
                
                loss, accuracy = model.get_metrics(batch, target)        

                loss.backward()
                optimizer.step()

        loss, accuracy = model.get_metrics(test_task[-1], target)
        test_loss += loss.data[0] * (len(test_task) - 1)
        correct += accuracy * (len(test_task) - 1)

        model.load_state_dict(state)
        
    test_loss /= len(loader.dataset)
    print('{} | loss: {:.6f} | accuracy: {}/{} ({:.2f}%)\n'.format(
        name, test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)
    ))
    


# In[16]:


for i in range(100):
    train(train_loader, i)
    test(test_loader)


# In[ ]:


for tr_data in test_loader:
    for i in range(3):
        for batch in tr_data[:-1]:
            optimizer.zero_grad()

            target = torch.arange(batch.shape[0]).long()

            if USE_CUDA:
                batch = batch.cuda()
                target = target.cuda()

            loss, accuracy = model.get_metrics(batch, target)        

            loss.backward()
            optimizer.step()

    loss, accuracy = model.get_metrics(tr_data[-1], target)        

    break


# In[ ]:


def test(loader, name="Test set"):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        loss, accuracy = model.get_metrics(data, target)
        
        test_loss += loss.data[0] * len(data)
        correct += accuracy * len(data)

    test_loss /= len(loader.dataset)
    print('{} | loss: {:.6f} | accuracy: {}/{} ({:.2f}%)\n'.format(
        name, test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)
    ))


# In[ ]:


epoch_iter = trange(1, 5000)

for epoch in epoch_iter:
    loss, acc = train(train_loader, epoch, verbose=True)
    epoch_iter.set_postfix_str("loss: {:.6f}, acc: {:.6f}".format(loss, acc))
    
    # if epoch % 100 == -1:
    # test(test_loader, "Test set")
    # test(resampled_test_loader, "Resampled test set")


# In[ ]:


fig, axs = plt.subplots(29, 7, figsize=(11, 50))
predict = model.forward(Variable(batch))[1].data.numpy()[0]
print(np.argmax(predict), class_)

predict_ = np.argmax(predict)
image = image

axs[0][0].imshow(image)
axs[0][0].set_xticks([])
axs[0][0].set_yticks([])

plt.margins(0)

# heatmap -= heatmap.min()
# heatmap /= heatmap.max()

if True:
    high = max(np.abs(heatmap.min()), np.abs(heatmap.max()))
    low = -high
else:
    high = heatmap.max()
    low = heatmap.min()
    
for i, (ax, img, conf) in enumerate(zip(axs.flatten()[1:], list(heatmap), predict)):
    # ax.imshow(image)
    
    ax.imshow(
        img,
        cmap="seismic",
        vmin=low,
        vmax=high,
        alpha=1
    )
    ax.set_title("{}: {:.3f}{}{}".format(i, conf, "; T" if i==class_ else "", "; P" if i==predict_ else ""))
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()

# plt.savefig("fig.jpg")


# In[ ]:


temp = nn.Conv2d(FEAT_NUM, num_classes, 1)


# In[ ]:


temp.weight.shape


# In[ ]:


model.dense.weight.shape


# In[ ]:


from collections import defaultdict
exs = defaultdict(list)
counts = defaultdict(int)

s = DatasetFolder(
    "rtsd-r3/train",
    class_to_idx=train_loader.dataset.class_to_idx,
    transform=transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
)


for img, t in iter(s):
    img_np = img.numpy().transpose(1, 2, 0)
    exs[t].append(img_np)
    counts[t] += 1


# In[ ]:


len(exs)


# In[ ]:


fig, axs = plt.subplots(16, 7, figsize=(11, 25))

plt.margins(0)

for i in exs.keys():
    axs.flatten()[i].imshow(np.mean(exs[i]))
    axs.flatten()[i].set_title("{}, total {}".format(i, counts[i]))
    axs.flatten()[i].set_xticks([])
    axs.flatten()[i].set_yticks([])
    
# axs.flatten()[max(exs.keys()) + 1].imshow(np.mean(list(exs.values()), axis=0) / 200)
fig.tight_layout()

# plt.savefig("ex.jpg")

