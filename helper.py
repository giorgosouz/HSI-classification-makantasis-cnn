# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:47:27 2021

@author: vader
"""




import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
import itertools
import pandas as pd
import tifffile
from tifffile import memmap
import imageio as io
import os
import shutil






def min_max(img_list):
    
    min_val=np.ones((img_list[0].shape[-1]))*999999
    max_val=np.zeros((img_list[0].shape[-1]))
    
    for img in img_list:

        mx=np.max(img,axis=(0,1))
        mn=np.min(img,axis=(0,1))

        min_val=np.minimum(min_val,mn)
        max_val=np.maximum(max_val,mx)

            
            
    return min_val,max_val




def norm_img(img,min_val,max_val):


        img=img.astype(np.float32)
        # for i in range(img[0].shape[-1]):
            # img[:,:,i] = (img[:,:,i] - min_val[i]) /(max_val[i]  - min_val[i])
            
        img = (img - min_val) /(max_val  - min_val)
        
        
        
        
        return img
 

def store_to_disk(x,name):
    # shutil.copyfile(path,name)
    # return memmap(name, mode='r+',dtype=np.float32)
    if type(x)==str:
        x=memmap(x, mode='r')
        x=x.astype(np.float32)
        

    
    np.save(name,x)
    return np.load(name+'.npy',mmap_mode='r+')


def collect_data(img,gt,backround_labels=[0]):
  samples,labels=[],[]
  if img.shape[0] == gt.shape[0] and img.shape[1] == gt.shape[1]:
    for label in np.unique(gt):
      if label in backround_labels: continue
      else:
        ind=np.nonzero(gt == label)
        samples += list(img[ind])
        labels += len(ind[0])*[label]
  else: print ('Images have different shapes')
  return np.asarray(samples),np.asarray(labels)





def conc_images(img1,gt1,img2,gt2,dtype='int16'):

  top, bottom, left, right = [10]*4
  img1=cv2.copyMakeBorder(img1,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0])
  img2=cv2.copyMakeBorder(img2,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0])
  gt1=cv2.copyMakeBorder(gt1,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0])
  gt2=cv2.copyMakeBorder(gt2,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0])


  dif=img1.shape[1]-img2.shape[1]
  if dif <= 0:
        # if dif==0:dif=-1
        a=np.zeros((img1.shape[0],dif*(-1),img1.shape[2]), dtype=dtype)
        im1=np.concatenate((img1,a),axis=1)
        im2=np.concatenate((img2,im1),axis=0)

        a=np.zeros((img1.shape[0],dif*(-1)), dtype='int8' )
        gt1=np.concatenate((gt1,a),axis=1)
        gt2=np.concatenate((gt2,gt1),axis=1)


        return im2,gt2 
  else:
        a=np.zeros((img2.shape[0],dif,img2.shape[2]), dtype=dtype )
        im2=np.concatenate((img2,a),axis=1)
        im1=np.concatenate((img1,im2),axis=0)


        a=np.zeros((img2.shape[0],dif), dtype='int8' )
        gt2=np.concatenate((gt2,a),axis=1)
        gt1=np.concatenate((gt1,gt2),axis=0)


        return im1,gt1




def sample_gt(gt, train_size):
    ind = np.nonzero(gt)
    X = list(zip(*ind)) # x,y features
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    train_indices, test_indices = [], []
    for c in np.unique(gt):
        if c == 0:
           continue
        indices = np.nonzero(gt == c)
        X = list(zip(*indices)) # x,y features
        train, test = train_test_split(X, train_size=train_size,random_state=0)
        train_indices += train
        test_indices += test
    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]
    train_gt[train_indices] = gt[train_indices]
    test_gt[test_indices] = gt[test_indices]
    return train_gt, test_gt

def sliding_window(image, step=10, window_size=(20, 20), with_data=True):

    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step

    for x in range(0, W - w + offset_w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h

def count_sliding_window(top, step=10, window_size=(20, 20)):

    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)

def grouper(n, iterable):

    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def test(net, img, n_classes):

    net.eval()
    patch_size = 5
    center_pixel = True
    batch_size, device = 1024, torch.device("cuda" if torch.cuda.is_available() else "cpu")




    kwargs = {
            "step":1,
            "window_size": (patch_size, patch_size)}
    
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():

            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose(0, 3, 1, 2)
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs


class HY_dataset(torch.utils.data.Dataset):
    def __init__(self, data, gt):

        super(HY_dataset, self).__init__()
        self.data = data
        self.label = gt
        
        self.patch_size = 5
        self.ignored_labels = [0]
        
        self.center_pixel = True

        # Fully supervised : use all pixels with label not ignored

        mask = np.ones_like(gt)
        for l in self.ignored_labels:
            mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label


class KarankEtAl(nn.Module):

            
    def __init__(self, input_channels, n_classes, patch_size=5):
        super(KarankEtAl, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.n_classes=n_classes


        self.conv1 = nn.Conv3d(
                1, 3*self.input_channels, (1, 3, 3))
        self.conv2 = nn.Conv3d(
                3*self.input_channels, 9*self.input_channels, (1, 3, 3))
        self.features_size = self._get_final_flattened_size()
        
        self.fc1 = nn.Linear(self.features_size, 6*self.input_channels)
        self.fc2 = nn.Linear(6*self.input_channels, self.n_classes)
        



    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
      
            x = self.conv1(x)
      
            x = self.conv2(x)
      
            _, t, c, w, h = x.size()

        return t * c * w * h


    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

class KarankEtAl2(nn.Module):

            
    def __init__(self, input_channels, n_classes,n_comp=12, patch_size=5):
        super(KarankEtAl2, self).__init__()
        self.patch_size = patch_size
        self.input_channels = n_comp
        self.n_classes=n_classes
        self.n_comp=n_comp

        self.conv1 = nn.Conv3d(
                1, 3*self.input_channels, (1, 3, 3))
        self.conv2 = nn.Conv3d(
                3*self.input_channels, 9*self.input_channels, (1, 3, 3))
        self.features_size = self._get_final_flattened_size()
        
        self.fc1 = nn.Linear(self.features_size, 6*self.input_channels)
        self.fc2 = nn.Linear(6*self.input_channels, self.n_classes)
        



    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
      
            x = self.conv1(x)
      
            x = self.conv2(x)
      
            _, t, c, w, h = x.size()

        return t * c * w * h


    def forward(self, x):
        # x=torch.rand([1024, 1, 176, 5, 5])
        z=torch.empty([x.shape[0], 1, self.n_comp, 5, 5])
        for i in range(x.shape[0]):
            
            y=x[i]
            y=torch.squeeze(y)
            # print(y.shape)
            y=y.view(-1,y.shape[0])
            # print(y.shape)
            U,S,V = torch.pca_lowrank(y, q=12, center=True, niter=3)
            # print(U.shape,S.shape,V.shape)
            y=torch.matmul(y, V[:, :self.n_comp ])
            y=y.view(-1,self.patch_size,self.patch_size)
            y=torch.unsqueeze(y,0)
            # print(y.shape,z.shape)
            z[i]=y
            
        z = F.relu(self.conv1(z))

        z = F.relu(self.conv2(z))

        
        z = z.view(-1, self.features_size)
        z = self.fc1(z)
        z = self.fc2(z)
        return z



    
def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    device,
    dataset,
    val_loader=None
):


    net.to(device)


    accuracy=0.
    total=0
    iter_ = 1


    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        # Run the training loop for one epoch
        for batch_idx, (data, target) in (enumerate(data_loader)):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = net(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            
            
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if pred.item() in [0]:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
            
            iter_ += 1
            del (data, target, loss, output)
            
        # Update the scheduler
        
        avg_loss /= len(data_loader)
        print('Epoch =',e)
        print('Train loss',avg_loss,'Train ac',accuracy / total)
        
        
        if val_loader is not None:
            val_acc,val_loss = val(net, val_loader,criterion,device)
            print('Val loss',val_loss,'Val ac',val_acc)
        
        torch.save(net.state_dict(), dataset+'_'+str(e)+'_'+str(val_acc)+".pth")




def val(net, data_loader,criterion,device):
    # TODO : fix me using metrics()
    net.eval()
    accuracy, total = 0.0, 0.0
    ignored_labels = [0]
    avg_loss=0.
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            avg_loss += loss.item()
            
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    
    avg_loss /= len(data_loader)
    return accuracy / total ,avg_loss       
    

def metrics(prediction, target, ignored_labels=[], n_classes=None,label_values=None):


    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask] 
    prediction = prediction[ignored_mask]



    n_classes = np.max(target)  if n_classes is None else n_classes
    

    
    
        

        
    
    cr=classification_report(target,prediction,output_dict=True,
                             target_names=label_values)
    
    cm=confusion_matrix(target,prediction)
    
    return pd.DataFrame(cr),pd.DataFrame(cm)

    
    
