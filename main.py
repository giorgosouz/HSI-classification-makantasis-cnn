# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:53:14 2021

@author: vader
"""

import imageio as io
import numpy as np
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from helper import *
from dataset import get_dataset
import cv2

dataset='HyRANK2'
checkpoint=None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
leak=True


if dataset=='HyRANK1':
    img,gt,Erato,Kirki,Nefeli=get_dataset(dataset)
    
    train_gt,val_gt=sample_gt(gt, 0.8)
    
    train_dataset = HY_dataset(img, train_gt)
    train_loader = data.DataLoader(
              train_dataset,
              batch_size=4,
              pin_memory=device,
              shuffle=True
          )



    if leak==True:
        
        gt1=io.imread('Erato_GT.tif')
        gt2=io.imread('Kirki_GT.tif')
        gt3=io.imread('Nefeli_GT.tif')
    
            
        val1_dataset = HY_dataset(Erato, gt1)
        
        val2_dataset = HY_dataset(Kirki, gt2)
        
        val3_dataset = HY_dataset(Nefeli, gt3)
        
        val_dataset=data.ConcatDataset([val1_dataset,val2_dataset,val3_dataset])
        val_loader = data.DataLoader(
                  val_dataset,
                  batch_size=2048,
                  shuffle=True,
                  pin_memory=device
              )

        
    else:
        
        
        
        
        
        
        val_dataset = HY_dataset(img, val_gt)
        val_loader = data.DataLoader(
                  val_dataset,
                  batch_size=2048,
                  shuffle=True,
                  pin_memory=device
              )

    n_bands=img.shape[2]
    n_classes=len(np.unique(gt))
    model=KarankEtAl(n_bands,n_classes)
    weights = torch.ones(n_classes)
    weights = weights.to(device)
    weights[torch.LongTensor([0])] = 0.
    epochs=20
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
elif dataset=='HyRANK2':
    img1,img2,img3,img4,img5,gt1,gt2,gt5=get_dataset(dataset)
    
    
    train_gt1,val_gt1=sample_gt(gt1, 0.8)
    train_gt2,val_gt2=sample_gt(gt2, 0.8)
    train_gt5,val_gt5=sample_gt(gt5, 0.8)



    train_dataset1 = HY_dataset(img1, train_gt1)
    train_dataset2 = HY_dataset(img2, train_gt2)
    train_dataset3 = HY_dataset(img5, train_gt5)
    
    train_dataset=data.ConcatDataset([train_dataset1,train_dataset2,train_dataset3])





    train_loader = data.DataLoader(
              train_dataset,
              batch_size=512,
              pin_memory=device,
              shuffle=True
          )



    if leak==True:
    

        gt3=io.imread('GT3.tif')
        gt4=io.imread('GT4.tif')
    
        val1_dataset = HY_dataset(img3, gt3)
        
        val2_dataset = HY_dataset(img4, gt4)
        
    
        
        val_dataset=data.ConcatDataset([val1_dataset,val2_dataset])
        val_loader = data.DataLoader(
                  val_dataset,
                  batch_size=2048,
                  shuffle=True,
                  pin_memory=device
              )
        
    else:
        

        val1_dataset = HY_dataset(img1, val_gt1)
        
        val2_dataset = HY_dataset(img2, val_gt2)
        
        val3_dataset = HY_dataset(img5, val_gt5)
        
        val_dataset=data.ConcatDataset([val1_dataset,val2_dataset,val3_dataset])
        val_loader = data.DataLoader(
                  val_dataset,
                  batch_size=2048,
                  shuffle=True,
                  pin_memory=device
              )


    n_bands=img1.shape[2]
    n_classes=18
    model=KarankEtAl(n_bands,n_classes)
    weights = torch.ones(n_classes)
    weights = weights.to(device)
    weights[torch.LongTensor([0])] = 0.
    epochs=20
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss(weight=weights)

    






 
with torch.no_grad():
            for input, _ in train_loader:
                break
            summary(model.to(device), input.size()[1:])



if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))

try:
            train(model,optimizer,criterion,train_loader,epoch=epochs,
                  dataset=dataset,val_loader=val_loader,device=device)
except KeyboardInterrupt:
            # Allow the user to stop the training
  pass


if dataset=='HyRANK1':

    for img,name in [(img,'Train'),(Erato,'Erato'),(Kirki,'Kirki'),(Nefeli,'Nefeli')]:
        top, bottom, left, right = [10]*4
        img=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0])
        probabilities = test(model, img, n_classes)
        prediction = np.argmax(probabilities, axis=-1)
        prediction = prediction.astype(np.int8)
        io.imsave(name+'NN'+'.tif', prediction)
elif dataset=='HyRANK2':
    for img,name in [(img1,'image1'),(img2,'image2'),(img5,'image5'),(img3,'image3'),(img4,'image4')]:
        top, bottom, left, right = [10]*4
        img=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0])
        probabilities = test(model, img, n_classes)
        prediction = np.argmax(probabilities, axis=-1)
        prediction = prediction.astype(np.int8)
        io.imsave(name+'NN'+'.tif', prediction)


