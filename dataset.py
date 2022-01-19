# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:28:34 2021

@author: vader
"""
import imageio as io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA,IncrementalPCA
from helper import collect_data,conc_images,min_max,norm_img,store_to_disk
from tifffile import memmap
import os
import cv2

def get_dataset(dataset):
    if dataset == 'HyRANK1':

        
        if  not os.path.exists('Data1'):
            
            
            
            
            
            
            
            os.mkdir('Data1')
        
            img1=io.imread('HyRANK_satellite/TrainingSet/Dioni.tif')
            img2=io.imread('HyRANK_satellite/TrainingSet/Loukia.tif')
            gt1=io.imread('HyRANK_satellite/TrainingSet/Dioni_GT.tif')
            gt2=io.imread('HyRANK_satellite/TrainingSet/Loukia_GT.tif')
            
            Erato=io.imread('HyRANK_satellite/ValidationSet/Erato.tif')
            Kirki=io.imread('HyRANK_satellite/ValidationSet/Kirki.tif')
            Nefeli=io.imread('HyRANK_satellite/ValidationSet/Nefeli.tif')
            
            min_val,max_val=min_max([img1,img2,Erato,Nefeli,Kirki])
    

    
            
            pca = IncrementalPCA(n_components=10)
    
            for img in (img1,img2,Erato,Kirki,Nefeli):
                img=norm_img(img, min_val, max_val)
                or_shp=img.shape
                pca.partial_fit(img.reshape(-1,or_shp[2]))      
                print('fitted PCA')
    
            img_list=[]
            for img in (img1,img2,Erato,Kirki,Nefeli):
                or_shp=img.shape        
                img=pca.transform(img.reshape(-1,or_shp[2]))
                img=img.astype(np.float32)
                print('transformed PCA')
                img=img.reshape(or_shp[0],or_shp[1],-1)
                img_list.append(img)
            
            min_val2,max_val2=min_max(img_list)
            
            for img,name in zip(img_list,(1,2,3,4,5)):
                img=norm_img(img, min_val2, max_val2)
                np.save(('Data1/IMG'+str(name)) ,img)
                


        img1=np.load('Data1/IMG1.npy')
        img2=np.load('Data1/IMG2.npy')

        gt1=io.imread('HyRANK_satellite/TrainingSet/Dioni_GT.tif')
        gt2=io.imread('HyRANK_satellite/TrainingSet/Loukia_GT.tif')

        img,gt=conc_images(img1, gt1, img2, gt2)
        Erato=np.load('Data1/IMG3.npy')
        Kirki=np.load('Data1/IMG4.npy')
        Nefeli=np.load('Data1/IMG5.npy')

        
        
        
        return img,gt,Erato,Kirki,Nefeli
    
    
    
    elif dataset=='HyRANK2':

        

        
            

            
            
            
            
            
            
        if  not os.path.exists('Data2'):
            
            
            
            
            
            
            
            os.mkdir('Data2')
            
    
            
            
            
            img1=store_to_disk('Image1.tif','Data2/IM1')
            img2=store_to_disk('Image2.tif','Data2/IM2')
            img3=store_to_disk('Image3.tif','Data2/IM3')
            img4=store_to_disk('Image4.tif','Data2/IM4')
            img5=store_to_disk('Image5.tif','Data2/IM5')
    
            gt1=io.imread('GT1.tif')
            gt2=io.imread('GT2.tif')
            gt5=io.imread('GT5.tif')
            
            
            min_val,max_val=min_max([img1,img2,img3,img4,img5])
    
            pca = IncrementalPCA(n_components=30)
            
            for img in (img1,img2,img3,img4,img5):
                img[:]=norm_img(img, min_val, max_val)
                or_shp=img.shape
                pca.partial_fit(img.reshape(-1,or_shp[2]))      
                print('fitted PCA')
            
    

            
            img_list=[]
            for img in (img1,img2,img3,img4,img5):
                #img[:]=norm_img(img, min_val, max_val)
                or_shp=img.shape        
                img=pca.transform(img.reshape(-1,or_shp[2]))
                print('transformed PCA')
                img=img.astype(np.float32)

                img=img.reshape(or_shp[0],or_shp[1],-1)
                img_list.append(img)

            del img1,img2,img3,img4,img5


            
            
            
            
            min_val2,max_val2=min_max(img_list)
            for img,name in zip(img_list,(1,2,3,4,5)):
                img=norm_img(img, min_val2, max_val2)
                np.save(('Data2/IMG'+str(name)) ,img)

        
        
        img1=np.load('Data2/IMG1.npy', mmap_mode='r')
        img2=np.load('Data2/IMG2.npy', mmap_mode='r')
        img3=np.load('Data2/IMG3.npy', mmap_mode='r')
        img4=np.load('Data2/IMG4.npy', mmap_mode='r')
        img5=np.load('Data2/IMG5.npy', mmap_mode='r')
        
        gt1=io.imread('GT1.tif')
        gt2=io.imread('GT2.tif')
        gt5=io.imread('GT5.tif')

            
            

        

        
        
        return img1,img2,img3,img4,img5,gt1,gt2,gt5
