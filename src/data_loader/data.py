import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import time 
import os
import scipy.io
from scipy.ndimage.interpolation import rotate
from multiprocessing import Pool
import glob
import cv2
import tqdm

def horizontal_flip(image, rate=0.5):
    image = image[:, ::-1, :]
    return image


def vertical_flip(image, rate=0.5):
    image = image[::-1, :, :]
    return image

def random_rotation(image, angle):
    h, w, _ = image.shape
    image = rotate(image, angle)
    return image
    
class benchmark_data(Dataset):

    def __init__(self, data_dir, task, cache_data=True):

        self.task = task
        # load SIDD_Medium_sRGB and SIDD_Benchmark_Data
        medium_data_dir = '../../SIDD_Medium_Srgb/Data'
        self.noisy_path = glob.glob(os.path.join(medium_data_dir , '*/*NOISY*.PNG'), recursive=True) # GT or NOISY
        benchmark_data_dir = '../../SIDD_Benchmark_Data'
        self.noisy_path += glob.glob(os.path.join(benchmark_data_dir, '*/*NOISY*.PNG'), recursive=True) # GT or NOISY
        self.cache_data = cache_data
        if self.cache_data and self.task=="train":
            self.noisy_images = []
            print("Loading training images into memory ...")
            for path in tqdm.tqdm(self.noisy_path):
                self.noisy_images.append(cv2.imread(path)) 
        
        self.data_dir = data_dir
        # load SIDD validation data
        self.validation_gt = scipy.io.loadmat(self.data_dir+'ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']
        self.validation_noisy = scipy.io.loadmat(self.data_dir+'ValidationNoisyBlocksSrgb.mat')['ValidationNoisyBlocksSrgb']
        self.validation_gt = self.validation_gt.reshape([-1,256,256,3])
        self.validation_noisy = self.validation_noisy.reshape([-1,256,256,3])
        # load SIDD benchmark data
        self.benchmark_noisy = scipy.io.loadmat(self.data_dir+'BenchmarkNoisyBlocksSrgb.mat')['BenchmarkNoisyBlocksSrgb']
        self.benchmark_noisy = self.benchmark_noisy.reshape([-1,256,256,3])
        self.data_num = self.validation_noisy.shape[0] + self.benchmark_noisy.shape[0]
 
        self.indices = self._indices_generator()
        self.patch_size = 80

    def __len__(self):
        if self.task == "test": 
            return self.validation_noisy.shape[0]
        elif self.task == 'val':
            return self.benchmark_noisy.shape[0]
        else:
            # make epoch image number N times of actual image number
            N = 1
            return self.data_num + len(self.noisy_path) * N

    def __getitem__(self, index):

        def data_loader():
            if self.task=="test":
                img_noisy = self.benchmark_noisy[index]
                img_noisy = (np.transpose(img_noisy,(2, 0, 1))/255)
                return np.array(img_noisy, dtype=np.float32), index   
            elif self.task=="val": 
                img_noisy = self.validation_noisy[index]
                img_gt= self.validation_gt[index]
                img_noisy = (np.transpose(img_noisy,(2, 0, 1))/255)
                img_gt= (np.transpose(img_gt,(2, 0, 1))/255)   
                return np.array(img_noisy, dtype=np.float32),  np.array(img_gt, dtype=np.float32), index
            elif self.task=="train":
                if index < self.validation_noisy.shape[0]:
                    img_noisy = self.validation_noisy[index]
                elif index >= self.validation_noisy.shape[0] and index < self.data_num:
                    img_noisy = self.benchmark_noisy[index-self.validation_noisy.shape[0]]
                else:
                    if self.cache_data and self.task=="train":
                        img_noisy = self.noisy_images[(index-self.data_num) % len(self.noisy_path)]
                    else:
                        img_noisy = cv2.imread(self.noisy_path[(index-self.data_num) % len(self.noisy_path)])
                
                # random crop
                x_00 = torch.randint(0, img_noisy.shape[0] - self.patch_size, (1,))
                y_00 = torch.randint(0, img_noisy.shape[1] - self.patch_size, (1,))
                img_noisy = img_noisy[x_00[0]:x_00[0] + self.patch_size, y_00[0]:y_00[0] + self.patch_size]
                
                # Augmentation
                horizontal = torch.randint(0,2, (1,))
                vertical = torch.randint(0,2, (1,))
                rand_rot = torch.randint(0,4, (1,))
                rot = [0,90,180,270]
                if horizontal ==1:
                    img_noisy = horizontal_flip(img_noisy)
                if vertical ==1:
                    img_noisy = vertical_flip(img_noisy)
      
                img_noisy = random_rotation(img_noisy,rot[rand_rot])
                img_noisy = (np.transpose(img_noisy,(2, 0, 1))/255)
                
                return np.array(img_noisy, dtype=np.float32), index 

        if torch.is_tensor(index):
            index = index.tolist()
            
        if self.task=="val":
            input_noisy, input_GT, idx = data_loader()
            target = {'dir_idx': str(idx)}
            return target, input_noisy, input_GT
        else:
            input_noisy, idx = data_loader()
            target = {'dir_idx': str(idx)}
            return target, input_noisy
    
    def _indices_generator(self):
        return np.arange(self.data_num,dtype=int)
   
        

if __name__ == "__main__":
    time_print = True

    prev = time()
