import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import psnr
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import hdf5storage
import time
import tqdm

pad = 22

def savemat(filename, data_dict):
    """
    save data_dict to .mat file with version 7.3 (HDF5)
    """
    hdf5storage.write(data_dict, filename=filename, matlab_compatible=True)
    
def padr(img, pad=pad):
    pad_mod = 'reflect'
    img_pad = F.pad(input=img, pad=(pad,pad,pad,pad), mode=pad_mod)
    return img_pad
    
class Test(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, test_data_loader, val_data_loader,
                  lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer=optimizer, config=config)
        if torch.__version__ >= "2.0.0":
            self.origin_model = model
            self.model = torch.compile(model)
        self.config = config
        self.len_epoch = len_epoch
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.do_test = True
        self.gamma = 1.0
        self.lr_scheduler = lr_scheduler


        self.train_metrics = MetricTracker('Total_loss', writer=self.writer)
        self.test_metrics = MetricTracker('psnr', 'ssim', writer=self.writer)
        
        # make directories for output
        output_dir = '../output'
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir + '/C', exist_ok=True)
        os.makedirs(output_dir + '/GT', exist_ok=True)
        os.makedirs(output_dir + '/N', exist_ok=True)
        os.makedirs(output_dir + '/I', exist_ok=True)
        os.makedirs(output_dir + '/benchmark', exist_ok=True)
        os.makedirs(output_dir + '/benchmark/I', exist_ok=True)
        os.makedirs(output_dir + '/benchmark/C', exist_ok=True)
        os.makedirs(output_dir + '/benchmark/N', exist_ok=True)

    def test(self, save = True, do_validation = True, do_benchmark = True):
        self.model.eval()
        self.test_metrics.reset()
        epoch = self.start_epoch
        
        if save==True:
           os.makedirs(self.output_dir + '/C/'+str(epoch), exist_ok=True)
           os.makedirs(self.output_dir + '/N/'+str(epoch), exist_ok=True)
           os.makedirs(self.output_dir + '/benchmark/C/'+str(epoch), exist_ok=True)
           os.makedirs(self.output_dir + '/benchmark/N/'+str(epoch), exist_ok=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast(self.amp):
                time_mps_srgb = 0.0
                
                # val
                if do_validation:
                    for batch_idx, (target, input_noisy, input_GT) in enumerate(tqdm.tqdm(self.val_data_loader)):
                            input_noisy = input_noisy.to(self.device)
                            input_GT = input_GT.to(self.device)
                            
                            input_noisy = padr(input_noisy)
                            input_GT = padr(input_GT)

                            #clean = self.model(input_noisy)
                            clean = self.model.denoise(input_noisy)
                            noise_b = input_noisy - clean

                            size = [noise_b.shape[0],noise_b.shape[1],noise_b.shape[2]*noise_b.shape[3]]              
                            noise_b_normal = (noise_b-torch.min(noise_b.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1))/(torch.max(noise_b.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1)-torch.min(noise_b.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1))
                            if save==True:
                                for i in range(input_noisy.shape[0]):
                                    save_image(torch.clamp(clean[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().float().cpu(), self.output_dir + '/C/'+str(epoch)+'/'+target['dir_idx'][i]+'.PNG')
                                    save_image(torch.clamp(input_GT[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().float().cpu(), self.output_dir + '/GT/' +target['dir_idx'][i]+'.PNG')
                                    save_image(torch.clamp(noise_b_normal[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().cpu(), self.output_dir + '/N/'+str(epoch)+'/'+target['dir_idx'][i]+'.PNG')
                                    save_image(torch.clamp(input_noisy[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().float().cpu(), self.output_dir + '/I/' +target['dir_idx'][i]+'.PNG')
                    
                            self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
                            for met in self.metric_ftns:
                                if met.__name__=="psnr":
                                    psnr = met(input_GT[:,:,pad:-pad,pad:-pad].to(self.device), torch.clamp(clean[:,:,pad:-pad,pad:-pad],min=0,max=1))
                                    self.test_metrics.update('psnr', psnr)
                                elif met.__name__=="ssim":
                                    self.test_metrics.update('ssim', met(input_GT[:,:,pad:-pad,pad:-pad].to(self.device), torch.clamp(clean[:,:,pad:-pad,pad:-pad],min=0,max=1)))
                            del target
                    
                # benchmark
                if do_benchmark:
                    DenoisedBlocksSrgb = np.zeros((40*32, 256, 256, 3), dtype=np.uint8)    
                    counter = 0
                    for batch_idx, (target, input_noisy) in enumerate(tqdm.tqdm(self.test_data_loader)):
                        input_noisy = input_noisy.to(self.device)

                        input_noisy = padr(input_noisy)
                        
                        start_time = time.time()
                        #clean = self.model(input_noisy)
                        clean = self.model.denoise(input_noisy)
                        time_mps_srgb += time.time() - start_time
                        
                        noise_b = input_noisy - clean

                        size = [noise_b.shape[0],noise_b.shape[1],noise_b.shape[2]*noise_b.shape[3]]              
                        noise_b_normal = (noise_b-torch.min(noise_b.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1))/(torch.max(noise_b.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1)-torch.min(noise_b.view(size),-1)[0].unsqueeze(-1).unsqueeze(-1))
                        
                        for i in range(input_noisy.shape[0]):
                            clean_image = torch.clamp(clean[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().float()
                            DenoisedBlocksSrgb[counter] = clean_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                            counter += 1
                            if save==True:
                                save_image(clean_image, self.output_dir + '/benchmark/C/'+str(epoch)+'/'+target['dir_idx'][i]+'.PNG')
                                save_image(torch.clamp(noise_b_normal[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().cpu(), self.output_dir + '/benchmark/N/'+str(epoch)+'/'+target['dir_idx'][i]+'.PNG')
                                save_image(torch.clamp(input_noisy[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().float().cpu(), self.output_dir + '/benchmark/I/' +target['dir_idx'][i]+'.PNG')
                    
                    DenoisedBlocksSrgb = DenoisedBlocksSrgb.reshape(40, 32, 256, 256, 3)
                    # convert numpy array two 2D list (coresponding to matlab 2D cell array)
                    list_result = np.empty((40, 32), dtype=object)
                    for i in range(DenoisedBlocksSrgb.shape[0]):
                        for j in range(DenoisedBlocksSrgb.shape[1]):
                            list_result[i, j] = DenoisedBlocksSrgb[i, j]
                    time_mps_srgb = time_mps_srgb * 1024 * 1024 / (DenoisedBlocksSrgb.shape[0] * DenoisedBlocksSrgb.shape[1] * DenoisedBlocksSrgb.shape[2] * DenoisedBlocksSrgb.shape[3])
                    savemat(self.output_dir + '/SubmitSrgb.mat', 
                                            {'DenoisedBlocksSrgb': list_result, 
                                            'TimeMPSrgb': time_mps_srgb,
                                            })

        self.writer.close()
        return self.test_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

