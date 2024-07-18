import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import psnr
from torchvision.utils import save_image
import torch.nn.functional as F
from torch import autograd
import os
from model.neighbor2neighbor import Neighbor2Neighbor
from model.non_local_means import NonLocalMeansDenoiser
from model.masker import Masker
from model.gaussian_filter import GaussianFilter

pad = 22

def padr(img, pad=pad):
    pad_mod = 'reflect'
    img_pad = F.pad(input=img, pad=(pad, pad, pad, pad), mode=pad_mod)
    return img_pad

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, val_data_loader,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer=optimizer, config=config)
        
        if torch.__version__ >= "2.0.0":
            self.origin_model = model
            self.model = torch.compile(model)
        
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.val_data_loader = val_data_loader
        self.do_test = self.val_data_loader is not None
        self.do_test = True
        self.gamma = 1.0
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.neighbor2neighbor = Neighbor2Neighbor()
        self.nlm_denoiser = NonLocalMeansDenoiser()
        self.masker = Masker()
        self.gaussian_filter = GaussianFilter(channels=1, kernel_size=5, sigma=25/255.0).to(self.device)

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

    def _train_epoch(self, epoch):

        self.model.train()
        self.train_metrics.reset()
       
        scaler = torch.cuda.amp.GradScaler()
        for batch_idx, (target, input_noisy) in enumerate(self.data_loader):
            input_noisy = input_noisy.to(self.device)
            #input_noisy = padr(input_noisy)
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.amp):
                # model forward and compute noise
                clean, noise = self.model(input_noisy)
                #noise = input_noisy - clean
                # clean feeded to model; noise feeded to model
                clean_c, noise_c = self.model(clean)
                clean_n, noise_n = self.model(noise)

                # loss_main
                loss_main = self.criterion[0](input_noisy, clean, noise, self.gaussian_filter)
                
                # loss_aug
                loss_aug = self.criterion[1](clean_c, clean) +  self.criterion[1](noise_n, noise)
                for i in range(2):
                    alpha = np.random.rand()*0.2+0.8  # [0.2, 1]
                    beta = np.random.rand()*2         # [0, 2]
                    
                    sim_input_noisy = alpha * clean + beta * noise
                    sim_clean, sim_noise = self.model(sim_input_noisy)
                    loss_aug = loss_aug + self.criterion[1](sim_clean, alpha * clean) + self.criterion[1](sim_noise, beta * noise)
                
                # neighbor2neighbor
                #loss_neighbor2neighbor = self.neighbor2neighbor.train(input_noisy, self.model, weight=0.05)  # slightly better
                #loss_neighbor2neighbor = self.neighbor2neighbor.train_origin(input_noisy, clean, self.model, weight=1e-3,
                #                                                            gamma1=1.0, gamma2=1.0)
                
                #loss_neighbor2neighbor = self.neighbor2neighbor.train_origin_l1(input_noisy, clean, self.model, weight=1e-3,
                #                                                             gamma1=1.0, gamma2=1.0)
                
                # blind2unblind
                # loss_blind2unblind = self.masker.train(input_noisy, self.model, weight=0.05)
                
                # drdd
                #loss_drdd = self.drdd.train(input_noisy, noise_w, noise_b, clean, weight=0.1)
                
                loss_total = loss_main + loss_aug
                
                #if epoch > 10:
                   # non-local means algorithm (NLM)
                #   loss_nlm = self.nlm_denoiser.train(input_noisy, clean, noise) 
                #   loss_total += loss_nlm
            
            if self.amp:        
                scaler.scale(loss_total).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_total.backward()
                self.optimizer.step()
            
            self.train_metrics.update("Total_loss", loss_total.item())
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} TotalLoss: {:.6f}' .format(
                    epoch,
                    self._progress(batch_idx),
                    loss_total.item()
                ))

            if batch_idx == self.len_epoch:
                break

            del target
            del loss_total
            # debug
            # break

        log = self.train_metrics.result()

        if self.do_test:
            if epoch % 10 == 0:
                test_log = self._test_epoch(epoch, save=False)
                log.update(**{'test_' + k: v for k, v in test_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.writer.close()

        return log

    def _test_epoch(self, epoch, save=False):

        self.test_metrics.reset()
        self.model.eval()

        if save == True:
            os.makedirs(self.output_dir + '/C/'+str(epoch), exist_ok=True)
            os.makedirs(self.output_dir + '/N/'+str(epoch), exist_ok=True)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(self.amp):
                for batch_idx, (target, input_noisy, input_GT) in enumerate(self.val_data_loader):
                    input_noisy = input_noisy.to(self.device)
                    input_GT = input_GT.to(self.device)
                    
                    input_noisy = padr(input_noisy)
                    input_GT = padr(input_GT)

                    clean = self.model(input_noisy)
                    #clean = self.model.denoise(input_noisy)
                    noise = input_noisy - clean
            
                    if save == True:
                        for i in range(input_noisy.shape[0]):
                            save_image(torch.clamp(clean[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().float().cpu(), self.output_dir + '/C/'+str(epoch)+'/'+target['dir_idx'][i]+'.PNG')
                            save_image(torch.clamp(input_GT[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().float().cpu(), self.output_dir + '/GT/' +target['dir_idx'][i]+'.PNG')
                            save_image(torch.clamp(noise[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().cpu(), self.output_dir + '/N/'+str(epoch)+'/'+target['dir_idx'][i]+'.PNG')
                            save_image(torch.clamp(input_noisy[i,:,pad:-pad,pad:-pad],min=0,max=1).detach().float().cpu(), self.output_dir + '/I/' +target['dir_idx'][i]+'.PNG')

                    self.writer.set_step(
                        (epoch - 1) * len(self.val_data_loader) + batch_idx, 'test')
                    for met in self.metric_ftns:
                        if met.__name__ == "psnr":
                            psnr = met(input_GT[:, :, pad:-pad, pad:-pad].to(self.device),
                                    torch.clamp(clean[:, :, pad:-pad, pad:-pad], min=0, max=1))
                            self.test_metrics.update('psnr', psnr)
                        elif met.__name__ == "ssim":
                            self.test_metrics.update('ssim', met(input_GT[:, :, pad:-pad, pad:-pad].to(
                                self.device), torch.clamp(clean[:, :, pad:-pad, pad:-pad], min=0, max=1)))
                    self.writer.close()

                    del target

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
