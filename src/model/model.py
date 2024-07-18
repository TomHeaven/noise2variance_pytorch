if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from base import BaseModel
from model.DBSNl import DBSNl, CentralMaskedConv2d
from model.pd import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
       
class GenClean(nn.Module):
    def __init__(self, channels=3, num_of_layers=17, use_blind_spot=False):
        super(GenClean, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU())
        
        if use_blind_spot:
            stride = 2
            layers.append(CentralMaskedConv2d(features, features, kernel_size=2*stride-1, stride=1, padding=stride-1))
        else:
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU())
        
        for _ in range(num_of_layers-3):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.genclean = nn.Sequential(*layers)
        for m in self.genclean:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform_(m.weight)
               nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.genclean(x)
        return out
    
class GenNoise(nn.Module):
    def __init__(self, channels=3, NLayer=14, FSize=64):
        super(GenNoise, self).__init__()
        kernel_size = 3
        padding = 1
        m = [nn.Conv2d(channels, FSize, kernel_size=kernel_size, padding=padding),
             nn.ReLU(inplace=True)]             
        for i in range(NLayer-1):
            m.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding))
            m.append(nn.ReLU(inplace=True))
        m.append(nn.Conv2d(in_channels=FSize, out_channels=channels, kernel_size=1, padding=0))        
        self.body = nn.Sequential(*m)
        for m in self.body:
            if isinstance(m, nn.Conv2d):
               nn.init.xavier_uniform_(m.weight)
               nn.init.constant_(m.bias, 0)
       
    def forward(self, x):
        noise = self.body(x)
        m = torch.mean(torch.mean(noise,-1),-1).unsqueeze(-1).unsqueeze(-1)
        noise = noise-m     
        return noise
 
class Noise2Variance(BaseModel):
    def __init__(self):
        super().__init__()
        self.n_colors = 3
        self.genclean = GenClean(self.n_colors, use_blind_spot=False)
        self.gennoise = GenNoise(self.n_colors)
        self.pd_a = 5
        self.pd_b = 2
        self.pd_pad = 0
        self.R3 = True
        self.R3_T = 8
        self.R3_p = 0.16
        self.enable_pd = False

    def forward(self, x):
        if self.training:
            self.pd = self.pd_a
        else:
            self.pd = self.pd_b
            
        if self.enable_pd:
            x = pixel_shuffle_down_sampling(x, self.pd, self.pd_pad)              
        clean = self.genclean(x)
        noise = self.gennoise(x) 
        if self.enable_pd:   
            clean = pixel_shuffle_up_sampling(clean, self.pd, self.pd_pad) 
            
        if self.training:
            if self.enable_pd:
                 noise = pixel_shuffle_up_sampling(noise, self.pd, self.pd_pad)       
            return clean, noise
        else:
            return clean    
    
    def denoise(self, x):
        '''
        Denoising process for inference.
        '''

        # forward PD-BSN process with inference pd factor
        clean = self.forward(x).float()

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3 or PD refinement) '''
            return clean
        elif self.R3: # R3 Refinement
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(clean).detach()
                tmp_input[mask] = x[mask]
                p = self.pd_pad
                tmp_input = torch.nn.functional.pad(tmp_input, (p,p,p,p), mode='reflect')
                if self.pd_pad == 0:
                    denoised[..., t] = self.forward(tmp_input)
                else:
                    denoised[..., t] = self.forward(tmp_input)[:,:,p:-p,p:-p]
            return torch.mean(denoised, dim=-1)
        else: # PD Refinement
            denoised = torch.empty(*(x.shape), 4, device=x.device)
            for i in range(2):
                for j in range(2):
                    mask = torch.zeros_like(x, dtype=torch.bool)
                    mask[:,:,i::2, j::2] = 1
                    tmp_input = torch.clone(clean).detach()
                    tmp_input[mask] = x[mask]
                    p = self.pd_pad
                    tmp_input = torch.nn.functional.pad(tmp_input, (p,p,p,p), mode='reflect')
                    t = i*2+j
                    if self.pd_pad == 0:
                        denoised[..., t] = self.forward(tmp_input)
                    else:
                        denoised[..., t] = self.forward(tmp_input)[:,:,p:-p,p:-p]
                    denoised[..., t] = self.forward(tmp_input)
            return torch.mean(denoised, dim=-1)


        


