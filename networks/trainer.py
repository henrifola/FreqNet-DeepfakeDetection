import functools
import torch
import torch.nn as nn
from networks.freqnet import freqnet
from networks.model_loader import get_model_by_name
from networks.base_model import BaseModel, init_weights


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt, model_name='freqnet1'):
        super(Trainer, self).__init__(opt)

        # Change to dynamic model selection
        #self.model = freqnet()
        self.model_name = model_name
        self.model = get_model_by_name(model_name)

        
        for name,pa in self.model.named_parameters():
            if pa.requires_grad: print('='*20, 'requires_grad Ture',name)
        for name,pa in self.model.named_parameters():
            if not pa.requires_grad: print('='*20, 'requires_grad False',name)
        print()
        
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])
 

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.8
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.8} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

