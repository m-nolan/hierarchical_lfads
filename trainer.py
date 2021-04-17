import torch
import torch.nn.functional as F
from torch.nn import DataParallel
import torchvision
import time
import os
import pdb
import functools, collections, operator


class RunManager():
    def __init__(self, model, objective, optimizer, scheduler,
                 train_dl, valid_dl, transforms,
                 plotter=None, writer=None, do_health_check=False, detect_local_minima = False,
                 max_epochs=1000, save_loc = '/tmp/', load_checkpoint=False):
    
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu';
        self.model      = model
        self.objective  = objective
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.train_dl   = train_dl
        self.valid_dl   = valid_dl
        self.transforms = transforms
        self.writer     = writer
        self.plotter    = plotter
            
        self.max_epochs      = max_epochs
        self.do_health_check = do_health_check
        self.detect_local_minima = detect_local_minima
        self.save_loc        = save_loc
        
        self.epoch = 0
        self.step  = 0
        self.best  = float('inf')
        
        self.loss_dict = {'train' : {},
                          'valid' : {},
                          'l2'    : []}
        
        if load_checkpoint:
            self.load_checkpoint('recent')
            
    def run(self):  
        for epoch in range(self.epoch, self.max_epochs):
            if self.optimizer.param_groups[0]['lr'] < self.scheduler.min_lrs[0]:
                break
            self.epoch = epoch + 1
            tic = time.time()
            loss_dict_list = []
            
            self.model.train()
#             print(len(self.train_dl))
            for i,x in enumerate(self.train_dl):
                tr_tic = time.time()
#                 print(x[0].session)
                if isinstance(x, list) or isinstance(x,tuple):
                    x = x[0]
                self.optimizer.zero_grad()
                fw_tic = time.time()
                recon, latent = self.model(x)
#                 print('fw time: ', time.time()-fw_tic)
                loss_tic = time.time()
                loss, loss_dict = self.objective(x_orig= x,
                                                 x_recon= recon,
                                                 model= self.model)
#                 print('loss time: ', time.time()-loss_tic)
                loss_dict_list.append(loss_dict)

                bw_tic = time.time()
                loss.backward()
#                 print('bw time: ', time.time()-bw_tic)
                
#                 if torch.isnan(loss.data):
#                     break

                # Clip gradient norm
                if isinstance(self.model,DataParallel):
                    max_norm = self.model.module.max_norm
                else:
                    max_norm = self.model.max_norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                # update the weights
                self.optimizer.step()

                
                self.objective.weight_schedule_fn(self.step)
                if isinstance(self.model,DataParallel):
                    do_normalize_factors = self.model.module.do_normalize_factors
                else:
                    do_normalize_factors = self.model.do_normalize_factors
                if do_normalize_factors:
                    if isinstance(self.model,DataParallel):
                        self.model.module.normalize_factors()
                    else:
                        self.model.normalize_factors()

                    # Row-normalise fc_factors (See bullet-point 11 of section 1.9 of online methods)
                    
                self.optimizer, self.scheduler = self.model.change_parameter_grad_status(self.step, self.optimizer, self.scheduler)
                
                self.step += 1
            #if torch.isnan(loss.data):
            #    print('Loss is NaN')
            #    break
            
            
            train_data = x.clone()
            loss_dict = {} 
            
            for d in loss_dict_list: 
                for k in d.keys(): 
                    loss_dict[k] = loss_dict.get(k, 0) + d[k]/len(loss_dict_list)
            for key, val in loss_dict.items():
                if key in self.loss_dict['train'].keys():
                    self.loss_dict['train'][key].append(loss_dict[key])
                elif key == 'l2':
                    self.loss_dict[key].append(loss_dict[key])
                else:
                    self.loss_dict['train'][key] = [loss_dict[key]]
            
            

            self.scheduler.step(self.loss_dict['train']['total'][-1])
            loss_dict_list = []
            self.model.eval()
#             print(len(self.valid_dl))
            for i, x in enumerate(self.valid_dl):
                with torch.no_grad():
                    if isinstance(x, list) or isinstance(x,tuple):
                        x = x[0]
                    fw_val_tic = time.time()
                    recon, latent = self.model(x)
#                     print('fw val time: ',time.time()-fw_val_tic)
                    loss, loss_dict = self.objective(x_orig= x, x_recon= recon, model= self.model)
                    loss_dict_list.append(loss_dict)
                    
            valid_data = x.clone()
            loss_dict = {} 
            for d in loss_dict_list: 
                for k in d.keys(): 
                    loss_dict[k] = loss_dict.get(k, 0) + d[k]/len(loss_dict_list)

            for key, val in loss_dict.items():
                if key in self.loss_dict['valid'].keys():
                    self.loss_dict['valid'][key].append(loss_dict[key])
                elif key == 'l2':
                    pass
                else:
                    self.loss_dict['valid'][key] = [loss_dict[key]]
                    
            valid_loss = self.loss_dict['valid']['total'][-1]
            if not self.objective.any_zero_weights():
                if valid_loss < self.best:
                    self.best = 0
                    for key,val in self.loss_dict['valid'].items():
                        if 'recon' in key:
                            self.best += val[-1]
                        if ('kl' in key):
                            full_val = val[-1] / self.objective.loss_weights[key]['weight']
                            self.best += full_val
                    self.save_checkpoint('best')
                
            self.save_checkpoint()
            if self.writer is not None:
                self.write_to_tensorboard()
                if self.plotter is not None:
                    if self.epoch % 25 == 0:
                        self.plot_to_tensorboard()
                        
                if self.do_health_check:
                    self.health_check(self.model)
                    
            toc = time.time()
#             print('backward time: ',bw_toc - bw_tic,' forward time: ',fw_toc - fw_tic, ' optim time: ',opt_toc - opt_tic, ' forward val time: ',fw_val_toc-fw_val_tic)
            
            results_string = 'Epoch %5d, Epoch time = %.3f s, Loss (train, valid): '%(self.epoch, toc - tic)
            for key in self.loss_dict['train'].keys():
                train_loss = self.loss_dict['train'][key][self.epoch-1]
                valid_loss = self.loss_dict['valid'][key][self.epoch-1]
                results_string+= ' %s (%.3f, %.3f),'%(key, train_loss, valid_loss)
            
            results_string+= ' %s (%.3f)'%('l2', self.loss_dict['l2'][self.epoch-1])
            
            print(results_string, flush=True)
            
            # Check if local minima with 0 KL or L2 loss reached
            if self.detect_local_minima:
                in_local_minima = False
                if not self.objective.any_zero_weights():
                    for key,val in self.loss_dict['valid'].items():
                        if ('kl' in key or 'l2' in key):
                            if torch._np.abs(val[-1] / self.objective.loss_weights[key]['weight']) < 0.1:
                                in_local_minima = True
                        else:
                            if torch._np.abs(val[-1]) < 0.1:
                                in_local_minima = True

                if in_local_minima:
                    print('Stuck in local minima')
                    break
            
    def write_to_tensorboard(self):
        
        # Write loss to tensorboard
        
        for ix, key in enumerate(self.loss_dict['train'].keys()):
            train_loss = self.loss_dict['train'][key][self.epoch-1]
            valid_loss = self.loss_dict['valid'][key][self.epoch-1]
            
            self.writer.add_scalars('1_Loss/%i_%s'%(ix+1, key), {'Training' : float(train_loss),
                                                                 'Validation' : float(valid_loss)}, self.epoch)
        l2_loss = self.loss_dict['l2'][self.epoch-1]
        self.writer.add_scalar('1_Loss/4_L2_loss', float(l2_loss), self.epoch)

        for jx, grp in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar('2_Optimizer/1.%i_Learning_Rate_PG'%(jx+1), grp['lr'], self.epoch)
        
        for kx, key in enumerate(self.objective.loss_weights.keys()):
            weight = self.objective.loss_weights[key]['weight']
            self.writer.add_scalar('2_Optimizer/2.%i_%s_weight'%(kx+1, key), weight, self.epoch)
        
    def plot_to_tensorboard(self):
        figs_dict_train = self.plotter['train'].plot_summary(model= self.model, dl= self.train_dl)
        
        figs_dict_valid = self.plotter['valid'].plot_summary(model= self.model, dl= self.valid_dl)
        
        fig_names = ['traces', 'inputs', 'factors', 'rates', 'spikes']
        for fn in fig_names:
            if fn in figs_dict_train.keys():
                self.writer.add_figure('%s/train'%(fn), figs_dict_train[fn], self.epoch, close=True)
            elif 'truth_%s'%fn in figs_dict_train.keys():
                self.writer.add_figure('%s/train'%(fn), figs_dict_train['truth_%s'%fn], self.epoch, close=True)

            if fn in figs_dict_valid.keys():
                self.writer.add_figure('%s/valid'%(fn), figs_dict_valid[fn], self.epoch, close=True)
            elif 'truth_%s'%fn in figs_dict_valid.keys():
                self.writer.add_figure('%s/valid'%(fn), figs_dict_valid['truth_%s'%fn], self.epoch, close=True)
            
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    def health_check(self, model):
        '''
        Gets gradient norms for each parameter and writes to tensorboard
        '''
        
        for ix, (name, param) in enumerate(model.named_parameters()):
            if param.grad is not None:
                self.writer.add_scalar('3_Gradient_norms/%i_%s'%(ix, name), param.grad.data.norm(), self.epoch)
            else:
                self.writer.add_scalar('3_Gradient_norms/%i_%s'%(ix, name), 0.0, self.epoch)
                
            if 'weight' in name:
                self.writer.add_scalar('4_Weight_norms/%i_%s'%(ix, name), param.data.norm(), self.epoch)
        
    def save_checkpoint(self, output_filename='recent'):
                # Create dictionary of training variables
        train_dict = {'best' : self.best,
                      'loss_dict': self.loss_dict,
                      'loss_weights' : self.objective.loss_weights,
                      'epoch' : self.epoch, 'step' : self.step}
        
        # Save network parameters, optimizer state, and training variables
        if not os.path.isdir(self.save_loc+'checkpoints/'):
            os.mkdir(self.save_loc+'checkpoints/')
        if isinstance(self.model,torch.nn.DataParallel):
            net_state_dict = self.model.module.state_dict()
        else:
            net_state_dict = self.model.state_dict()
        torch.save({'net' : net_state_dict , 'opt' : self.optimizer.state_dict(),
                    'sched': self.scheduler.state_dict(), 'run_manager' : train_dict},
                     self.save_loc+'checkpoints/' + output_filename + '.pth')
        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
        
    def load_checkpoint(self, input_filename='recent'):
        if os.path.exists(self.save_loc + 'checkpoints/' + input_filename + '.pth'):
            state_dict = torch.load(self.save_loc + 'checkpoints/' + input_filename + '.pth')
            
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(state_dict['net'],strict=False)
            else:
                self.model.load_state_dict(state_dict['net'],strict=False)
            print(state_dict['run_manager']['step'])
            if len(state_dict['opt']['param_groups']) > 1:
                self.optimizer, self.scheduler = self.model.change_parameter_grad_status(state_dict['run_manager']['step'], self.optimizer, self.scheduler, loading_checkpoint=True)
#                 pdb.set_trace()
            self.optimizer.load_state_dict(state_dict['opt'])
            self.scheduler.load_state_dict(state_dict['sched'])

            self.best = state_dict['run_manager']['best']
            self.loss_dict = state_dict['run_manager']['loss_dict']
            self.objective.loss_weights = state_dict['run_manager']['loss_weights']
            self.epoch = state_dict['run_manager']['epoch']
            self.step  = state_dict['run_manager']['step']
            
