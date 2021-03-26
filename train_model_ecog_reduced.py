#!/usr/bin/env python

import argparse
import os
from shutil import rmtree # dangerous...

import torch
from torch.nn import DataParallel
import torchvision
import torch.optim as opt
import torchvision.transforms as trf
from torch.utils.data.dataset import Dataset
import pickle

from orion.client import report_results

from trainer import RunManager
from scheduler import LFADS_Scheduler

from utils import read_data, load_parameters, save_parameters
from plotter import Plotter

from math import floor

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-p', '--hyperparameter_path', type=str)
parser.add_argument('-o', '--output_dir', default='/tmp', type=str)
parser.add_argument('--max_epochs', default=2000, type=int)
parser.add_argument('--batch_size', default=None, type=int)
parser.add_argument('--data_suffix', default='data', type=str)
parser.add_argument('--detect_local_minima', action='store_true', default=False)

parser.add_argument('-t', '--use_tensorboard', action='store_true', default=False)
parser.add_argument('-r', '--restart', action='store_true', default=False)
parser.add_argument('-c', '--do_health_check', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--log10_lr', type=float, default=None)
parser.add_argument('--kl_deep_max', type=float, default=None)
parser.add_argument('--kl_obs_max', type=float, default=None)
parser.add_argument('--kl_obs_dur', type=int, default=None)
parser.add_argument('--kl_obs_dur_scale', type=int, default=1.0)
parser.add_argument('--deep_start_p', type=int, default=None)
parser.add_argument('--deep_start_p_scale', type=float, default=1.0)
parser.add_argument('--l2_gen_scale', type=float, default=None)
parser.add_argument('--l2_con_scale', type=float, default=None)
parser.add_argument('--log10_l2_gen_scale', type=float, default=None)
parser.add_argument('--log10_l2_con_scale', type=float, default=None)
parser.add_argument('--drop_ratio', type=float, default=0.0)

parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--ch_idx', nargs='+', type=int, default=None)
parser.add_argument('--device_num', type=int, default=None)
parser.add_argument('--multidevice', action='store_true', default=False)
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--use_fdl', action='store_true', default=False)
parser.add_argument('--predict', action='store_true', default=False)
parser.add_argument('--attention', action='store_true', default=False)

def main():
    args = parser.parse_args()
    
    if args.device_num is None:
        device_str = ''
    else:
        # add a check on this int. Less than total # of units, reallocate if busy, etc
        device_str = f':{args.device_num}'
    device = f'cuda{device_str}' if torch.cuda.is_available() else 'cpu'
    print(f'using device:\t{device}')

    hyperparams = load_parameters(args.hyperparameter_path)
    
    orion_hp_string, hyperparams = prep_orion(args, hyperparams)

    save_loc, hyperparams = generate_save_loc(args, hyperparams, orion_hp_string)
    save_loc = save_loc[:-1] + 'varstd'
    if args.attention:
        save_loc = save_loc + '-attn'
    if args.drop_ratio > 0:
        save_loc = save_loc + f'-droprat{args.drop_ratio}'
    if args.use_fdl:
        save_loc = save_loc + '-fdl'
    save_loc = save_loc + os.sep
    
    save_parameters(save_loc, hyperparams)
    
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        
    data_dict   = read_data(args.data_path)

    mse = args.loss == 'mse'
    
    if args.drop_ratio > 0:
        transforms = trf.Compose([DropChannels(drop_ratio=args.drop_ratio)])
    else:
        transforms = None

    train_dl, valid_dl, plotter, model, objective = prep_model(model_name  = args.model,
                                                               data_dict   = data_dict,
                                                               data_suffix = args.data_suffix,
                                                               batch_size  = args.batch_size,
                                                               seq_len = args.seq_len,
                                                               ch_idx = args.ch_idx,
                                                               device = device,
                                                               hyperparams = hyperparams,
                                                               multidevice = args.multidevice,
                                                               mse = mse,
                                                               attention = args.attention,
                                                               transform = transforms,
                                                               use_fdl = args.use_fdl)
        
    print_model_description(model)
    
    optimizer, scheduler = prep_optimizer(model, hyperparams)
        
    if args.use_tensorboard:
        writer, rm_plotter = prep_tensorboard(save_loc, plotter, args.restart)
    else:
        writer = None
        rm_plotter = None
    
    # the plotter is broken. ignore for now, plot results later.
    rm_plotter = None
    
    run_manager = RunManager(model      = model,
                             objective  = objective,
                             optimizer  = optimizer,
                             scheduler  = scheduler,
                             train_dl   = train_dl,
                             valid_dl   = valid_dl,
                             transforms = transforms,
                             writer     = writer,
                             plotter    = rm_plotter,
                             max_epochs = args.max_epochs,
                             save_loc   = save_loc,
                             do_health_check = args.do_health_check,
                             detect_local_minima = args.detect_local_minima,
                             load_checkpoint=(not args.restart))
    run_manager.run()
        
    save_figs(save_loc, run_manager.model, run_manager.valid_dl, plotter)
    pickle.dump(run_manager.loss_dict, open(save_loc+'/loss.pkl', 'wb'))

#-------------------------------------------------------------------
#-------------------------------------------------------------------

class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_model(model_name, data_dict, data_suffix, batch_size, device, hyperparams, seq_len=None, ch_idx=None, multidevice=False, mse=True, attention=False, transform=None, use_fdl=False):
    if model_name == 'lfads':
        train_dl, valid_dl, input_dims, plotter = prep_data(data_dict=data_dict, data_suffix=data_suffix, batch_size=batch_size, seq_len=seq_len, device=device, ch_idx=ch_idx)
        model, objective = prep_lfads(input_dims = input_dims,
                                      hyperparams=hyperparams,
                                      device= device,
                                      dtype=train_dl.dataset.tensors[0].dtype,
                                      dt= data_dict['dt']
                                      )

    if model_name == 'lfads_ecog':
        train_dl, valid_dl, input_dims, plotter = prep_data(data_dict=data_dict, data_suffix=data_suffix, batch_size=batch_size, device=device, seq_len=seq_len, ch_idx=ch_idx, transform=transform)
        model, objective = prep_lfads_ecog(input_dims = input_dims,
                                      hyperparams=hyperparams,
                                      device= device,
                                      dtype=train_dl.dataset.tensors[0].dtype,
                                      dt= data_dict['dt'],
                                      multidevice=multidevice,
                                      mse=mse,
                                      attention=attention,
                                      use_fdl=use_fdl)
        
    elif model_name == 'svlae':
        train_dl, valid_dl, input_dims, plotter = prep_data(data_dict=data_dict, data_suffix=data_suffix, batch_size=batch_size, device=device)
        
        if 'obs_gain_init' in data_dict.keys():
            print('gain= %.4f'%data_dict['obs_gain_init'].mean())
            hyperparams['model']['obs']['gain']['value'] = data_dict['obs_gain_init']
        if 'obs_bias_init' in data_dict.keys():
            print('bias= %.4f'%data_dict['obs_bias_init'].mean())
            hyperparams['model']['obs']['bias']['value'] = data_dict['obs_bias_init']
        if 'obs_var_init' in data_dict.keys():
            print('var= %.4f'%data_dict['obs_var_init'].mean())
            hyperparams['model']['obs']['var']['value'] = data_dict['obs_var_init']
        if 'obs_tau_init' in data_dict.keys():
            print('tau= %.4f'%data_dict['obs_tau_init'].mean())
            hyperparams['model']['obs']['tau']['value'] = data_dict['obs_tau_init']
        
        model, objective = prep_svlae(input_dims = input_dims,
                                      hyperparams=hyperparams,
                                      device= device,
                                      dtype=train_dl.dataset.tensors[0].dtype,
                                      dt=data_dict['dt']
                                      )
        
    elif model_name == 'conv3d_lfads':
        train_dl, valid_dl, input_dims, plotter = prep_video(data_dict=data_dict, batch_size=batch_size, device=device)
        model, objective = prep_conv3d_lfads(input_dims = input_dims,
                                             hyperparams=hyperparams,
                                             device= device,
                                             dtype=train_dl.dataset.dtype
                                             )
    else:
        raise NotImplementedError('Model must be one of \'lfads\', \'conv3d_lfads\', or \'svlae\'')
        
    return train_dl, valid_dl, plotter, model, objective
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
        
def prep_lfads(input_dims, hyperparams, device, dtype, dt):
    from objective import LFADS_Loss, LogLikelihoodPoisson
    from lfads import LFADS_SingleSession_Net

    model = LFADS_SingleSession_Net(input_size           = input_dims,
                                    factor_size          = hyperparams['model']['factor_size'],
                                    g_encoder_size       = hyperparams['model']['g_encoder_size'],
                                    c_encoder_size       = hyperparams['model']['c_encoder_size'],
                                    g_latent_size        = hyperparams['model']['g_latent_size'],
                                    u_latent_size        = hyperparams['model']['u_latent_size'],
                                    controller_size      = hyperparams['model']['controller_size'],
                                    generator_size       = hyperparams['model']['generator_size'],
                                    prior                = hyperparams['model']['prior'],
                                    clip_val             = hyperparams['model']['clip_val'],
                                    dropout              = hyperparams['model']['dropout'],
                                    do_normalize_factors = hyperparams['model']['normalize_factors'],
                                    max_norm             = hyperparams['model']['max_norm'],
                                    device               = device).to(device)
    
    loglikelihood = LogLikelihoodPoisson(dt=float(dt))

    objective = LFADS_Loss(loglikelihood            = loglikelihood,
                           loss_weight_dict         = {'kl': hyperparams['objective']['kl'], 
                                                       'l2': hyperparams['objective']['l2']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)

    return model, objective

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_lfads_ecog(input_dims, hyperparams, device, dtype, dt, multidevice, mse=True, attention=False, use_fdl=True):
    from objective import LFADS_Loss, LogLikelihoodGaussian
    from lfads import LFADS_Ecog_SingleSession_Net

    model = LFADS_Ecog_SingleSession_Net(input_size           = input_dims,
                                    factor_size          = hyperparams['model']['factor_size'],
                                    g_encoder_size       = hyperparams['model']['g_encoder_size'],
                                    c_encoder_size       = hyperparams['model']['c_encoder_size'],
                                    g_latent_size        = hyperparams['model']['g_latent_size'],
                                    u_latent_size        = hyperparams['model']['u_latent_size'],
                                    controller_size      = hyperparams['model']['controller_size'],
                                    generator_size       = hyperparams['model']['generator_size'],
                                    attention            = attention,
                                    prior                = hyperparams['model']['prior'],
                                    clip_val             = hyperparams['model']['clip_val'],
                                    dropout              = hyperparams['model']['dropout'],
                                    do_normalize_factors = hyperparams['model']['normalize_factors'],
                                    max_norm             = hyperparams['model']['max_norm'],
                                    device               = device)
    
    if multidevice and torch.cuda.device_count() > 1:
        model = DataParallelPassthrough(model)

    model.to(device)
    
    loglikelihood = LogLikelihoodGaussian(mse=mse)

    objective = LFADS_Loss(loglikelihood            = loglikelihood,
                           use_fdl                  = use_fdl,
                           loss_weight_dict         = {'kl': hyperparams['objective']['kl'], 
                                                       'l2': hyperparams['objective']['l2']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)

    return model, objective
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_conv3d_lfads(input_dims, hyperparams, device, dtype):
    
    from synthetic_data import SyntheticCalciumVideoDataset
    from objective import Conv_LFADS_Loss, LogLikelihoodGaussian
    from conv_lfads import Conv3d_LFADS_Net
    
    model = Conv3d_LFADS_Net(input_dims      = (num_steps, width, height),
                             channel_dims    = hyperparams['model']['channel_dims'],
                             factor_size     = hyperparams['model']['factor_size'],
                             g_encoder_size  = hyperparams['model']['g_encoder_size'],
                             c_encoder_size  = hyperparams['model']['c_encoder_size'],
                             g_latent_size   = hyperparams['model']['g_latent_size'],
                             u_latent_size   = hyperparams['model']['u_latent_size'],
                             controller_size = hyperparams['model']['controller_size'],
                             generator_size  = hyperparams['model']['generator_size'],
                             prior           = hyperparams['model']['prior'],
                             clip_val        = hyperparams['model']['clip_val'],
                             conv_dropout    = hyperparams['model']['conv_dropout'],
                             lfads_dropout   = hyperparams['model']['lfads_dropout'],
                             do_normalize_factors = hyperparams['model']['normalize_factors'],
                             max_norm        = hyperparams['model']['max_norm'],
                             device          = device).to(device)
    
    model.to(dtype=dtype)
    torch.set_default_dtype(dtype)
    
    loglikelihood = LogLikelihoodGaussian()
    objective = Conv_LFADS_Loss(loglikelihood=loglikelihood,
                                loss_weight_dict={'kl': hyperparams['objective']['kl'],
                                                  'l2': hyperparams['objective']['l2']},
                                                   l2_con_scale= hyperparams['objective']['l2_con_scale'],
                                                   l2_gen_scale= hyperparams['objective']['l2_gen_scale']).to(device)
    
    
    return model, objective

#-------------------------------------------------------------------
#-------------------------------------------------------------------

# using a conv input, output block to model ECoG, LFP data

def prep_conv1d_lfads_ecog(input_dims, hyperparams, device, dtype):
    
    from objective import Conv_LFADS_Ecog_Loss
    from conv_lfads import Conv1d_LFADS_Net, LogLikelihoodGaussian
    
    model = Conv1d_LFADS_Net(input_dims      = (num_steps, width, height),
                             channel_dims    = hyperparams['model']['channel_dims'],
                             factor_size     = hyperparams['model']['factor_size'],
                             g_encoder_size  = hyperparams['model']['g_encoder_size'],
                             c_encoder_size  = hyperparams['model']['c_encoder_size'],
                             g_latent_size   = hyperparams['model']['g_latent_size'],
                             u_latent_size   = hyperparams['model']['u_latent_size'],
                             controller_size = hyperparams['model']['controller_size'],
                             generator_size  = hyperparams['model']['generator_size'],
                             prior           = hyperparams['model']['prior'],
                             clip_val        = hyperparams['model']['clip_val'],
                             conv_dropout    = hyperparams['model']['conv_dropout'],
                             lfads_dropout   = hyperparams['model']['lfads_dropout'],
                             do_normalize_factors = hyperparams['model']['normalize_factors'],
                             max_norm        = hyperparams['model']['max_norm'],
                             device          = device).to(device)
    model.to(dtype=dtype)
    torch.set_default_dtype(dtype)
    
    loglikelihood = LogLikelihoodGaussian()
    objective = Conv_LFADS_Loss(loglikelihood=loglikelihood,
                                use_fdl=use_fdl,
                                loss_weight_dict={'kl': kyperparams['objective']['kl'],
                                                  'l2': hyperparams['objective']['l2']},
                                l2_con_scale=hyperparams['objective']['l2_con_scale'],
                                l2_gen_scale=hyperparams['objective']['l2_gen_scale']).to(device)
    
    return model, objective

#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_svlae(input_dims, hyperparams, device, dtype, dt):
    from svlae import SVLAE_Net
    from objective import LogLikelihoodGaussian, LogLikelihoodPoissonSimplePlusL1, SVLAE_Loss

    loglikelihood_obs  = LogLikelihoodGaussian()
    loglikelihood_deep = LogLikelihoodPoissonSimplePlusL1(dt=float(dt))
    
    objective = SVLAE_Loss(loglikelihood_obs        = loglikelihood_obs,
                           loglikelihood_deep       = loglikelihood_deep,
                           loss_weight_dict         = {'kl_deep'    : hyperparams['objective']['kl_deep'],
                                                       'kl_obs'     : hyperparams['objective']['kl_obs'],
                                                       'l2'         : hyperparams['objective']['l2'],
                                                       'recon_deep' : hyperparams['objective']['recon_deep']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)
    
    hyperparams['model']['obs']['tau']['value']/=float(dt)
    
    model = SVLAE_Net(input_size            = input_dims,
                      factor_size           = hyperparams['model']['factor_size'],
                      obs_encoder_size      = hyperparams['model']['obs_encoder_size'],
                      obs_latent_size       = hyperparams['model']['obs_latent_size'],
                      obs_controller_size   = hyperparams['model']['obs_controller_size'],
                      deep_g_encoder_size   = hyperparams['model']['deep_g_encoder_size'],
                      deep_c_encoder_size   = hyperparams['model']['deep_c_encoder_size'],
                      deep_g_latent_size    = hyperparams['model']['deep_g_latent_size'],
                      deep_u_latent_size    = hyperparams['model']['deep_u_latent_size'],
                      deep_controller_size  = hyperparams['model']['deep_controller_size'],
                      generator_size        = hyperparams['model']['generator_size'],
                      prior                 = hyperparams['model']['prior'],
                      clip_val              = hyperparams['model']['clip_val'],
                      generator_burn        = hyperparams['model']['generator_burn'],
                      dropout               = hyperparams['model']['dropout'],
                      do_normalize_factors  = hyperparams['model']['normalize_factors'],
                      factor_bias           = hyperparams['model']['factor_bias'],
                      max_norm              = hyperparams['model']['max_norm'],
                      deep_unfreeze_step    = hyperparams['model']['deep_unfreeze_step'],
                      obs_early_stop_step   = hyperparams['model']['obs_early_stop_step'],
                      obs_continue_step     = hyperparams['model']['obs_continue_step'],
                      ar1_start_step        = hyperparams['model']['ar1_start_step'],
                      obs_params            = hyperparams['model']['obs'],
                      device                = device).to(device)
    
    return model, objective
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# move this later
class EcogTensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, device='cpu', transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.device = device
        self.transform = transform

    def __getitem__(self, index):
        sample = tuple(tensor[index].to(self.device) for tensor in self.tensors)
        if self.transform:
            sample = tuple(self.transform(s) for s in sample)
        return sample

    def __len__(self):
        return self.tensors[0].size(0)

#-------------------------------------------------------------------
#-------------------------------------------------------------------
# data dropout transforms
class DropChannels(object):
    '''
        Dataset transform to randomly drop channels (i.e. set all values to zero) within a sample.
        The number of dropped channels is determined by the drop ratio:
            n_drop = floor(drop_ratio*n_ch)
        Channel dimension is assumed to be the last indexed tensor dimension. This may need to be
        adjusted for multidimensional time series data, e.g. spectrograms.
    '''
    def __init__(self,drop_ratio=0.1):
        self.drop_ratio = drop_ratio

    def __call__(self,sample):
        n_ch = sample.shape[-1]
        n_ch_drop = floor(self.drop_ratio*n_ch)
        drop_ch_idx = torch.randperm(n_ch)[:n_ch_drop]
        sample[:,drop_ch_idx] = 0.
        return sample
        

#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_data(data_dict, data_suffix, batch_size, device, seq_len=None, ch_idx=None, transform=None):
    if seq_len is None:
        seq_len = data_dict[f'train_{data_suffix}'].shape[1]
    if ch_idx is None:
        n_ch = data_dict[f'train_{data_suffix}'].shape[2]
        ch_idx = torch.arange(n_ch)
    train_data  = torch.Tensor(data_dict['train_%s'%data_suffix])[:,:seq_len,ch_idx]
    valid_data  = torch.Tensor(data_dict['valid_%s'%data_suffix])[:,:seq_len,ch_idx]
    
    num_trials, num_steps, input_size = train_data.shape
    
    train_ds    = EcogTensorDataset(train_data,device=device,transform=transform)
    valid_ds    = EcogTensorDataset(valid_data,device=device,transform=transform)
    
    train_dl    = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    valid_dl    = torch.utils.data.DataLoader(valid_ds, batch_size = batch_size)
    
    TIME = torch.arange(num_steps)*data_dict['dt']
    
    train_truth = {}
    if 'train_rates' in data_dict.keys():
        train_truth['rates'] = data_dict['train_rates']
    if 'train_latent' in data_dict.keys():
        train_truth['latent'] = data_dict['train_latent']
    if 'valid_spikes' in data_dict.keys():
        train_truth['spikes'] = data_dict['train_spikes']
        
    valid_truth = {}
    if 'valid_rates' in data_dict.keys():
        valid_truth['rates'] = data_dict['valid_rates']
    if 'valid_latent' in data_dict.keys():
        valid_truth['latent'] = data_dict['valid_latent']
    if 'valid_spikes' in data_dict.keys():
        valid_truth['spikes'] = data_dict['valid_spikes']

    plotter = {'train' : Plotter(time=TIME, truth=train_truth),
               'valid' : Plotter(time=TIME, truth=valid_truth)}
    
    return train_dl, valid_dl, input_size, plotter
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_video(data_dict, batch_size, device):
    train_dl    = torch.utils.data.DataLoader(SyntheticCalciumVideoDataset(traces= data_dict['train_fluor'], cells=data_dict['cells'], device=device), batch_size=args.batch_size)
    valid_dl    = torch.utils.data.DataLoader(SyntheticCalciumVideoDataset(traces= data_dict['valid_fluor'], cells=data_dict['cells'], device=device), batch_size=args.batch_size)
    
    num_trials, num_steps, num_cells = data_dict['train_fluor'].shape
    num_cells, width, height = data_dict['cells'].shape
    
    input_dims = (num_steps, width, height)
    
    TIME = torch._np.arange(0, num_steps*data_dict['dt'], data_dict['dt'])
    
    train_truth = {}
    if 'train_latent' in data_dict.keys():
        train_truth['latent'] = data_dict['train_latent']
        
    valid_truth = {}
    if 'valid_latent' in data_dict.keys():
        valid_truth['latent'] = data_dict['valid_latent']

    plotter = {'train' : Plotter(time=TIME, truth=train_truth),
               'valid' : Plotter(time=TIME, truth=valid_truth)}
    
    return train_dl, valid_dl, input_size, plotter

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_optimizer(model, hyperparams):
    
    if isinstance(model,DataParallel):
        model=model.module
    optimizer = opt.Adam([p for p in model.parameters() if p.requires_grad],
                         lr=hyperparams['optimizer']['lr_init'],
                         betas=hyperparams['optimizer']['betas'],
                         eps=hyperparams['optimizer']['eps'])
    
    scheduler = LFADS_Scheduler(optimizer      = optimizer,
                                mode           = 'min',
                                factor         = hyperparams['scheduler']['scheduler_factor'],
                                patience       = hyperparams['scheduler']['scheduler_patience'],
                                verbose        = True,
                                threshold      = 1e-4,
                                threshold_mode = 'abs',
                                cooldown       = hyperparams['scheduler']['scheduler_cooldown'],
                                min_lr         = hyperparams['scheduler']['lr_min'])
    
    return optimizer, scheduler

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def print_model_description(model):
    total_params = 0
    for ix, (name, param) in enumerate(model.named_parameters()):
        print(ix, name, list(param.shape), param.numel(), param.requires_grad)
        total_params += param.numel()
    
    print('Total parameters: %i'%total_params)

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_tensorboard(save_loc, plotter, restart):
    import importlib
    if importlib.util.find_spec('torch.utils.tensorboard'):
        tb_folder = save_loc + 'tensorboard/'
        if not os.path.exists(tb_folder):
            os.mkdir(tb_folder)
        elif os.path.exists(tb_folder) and restart:
            os.system('rm -rf %s'%tb_folder)
            os.mkdir(tb_folder)

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tb_folder)
        rm_plotter = plotter
    else:
        writer = None
        rm_plotter = None
            
    return writer, rm_plotter

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_orion(args, hyperparams):

    orion_hp_string = ''
    if args.lr or args.log10_lr:
        if args.log10_lr:
            lr = 10**args.log10_lr
        else:
            lr = args.lr
        hyperparams['optimizer']['lr_init'] = lr
        hyperparams['scheduler']['lr_min']  = lr * 1e-3
        orion_hp_string += 'lr= %.4f\n'%lr
        
    if args.kl_obs_dur:
        hyperparams['objective']['kl_obs']['schedule_dur'] = args.kl_obs_dur * args.kl_obs_dur_scale
        orion_hp_string += 'kl_obs_dur= %i\n'%(args.kl_obs_dur*args.kl_obs_dur_scale)

    if args.kl_obs_max:
        hyperparams['objective']['kl_obs']['max'] = args.kl_obs_max
        orion_hp_string += 'kl_obs_max= %.3f\n'%(args.kl_obs_max)
        
    if args.kl_deep_max:
        hyperparams['objective']['kl_deep']['max'] = args.kl_deep_max
        orion_hp_string += 'kl_deep_max= %.3f\n'%(args.kl_deep_max)
    
    if args.deep_start_p:
        deep_start = int(args.deep_start_p * args.deep_start_p_scale * hyperparams['objective']['kl_obs']['schedule_dur'])
        hyperparams['objective']['kl_deep']['schedule_start'] = deep_start
        hyperparams['objective']['l2']['schedule_start'] = deep_start
        hyperparams['model']['deep_unfreeze_step'] = deep_start
        orion_hp_string += 'deep_start= %i\n'%deep_start
        
    if args.l2_gen_scale or args.log10_l2_gen_scale:
        if args.log10_l2_gen_scale:
            l2_gen_scale = 10**args.log10_l2_gen_scale
        else:
            l2_gen_scale = args.l2_gen_scale
        hyperparams['objective']['l2_gen_scale'] = l2_gen_scale
        orion_hp_string += 'l2_gen_scale= %.3f\n'%l2_gen_scale
    
    if args.l2_con_scale or args.log10_l2_con_scale:
        if args.log10_l2_con_scale:
            l2_con_scale = 10**args.log10_l2_con_scale
        else:
            l2_con_scale = args.l2_con_scale
        hyperparams['objective']['l2_con_scale'] = l2_con_scale
        orion_hp_string += 'l2_con_scale= %.3f\n'%l2_con_scale
        
    orion_hp_string = orion_hp_string.replace('\n', '-').replace(' ', '').replace('=', '')
    orion_hp_string = '_orion-'+orion_hp_string
        
    return orion_hp_string, hyperparams

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def generate_save_loc(args, hyperparams, orion_hp_string):
    data_name = args.data_path.split(os.path.sep)[-1]
    model_name = hyperparams['model_name']
    if args.data_suffix == 'ospikes':
        model_name += '_oasis'
    mhp_list = [key.replace('size', '').replace('deep', 'd').replace('obs', 'o').replace('_', '')[:4] + str(val) for key, val in hyperparams['model'].items() if 'size' in key]
    mhp_list.append(f'seqlen{args.seq_len}')
    if args.ch_idx is None:
        n_ch = 42 # bad magic number, get dataset info involved
    else:
        n_ch = len(args.ch_idx)
    mhp_list.append(f'nch{n_ch}')
    mhp_list.sort()
    hyperparams['run_name'] = '_'.join(mhp_list)
    hyperparams['run_name'] += orion_hp_string
    save_loc = '%s/%s/%s/%s/'%(args.output_dir, data_name, model_name, hyperparams['run_name'])
    return save_loc, hyperparams

#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def save_figs(save_loc, model, dl, plotter):
    fig_folder = save_loc + 'figs' + os.path.sep
    
    if os.path.exists(fig_folder):
        rmtree(fig_folder)
    os.mkdir(fig_folder)
    
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')
    fig_dict = plotter['valid'].plot_summary(model= model, dl= dl)
    for k, v in fig_dict.items():
        if type(v) == Figure:
            v.savefig(fig_folder+k+'.svg')

#-------------------------------------------------------------------
#-------------------------------------------------------------------

if __name__ == '__main__':
    main()
