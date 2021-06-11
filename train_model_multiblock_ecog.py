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

from dataset import EcogTensorDataset, MultiblockEcogTensorDataset, DropChannels, FilterData, create_n_block_w

from trainer import RunManager
from scheduler import LFADS_Scheduler

from utils import read_data, read_multiband_data, load_parameters, save_parameters
from plotter import Plotter

from math import floor, log

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
parser.add_argument('--block_tdl', action='store_true', default=False)
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
    if args.use_fdl and args.block_tdl:
        save_loc = save_loc + 'only'
    save_loc = save_loc + os.sep
    
    save_parameters(save_loc, hyperparams)
    
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        
    # data_dict   = read_data(args.data_path)
    full_data_record, filt_data_record = read_multiband_data(args.data_path,hyperparams['model']['n_block'])

    mse = args.loss == 'mse'
    
    # create data filter transformer
    w = create_n_block_w(hyperparams['model']['n_block'])
    transforms = FilterData(w=w,
                            n=hyperparams['model']['filter_order'])
    # transforms = torch.jit.script(transforms) # this might not work - update, it didn't work.
    
    train_dl, valid_dl, plotter, model, objective = prep_model(model_name  = args.model,
                                                               data_records   = [full_data_record,filt_data_record],
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
                                                               use_fdl = args.use_fdl,
                                                               use_tdl = not args.block_tdl)
        
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
                             load_checkpoint=(not args.restart),
                             mode='pred') # 'pred' works for the filtered-inputs-to-whole-outputs comparison.
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

def prep_model(model_name, data_records, data_suffix, batch_size, device, hyperparams, seq_len=None, ch_idx=None, multidevice=False, mse=True, attention=False, transform=None, use_fdl=False, use_tdl=True):
    if model_name == 'multiblock_lfads_ecog':
        train_dl, valid_dl, input_dims, plotter = prep_data(data_records=data_records, data_suffix=data_suffix, n_band=hyperparams['model']['n_block'], batch_size=batch_size, device=device, seq_len=seq_len, ch_idx=ch_idx, transform=transform)
        breakpoint()
        model, objective = prep_multiblock_lfads_ecog(input_dims = input_dims,
                                      hyperparams=hyperparams,
                                      device= device,
                                      dtype=train_dl.dataset.dtype,
                                      dt= data_records[0]['dt'],
                                      multidevice=multidevice,
                                      mse=mse,
                                      attention=attention,
                                      use_fdl=use_fdl,
                                      use_tdl=use_tdl)
    else:
        raise NotImplementedError('Model must be ''multiblock_lfads_ecog''.')
        
    return train_dl, valid_dl, plotter, model, objective
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_lfads_ecog(input_dims, hyperparams, device, dtype, dt, multidevice, mse=True, attention=False, use_fdl=True, use_tdl=True):
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
                           use_tdl                  = use_tdl,
                           loss_weight_dict         = {'kl': hyperparams['objective']['kl'], 
                                                       'l2': hyperparams['objective']['l2']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)

    return model, objective

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_multiblock_lfads_ecog(input_dims, hyperparams, device, dtype, dt, multidevice, mse=True, attention=False, use_fdl=False, use_tdl=True):
    from objective import Multiblock_LFADS_Loss, LogLikelihoodGaussian
    from lfads import LFADS_Multiblock_Net

    model = LFADS_Multiblock_Net(   input_size           = input_dims,
                                    factor_size          = hyperparams['model']['factor_size'],
                                    g_encoder_size       = hyperparams['model']['g_encoder_size'],
                                    c_encoder_size       = hyperparams['model']['c_encoder_size'],
                                    g_latent_size        = hyperparams['model']['g_latent_size'],
                                    u_latent_size        = hyperparams['model']['u_latent_size'],
                                    controller_size      = hyperparams['model']['controller_size'],
                                    generator_size       = hyperparams['model']['generator_size'],
                                    n_block              = hyperparams['model']['n_block'],
                                    prior                = hyperparams['model']['prior'],
                                    clip_val             = hyperparams['model']['clip_val'],
                                    dropout              = hyperparams['model']['dropout'],
                                    do_normalize_factors = hyperparams['model']['normalize_factors'],
                                    max_norm             = hyperparams['model']['max_norm'],
                                    device               = device).to(device)
    
    loglikelihood = LogLikelihoodGaussian(mse=mse)

    objective = Multiblock_LFADS_Loss(loglikelihood=loglikelihood,
                           use_fdl=use_fdl,
                           use_tdl=use_tdl,
                           loss_weight_dict         = {'kl': hyperparams['objective']['kl'], 
                                                       'l2': hyperparams['objective']['l2']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)
    
    return model, objective
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_data(data_records, data_suffix, n_band, batch_size, device, seq_len=None, ch_idx=None, transform=None):
    full_data_record = data_records[0]
    filt_data_record = data_records[1]
    if seq_len is None:
        seq_len = full_data_record[f'train_{data_suffix}'].shape[1] # data_suffix use is archaic here, correct!
    if ch_idx is None:
        n_ch = full_data_record[f'train_{data_suffix}'].shape[2]
        ch_idx = torch.arange(n_ch)
    # train_data  = torch.Tensor(data_dict['train_%s'%data_suffix])[:,:seq_len,ch_idx]
    # valid_data  = torch.Tensor(data_dict['valid_%s'%data_suffix])[:,:seq_len,ch_idx]
    
    # num_trials, num_steps, input_size = train_data.shape
    num_trials, num_steps, input_size = full_data_record['train_ecog'].shape
    
    # transform_mask = [True, False]
    # train_ds    = EcogTensorDataset(train_data,train_data,device=device,transform=transform,transform_mask=transform_mask)
    train_ds    = MultiblockEcogTensorDataset(full_data_record, filt_data_record, n_band, part_str='train', device=device)
    # valid_ds    = EcogTensorDataset(valid_data,valid_data,device=device,transform=transform,transform_mask=transform_mask)
    valid_ds    = MultiblockEcogTensorDataset(full_data_record, filt_data_record, n_band, part_str='valid', device=device)
    
    train_dl    = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    valid_dl    = torch.utils.data.DataLoader(valid_ds, batch_size = batch_size)

    breakpoint() # check your train_ds, valid_ds
    
    # TIME = torch.arange(num_steps)*data_dict['dt']
    
    # train_truth = {}
    # if 'train_rates' in data_dict.keys():
    #     train_truth['rates'] = data_dict['train_rates']
    # if 'train_latent' in data_dict.keys():
    #     train_truth['latent'] = data_dict['train_latent']
    # if 'valid_spikes' in data_dict.keys():
    #     train_truth['spikes'] = data_dict['train_spikes']
        
    # valid_truth = {}
    # if 'valid_rates' in data_dict.keys():
    #     valid_truth['rates'] = data_dict['valid_rates']
    # if 'valid_latent' in data_dict.keys():
    #     valid_truth['latent'] = data_dict['valid_latent']
    # if 'valid_spikes' in data_dict.keys():
    #     valid_truth['spikes'] = data_dict['valid_spikes']

    # plotter = {'train' : Plotter(time=TIME, truth=train_truth),
    #            'valid' : Plotter(time=TIME, truth=valid_truth)}
    plotter = {}
    
    return train_dl, valid_dl, input_size, plotter
    
# #-------------------------------------------------------------------
# #-------------------------------------------------------------------
    
# def prep_video(data_dict, batch_size, device):
#     train_dl    = torch.utils.data.DataLoader(SyntheticCalciumVideoDataset(traces= data_dict['train_fluor'], cells=data_dict['cells'], device=device), batch_size=args.batch_size)
#     valid_dl    = torch.utils.data.DataLoader(SyntheticCalciumVideoDataset(traces= data_dict['valid_fluor'], cells=data_dict['cells'], device=device), batch_size=args.batch_size)
    
#     num_trials, num_steps, num_cells = data_dict['train_fluor'].shape
#     num_cells, width, height = data_dict['cells'].shape
    
#     input_dims = (num_steps, width, height)
    
#     TIME = torch._np.arange(0, num_steps*data_dict['dt'], data_dict['dt'])
    
#     train_truth = {}
#     if 'train_latent' in data_dict.keys():
#         train_truth['latent'] = data_dict['train_latent']
        
#     valid_truth = {}
#     if 'valid_latent' in data_dict.keys():
#         valid_truth['latent'] = data_dict['valid_latent']

#     plotter = {'train' : Plotter(time=TIME, truth=train_truth),
#                'valid' : Plotter(time=TIME, truth=valid_truth)}
    
#     return train_dl, valid_dl, input_size, plotter

# #-------------------------------------------------------------------
# #-------------------------------------------------------------------

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
    n_block = hyperparams['model']['n_block']
    mhp_list.append(f'nblock{n_block}')
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
