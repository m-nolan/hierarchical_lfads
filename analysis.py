import torch
from torch.nn.parallel.data_parallel import DataParallel
from statsmodels.tsa.api import VAR
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys
import pickle as pkl
from glob import glob
import site
site.addsitedir(os.path.curdir + '\..')
from utils import load_parameters

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_model(model_name, data_dict, data_suffix, batch_size, device, hyperparams, input_dims=None):
    if model_name == 'lfads_ecog':
        # train_dl, valid_dl, input_dims, plotter = prep_data(data_dict=data_dict, data_suffix=data_suffix, batch_size=batch_size, device=device)
        if not input_dims:
            input_dims = data_dict[f'test_{data_suffix}'].shape[-1]
        model, objective = prep_lfads_ecog(input_dims = input_dims,
                                      hyperparams=hyperparams,
                                      device= device,
                                      dtype=data_dict[f'test_{data_suffix}'].dtype,
                                      dt= data_dict['dt']
                                      )
    return model

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def prep_lfads_ecog(input_dims, hyperparams, device, dtype, dt):
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
                                    prior                = hyperparams['model']['prior'],
                                    clip_val             = hyperparams['model']['clip_val'],
                                    dropout              = hyperparams['model']['dropout'],
                                    do_normalize_factors = hyperparams['model']['normalize_factors'],
                                    max_norm             = hyperparams['model']['max_norm'],
                                    device               = device).to(device)
    
    loglikelihood = LogLikelihoodGaussian()

    objective = LFADS_Loss(loglikelihood            = loglikelihood,
                           loss_weight_dict         = {'kl': hyperparams['objective']['kl'], 
                                                       'l2': hyperparams['objective']['l2']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)

    return model, objective

#-------------------------------------------------------------------
#-------------------------------------------------------------------
    
def prep_data(data_dict, data_suffix, batch_size, device, input_size=None):
    train_data  = torch.Tensor(data_dict['train_%s'%data_suffix])
    valid_data  = torch.Tensor(data_dict['valid_%s'%data_suffix])
    
    num_trials, num_steps, _ = train_data.shape
    if not input_size: # very hacky, I apologize to the lord
        input_size = train_data.shape(-1)
    
    train_ds    = EcogTensorDataset(train_data,device=device)
    valid_ds    = EcogTensorDataset(valid_data,device=device)
    
    train_dl    = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    valid_dl    = torch.utils.data.DataLoader(valid_ds, batch_size = batch_size)
    
    TIME = torch._np.arange(0, num_steps*data_dict['dt'], data_dict['dt'])
    
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

    plotter = None
    # plotter = {'train' : Plotter(time=TIME, truth=train_truth),
    #            'valid' : Plotter(time=TIME, truth=valid_truth)}
    
    return train_dl, valid_dl, input_size, plotter

#-------------------------------------------------------------------
#-------------------------------------------------------------------

from torch.utils.data.dataset import Dataset
class EcogTensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, device='cpu'):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.device = device

    def __getitem__(self, index):
        return tuple(tensor[index].to(self.device) for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

#-------------------------------------------------------------------
#-------------------------------------------------------------------

import h5py
def read_data(data_fname,keys):
    
    """ Read saved data in HDF5 format.

    Args:
        data_fname: The filename of the file from which to read the data.
    Returns:
        A dictionary whose keys will vary depending on dataset (but should
        always contain the keys 'train_data' and 'valid_data') and whose
        values are numpy arrays.
    """
    try:
        with h5py.File(data_fname, 'r') as hf:
            data_dict = {k: np.array(v) for k, v in hf.items() if k in keys}
            return data_dict
    except IOError:
        print("Cannot open %s for reading." % data_fname)
        raise

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def compute_prediction_metrics(test_data,pred,pred_window_T,bin_T,p_lim=[2.5, 97.5],srate=250):
    n_trial, n_sample, n_ch = test_data.shape
    time = np.arange(pred_window_T*srate)/srate
    bin_T_left_edge = np.arange(pred_window_T,step=bin_T)
    bin_T_right_edge = bin_T_left_edge + bin_T
    n_time_bin = len(bin_T_left_edge)
    mae, rmse, rpe = compute_prediction_error(test_data, pred)
    mae_bin = np.empty((n_trial,n_time_bin))
    rmse_bin = np.empty((n_trial,n_time_bin))
    rpe_bin = np.empty((n_trial,n_time_bin))
    for tb_idx in range(n_time_bin):
        bin_idx = np.logical_and(time >= bin_T_left_edge[tb_idx], time < bin_T_right_edge[tb_idx])
        mae_bin[:,tb_idx], rmse_bin[:,tb_idx], rpe_bin[:,tb_idx] = compute_prediction_error(test_data[:,bin_idx,:],pred[:,bin_idx,:])
    # get stats from sample distributions
    stat_dict = {
        'mae_mean': mae.mean(),
        'mae_95ci': np.percentile(mae,p_lim),
        'mae_bin_mean': mae_bin.mean(axis=0),
        'mae_bin_95ci': np.percentile(mae_bin,p_lim,axis=0),
        'rmse_mean': rmse.mean(),
        'rmse_95ci': np.percentile(rmse,p_lim),
        'rmse_bin_mean': rmse_bin.mean(axis=0),
        'rmse_bin_95ci': np.percentile(rmse_bin,p_lim,axis=0),
        'rpe_mean': rpe.mean(),
        'rpe_95ci': np.percentile(rpe,p_lim),
        'rpe_bin_mean': np.nanmean(rpe_bin,axis=0),
        'rpe_bin_95ci': np.nanpercentile(rpe_bin,p_lim,axis=0),
        # 'corr_mean': np.tanh(np.arctanh(corr).mean(axis=0)),
        # 'corr_95ci': np.percentile(corr,p_lim,axis=0),
        # 'corr_bin_mean': np.tanh(np.arctanh(corr_bin).mean(axis=0)),
        # 'corr_bin_95ci': np.percentile(corr_bin,p_lim,axis=0)
    }
    return stat_dict, bin_T_left_edge

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def compute_prediction_error(trg,pred):
    err = trg - pred
    mae = np.abs(err).mean(axis=(1,2))
    rmse = np.sqrt((err**2).mean(axis=(1,2)))
    trg_std = trg.std(axis=1)
    rpe = (err.std(axis=1)/trg_std).mean(axis=-1)
    rpe[np.isinf(rpe)] == np.nan
    return mae, rmse, rpe

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def bootstrap_est(data,n_boot,f):
    n_sample = data.shape[0]
    est = []
    for n in range(n_boot):
        _idx = np.random.choice(np.arange(n_sample),size=n_sample,replace=True)
        est.append(f(data[_idx,]))
    est = np.stack(est,axis=0)
    return est

def get_model_params(model_dir_path):
    # break down model_dir_path
    model_key_list = model_dir_path.split("\\")
    conf_str = model_key_list[-1]
    model_name = model_key_list[-2]
    data_keys = model_key_list[-3].split("_")
    if len(data_keys) > 2:
        data_suffix = 'ecog_' + data_keys[2]
    else:
        data_suffix = 'ecog'
    # break down conf_str
    conf_key_list = conf_str.split("_")
    n_ch = int(conf_key_list[-4][3:])
    seq_len = int(conf_key_list[-3][6:])
    return model_name, data_suffix, n_ch, seq_len

#-------------------------------------------------------------------
#-------------------------------------------------------------------

class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    def __setattr__(self, name, value):
        try:
            return super().__setattr__(name,value)
        except AttributeError:
            return setattr(self.module, name, value)

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def load_configure_model_data(model_dir_path,data_path,hyperparameter_path,std_thresh=0.5):
    model_name, data_suffix, n_ch, seq_len = get_model_params(model_dir_path)
    batch_size = 1000
    device = 'cpu'
    hyperparams = load_parameters(hyperparameter_path)
    data_dict   = read_data(data_path,keys = [f'test_{data_suffix}','dt'])
    model= prep_model(model_name = model_name,
                                            data_dict = data_dict,
                                            data_suffix = data_suffix,
                                            batch_size = batch_size,
                                            device = device,
                                            hyperparams = hyperparams,
                                            input_dims= n_ch)
    checkpoint_file_path = os.path.join(model_dir_path,"checkpoints","best.pth")
    checkpoint = torch.load(checkpoint_file_path,map_location=device)
    if list(checkpoint['net'].keys())[0][0:6] == 'module': # very dirty
        model = DataParallelPassthrough(model)
    model.load_state_dict(checkpoint['net'], strict=False)
    if list(checkpoint['net'].keys())[0][0:6] == 'module':
        model = model.module
        model.g_posterior_mean = checkpoint['net']['module.g_posterior_mean']
        model.g_posterior_logvar = checkpoint['net']['module.g_posterior_logvar']
    # grab data
    if n_ch <= 32:
        test_data = np.array(data_dict[f"test_{data_suffix}"][:,:seq_len,10:10+n_ch],dtype=np.float32)
    else:
        test_data = np.array(data_dict[f"test_{data_suffix}"][:,:seq_len,:n_ch],dtype=np.float32)
    test_data_mask = test_data.std(axis=(1,2)) < std_thresh
    test_data = test_data[~test_data_mask,:,:]
    return model.to('cpu'), test_data, test_data_mask

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def compute_model_outputs(model,test_data):
    model.eval()
    with torch.no_grad():
        recon, (factors, generators, gen_inputs) = model.forward_all(test_data)
    return recon, factors, generators

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def compute_metric_table(test_data,recon,model_dir_path):
    n_trial, n_sample, n_ch = test_data.shape
    model_name, data_suffix, n_ch, seq_len = get_model_params(model_dir_path)
    # compute metrics
    mse = np.mean((recon['data'].numpy() - test_data)**2,axis=(1,2))
    rmse = np.sqrt(mse)
    mae = np.abs(recon['data'].numpy() - test_data).mean(axis=(1,2))
    trg_std = np.std(test_data, axis=(1,2))
    rpe = rmse/trg_std
    corr = np.array([np.corrcoef(t.T,r.T)[0,1] for t,r in zip(test_data.reshape(n_trial,-1),recon['data'].numpy().reshape(n_trial,-1))])
    # compute statistics
    f_est = lambda x: np.nanmean(x)
    f_zcorr_est = lambda x: np.nanmean(np.tanh(x))
    n_boot = 1000
    mse_bsd = bootstrap_est(mse,n_boot,f_est)
    rmse_bsd = bootstrap_est(rmse,n_boot,f_est)
    mae_bsd = bootstrap_est(mae,n_boot,f_est)
    rpe_bsd = bootstrap_est(rpe,n_boot,f_est)
    zcorr_bsd = bootstrap_est(corr,n_boot,f_zcorr_est) # fisher transform for better stats
    stat_dict = {
        'model_path': model_dir_path,
        'model_name': model_name,
        'data_suffix': data_suffix,
        'n_ch': n_ch,
        'seq_len': seq_len,
        'seq_t': seq_len/250, # whoopsie with the magic number
        'mse_mean': [mse_bsd.mean()],
        'mse_mean_2.5ci': [np.percentile(mse_bsd,2.5)],
        'mse_mean_97.5ci': [np.percentile(mse_bsd,97.5)],
        'mse_2.5ci': [np.percentile(mse,2.5)],
        'mse_97.5ci': [np.percentile(mse,97.5)],
        'rmse_mean': [rmse_bsd.mean()],
        'rmse_mean_2.5ci': [np.percentile(rmse_bsd,2.5)],
        'rmse_mean_97.5ci': [np.percentile(rmse_bsd,97.5)],
        'rmse_2.5ci': [np.percentile(rmse,2.5)],
        'rmse_97.5ci': [np.percentile(rmse,97.5)],
        'mae_mean': [mae_bsd.mean()],
        'mae_mean_2.5ci': [np.percentile(mae_bsd,2.5)],
        'mae_mean_97.5ci': [np.percentile(mae_bsd,97.5)],
        'mae_2.5ci': [np.percentile(mae,2.5)],
        'mae_97.5ci': [np.percentile(mae,97.5)],
        'rpe_mean': [rpe_bsd.mean()],
        'rpe_mean_2.5ci': [np.percentile(rpe_bsd,2.5)],
        'rpe_mean_97.5ci': [np.percentile(rpe_bsd,97.5)],
        'rpe_2.5ci': [np.percentile(rpe,2.5)],
        'rpe_97.5ci': [np.percentile(rpe,97.5)],
        'corr_mean': [np.arctanh(zcorr_bsd.mean())],
        'corr_mean_2.5ci': [np.arctanh(np.percentile(zcorr_bsd,2.5))],
        'corr_mean_97.5ci': [np.arctanh(np.percentile(zcorr_bsd,97.5))],
        'corr_2.5ci': [np.percentile(corr,2.5)],
        'corr_97.5ci': [np.percentile(corr,97.5)],
    }
    stat_table = pd.DataFrame.from_dict(stat_dict)
    metric_dict = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'rpe': rpe,
        'corr': corr,
    }
    return stat_table, metric_dict

#-------------------------------------------------------------------
#-------------------------------------------------------------------

def get_model_performance_stat_table(model_dir_path,data_path,hyperparameter_path):
    print(f'loading model from:\t{model_dir_path}')
    model, test_data, test_data_mask = load_configure_model_data(model_dir_path,data_path,hyperparameter_path)
    print('computing test data reconstructions...')
    recon, factors, generators = compute_model_outputs(model,torch.tensor(test_data))
    print('computing metric statistics...')
    stat_table, metric_dict = compute_metric_table(test_data,recon,model_dir_path)
    return stat_table, metric_dict, test_data_mask

def plot_loss_curves(model_dir):
    # get loss curves
    tensorboard_dir = os.path.join(model_dir,'tensorboard')
    loss_dirs = glob(os.path.join(tensorboard_dir,'*_Loss_*'))
    loss_dict = {}
    for loss_dir in loss_dirs:
        loss_base = os.path.basename(loss_dir)
        ea = event_accumulator.EventAccumulator(os.path.join(tensorboard_dir,loss_dir))
        ea.Reload()
        loss_keys = ea.Tags()['scalars']
        for loss_key in loss_keys:
            _loss_data = np.array(ea.Scalars(loss_key))[:,-1]
            loss_dict[loss_base] = _loss_data
    loss_data = pd.DataFrame(loss_dict)
    # plot loss curves
    n_c = len(loss_dirs)
    n_r = 1
    fig, ax = plt.subplots(1,1,dpi=150,constrained_layout=True)
    sns.lineplot(data=loss_data,ax=ax)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Training Loss')
    return fig, loss_data

def plot_test_data_fits(recon,test_data,ar_model_dict,test_data_mask,n,srate,metrics,trial_idx=None):
    # assume 1ch for now
    n_trials = test_data.shape[0]
    n_time = test_data.shape[1]
    time = np.arange(n_time)/srate
    if trial_idx == None:
        trial_idx = np.random.choice(np.arange(n_trials),n,replace=False) # replace this with fixed values to recreate the poster traces
    else:
        None
    n_r = int(np.ceil(np.sqrt(n)))
    n_c = int(np.ceil(n/n_r))
    fig, ax = plt.subplots(n_r,n_c,dpi=150,constrained_layout=True,sharex=True,figsize=(6,6))
    ax = ax.reshape(-1)
    for idx, t_idx in enumerate(trial_idx):
        r_idx = idx // n_c
        c_idx = idx % n_c
        ax[idx].plot(time,test_data[t_idx,:,0],label='target')
        ax[idx].plot(time,recon['data'][t_idx,:,0],label='recon.')
        ax[idx].plot(time[ar_model_dict['ord']:],ar_model_dict['test_pred'][~test_data_mask,:,:][t_idx,:(n_time-ar_model_dict['ord']),0],label='AR')
        if r_idx == n_r - 1:
            ax[idx].set_xlabel('time (s)')
        if c_idx == 0:
            ax[idx].set_ylabel('a.u.')
        ax[idx].set_title(f'trial {t_idx}')
        metric_str = f"mse: {metrics['mse'][t_idx]:0.3f}\nrpe: {metrics['rpe'][t_idx]:0.3f}\ncorr: {metrics['corr'][t_idx]:0.3f}"
        text_x = ax[idx].get_xlim()[0]
        text_y = ax[idx].get_ylim()[0]
        ax[idx].text(text_x,text_y,metric_str,horizontalalignment='left',verticalalignment='bottom',fontsize=8,bbox=dict(alpha=0.1))
    ax[0].legend(loc=0)
    return fig, ax

def plot_test_data_fits_psd(recon, test_data, ar_model_dict, test_data_mask, srate, n_boot):
    # power features
    from scipy.signal import welch, detrend
    trial_mask = test_data[:,:,0].std(axis=1) < 0.5
    f_psd, data_psd = welch(detrend(test_data[~trial_mask,],type='linear',axis=-2),fs=srate,axis=1) # why axis=-2? multiple batch acceptance? Weird
    _, recon_psd = welch(detrend(recon['data'][~trial_mask,],type='linear',axis=-2),fs=srate,axis=1)
    _, diff_psd = welch(detrend(test_data[~trial_mask,]-recon['data'][~trial_mask,].numpy(),type='linear',axis=-2),fs=srate,axis=1)
    f_ar_psd, ar_psd = welch(detrend(ar_model_dict['test_pred'][~test_data_mask,][~trial_mask,],type='linear',axis=-2),fs=srate,axis=1)
    f_est = lambda x: x.mean(axis=0)
    data_psd_bsd = bootstrap_est(data_psd[:,:,0], n_boot, f_est) # all of this is JUST THE 1st CHANNEL? Why? Fix that later.
    data_psd_mean = data_psd_bsd.mean(axis=0)
    data_psd_95ci = np.percentile(data_psd_bsd,[2.5, 97.5],axis=0)
    recon_psd_bsd = bootstrap_est(recon_psd[:,:,0], n_boot, f_est)
    recon_psd_mean = recon_psd_bsd.mean(axis=0)
    recon_psd_95ci = np.percentile(recon_psd_bsd,[2.5, 97.5],axis=0)
    ar_psd_bsd = bootstrap_est(ar_psd[:,:,0], n_boot, f_est)
    ar_psd_mean = ar_psd_bsd.mean(axis=0)
    ar_psd_95ci = np.percentile(ar_psd_bsd,[2.5,97.5],axis=0)
    diff_psd_bsd = bootstrap_est(diff_psd[:,:,0],n_boot,f_est)
    diff_psd_mean = diff_psd_bsd.mean(axis=0)
    diff_psd_95ci = np.percentile(diff_psd_bsd,[2.5, 97.5],axis=0)
    fig, ax = plt.subplots(1,1,dpi=100,sharex=True)
    ax.fill_between(f_psd, 10*np.log10(data_psd_95ci[0,:]), 10*np.log10(data_psd_95ci[1,:]),alpha=0.2,label='data 95% ci')
    ax.plot(f_psd, 10*np.log10(data_psd_mean), label='data mean');
    ax.fill_between(f_psd, 10*np.log10(recon_psd_95ci[0,:]), 10*np.log10(recon_psd_95ci[1,:]),alpha=0.2,label='recon. 95% ci')
    ax.plot(f_psd, 10*np.log10(recon_psd_mean), label='recon. mean');
    ax.fill_between(f_ar_psd, 10*np.log10(ar_psd_95ci[0,:]), 10*np.log10(ar_psd_95ci[1,:]), alpha=0.2, label='AR 95% ci')
    ax.plot(f_ar_psd, 10*np.log10(ar_psd_mean), label='AR mean')
    ax.fill_between(f_psd, 10*np.log10(diff_psd_95ci[0,:]), 10*np.log10(diff_psd_95ci[1,:]), color='k', alpha=0.2, label='err. 95% ci')
    ax.plot(f_psd, 10*np.log10(diff_psd_mean), color='k', label='err. mean')
    ax.legend(loc=0)
    # consider changing this to relative error distribution instead of a straight residual.
    fig_diff, ax_diff = plt.subplots(1,1,dpi=100)
    ax_diff.fill_between(f_psd, 10*np.log10(diff_psd_95ci[0,:]), 10*np.log10(diff_psd_95ci[1,:]), color='k', alpha=0.2, label='err 95% ci')
    ax_diff.plot(f_psd, 10*np.log10(diff_psd_mean), color='k', label='err mean')
    ax_diff.legend(loc=0)
    ax.set_xlabel('freq. (Hz)')
    ax.set_ylabel('PSD (dB)')
    ax.set_title('Power Spectral Density, Data v. Reconstruction')
    ax.set_xlim(0,100)
    ax_diff.set_xlabel('freq. (Hz)')
    ax_diff.set_ylabel('PSD (dB)')
    ax_diff.set_title('Power Spectral Density, Reconstruction Error')
    ax_diff.set_xlim(0,100)
    return fig, ax, fig_diff, ax_diff, {'data_psd_mean': data_psd_mean,
                     'data_psd_95ci': data_psd_95ci,
                     'recon_psd_mean': recon_psd_mean,
                     'recon_psd_95ci': recon_psd_95ci,
                     'ar_psd_mean': ar_psd_mean,
                     'ar_psd_95ci': ar_psd_95ci,
                     'diff_psd_mean': diff_psd_mean,
                     'diff_psd_95ci': diff_psd_95ci,
                     'f_psd': f_psd,
                     'f_ar_psd': f_ar_psd}

def model_visualization(model_dir_path,data_path,hyperparameter_path,ar_model_dict,n,srate,n_boot,metrics):
    print(f'loading model from:\t{model_dir_path}')
    model, test_data, test_data_mask = load_configure_model_data(model_dir_path,data_path,hyperparameter_path)
    print('computing test data reconstructions...')
    recon, factors, generators = compute_model_outputs(model,torch.tensor(test_data))
    print('Generating test plots...')
    f_trace, _ = plot_test_data_fits(recon, test_data, ar_model_dict, test_data_mask, n, srate, metrics, trial_idx=[11149, 11086, 3908, 7024, 2172, 5330])
    f_psd, _, f_diff, _, psd_data_dict = plot_test_data_fits_psd(recon, test_data, ar_model_dict, test_data_mask, srate, n_boot)
    return f_trace, f_psd, f_diff, psd_data_dict

# - - -- --- ----- -------- ----- --- -- - -
# - - --  a better analysis function  -- - -
# - - -- --- ----- -------- ----- --- -- - -

def model_analysis(model_dir_path,data_path,hyperparameter_path,ar_model_dict,n,srate,n_boot):
    print(f'loading model from:\t{model_dir_path}')
    model, test_data, test_data_mask = load_configure_model_data(model_dir_path,data_path,hyperparameter_path)
    print('computing test data reconstructions...')
    recon, factors, generators = compute_model_outputs(model,torch.tensor(test_data))
    print('computing metric statistics...')
    stat_table, metric_dict = compute_metric_table(test_data,recon,model_dir_path)
    print('Generating test plots...')
    f_trace, _ = plot_test_data_fits(recon, test_data, ar_model_dict, test_data_mask, n, srate, metric_dict, trial_idx=[11149, 11086, 3908, 7024, 2172, 5330])
    f_psd, _, f_diff, _, psd_data_dict = plot_test_data_fits_psd(recon, test_data, ar_model_dict, test_data_mask, srate, n_boot)
    return stat_table, metric_dict, test_data_mask, f_trace, f_psd, f_diff, psd_data_dict