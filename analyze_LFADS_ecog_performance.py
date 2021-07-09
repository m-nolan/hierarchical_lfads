# analyze_LFADS_ecog_performance.py
#
# Assess ECoG signal reconstruction performance of a given LFADS model.
#   script based on code from notebook create_performance_table_20201104.ipynb
#
# Michael Nolan
# 2021-05-01

from analysis import *
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',type=str,required=True)
parser.add_argument('--ar_model',type=str,required=False,default='')
parser.add_argument('--dataset',type=str,required=True)
parser.add_argument('--pred',action='store_true',default=False)

n_trace = 6
srate = 250
n_boot = 500

def get_paths(args):
    # parse input path arguments and create hyperparamter path
    assert os.path.exists(args.model_dir), f'LFADS model directory {args.model_dir} not found\nterminating analysis.\n'
    assert os.path.exists(args.dataset), f'ECoG dataset {args.dataset} not found\nterminating analysis.\n'
    if os.path.exists(args.ar_model):
        ar_model_path = args.ar_model
    else:
        print('AR model path not recognized. Proceding without AR model comparison.')
        ar_model_path = None
    model_dir = args.model_dir
    dataset_path = args.dataset
    ar_model_path = args.ar_model
    hyperparameter_path = os.path.join(model_dir,'hyperparameters.yaml')
    assert os.path.exists(hyperparameter_path), f'{hyperparameter_path} not found\nterminating analysis.\n'
    return model_dir, dataset_path, ar_model_path, hyperparameter_path

def get_ar_model(ar_model_path):
    if ar_model_path:
        with open(ar_model_path,'rb') as f:
            ar_model_dict = pkl.load(f)
    else:
        ar_model_dict = None
    return ar_model_dict

def save_performance_table(model_dir_path,metric_stat_table_row):
    # save performance results to a new table
    # scan across these and concatenate them with a different analysis. Easy!
    metric_stat_table_file = os.path.join(model_dir_path,'performance_table.csv')
    metric_stat_table_row.to_csv(metric_stat_table_file)

def save_figures(f_trace,f_psd,f_diff,f_loss,psd_data_dict,model_dir_path):
    # save figures from analysis
    fig_path = os.path.join(model_dir_path,'figs')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    with open(os.path.join(fig_path,'psd_data_dict.pkl'),'wb') as f:
        pkl.dump(psd_data_dict,f)
    f_trace.savefig(os.path.join(fig_path,'trace.png'))
    f_trace.savefig(os.path.join(fig_path,'trace.svg'))
    f_psd.savefig(os.path.join(fig_path,'psd.png'))
    f_psd.savefig(os.path.join(fig_path,'psd.svg'))
    f_diff.savefig(os.path.join(fig_path,'error_psd.png'))
    f_diff.savefig(os.path.join(fig_path,'error_psd.svg'))
    f_loss.savefig(os.path.join(fig_path,'loss.png'))
    f_loss.savefig(os.path.join(fig_path,'loss.svg'))

def analyze(model_dir_path, dataset_path, ar_model_path, hyperparameter_path, dec=None, pred=False):
    ar_model_dict = get_ar_model(ar_model_path)
    # metric_stat_table_row, metrics, test_data_mask = get_model_performance_stat_table(model_dir_path, dataset_path, hyperparameter_path)
    # f_trace, f_psd = model_visualization(model_dir_path, dataset_path, hyperparameter_path,None,n_trace, srate,n_boot,metrics)
    metric_stat_table_row, metrics, test_data_mask, f_trace, f_psd, f_diff, psd_data_dict = model_analysis(
        model_dir_path, dataset_path, hyperparameter_path, ar_model_dict, n_trace, 
        srate, n_boot, dec, pred)
    f_loss, loss_data = plot_loss_curves(model_dir_path)
    # save performance table
    save_performance_table(model_dir_path, metric_stat_table_row)
    # save figures
    save_figures(f_trace,f_psd,f_diff,f_loss,psd_data_dict,model_dir_path)
    # close figures to prevent "you have too many figures" warning when looped over multiple directories
    plt.close(f_trace)
    plt.close(f_psd)
    plt.close(f_diff)
    plt.close(f_loss)

if __name__ == "__main__":
    t_start = time.time()
    args = parser.parse_args()
    model_dir_path, dataset_path, ar_model_path, hyperparameter_path = get_paths(args)
    analyze(model_dir_path, dataset_path, ar_model_path, hyperparameter_path,pred=args.pred)
    t_end = time.time()
    print(f'assessment complete. runtime: {t_end-t_start}\n')