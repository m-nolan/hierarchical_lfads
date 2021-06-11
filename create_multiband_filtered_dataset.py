# create_multiband_filtered_dataset.py
# 
# Generate an hdf5 record of filtered ECoG data for n-block LFADS training
# Use with paired unfiltered data (gw_250_renorm) for reconstruction training

# Michael Nolan
# 2021-06-10

import os
import h5py

import argparse

from dataset import create_n_block_w, tensor_zscore

import torch
import scipy.signal as sps

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir_path',default=r'D:\Users\mickey\Data\datasets\ecog\goose_wireless',type=str)
parser.add_argument('--n_band',default=2,type=int)
parser.add_argument('--n_iir',default=20,type=int)
parser.add_argument('--n_pad',default=49,type=int)

# load in dataset-to-be-filtered
def get_datasets(dataset_dir_path,n_band):
    dataset_path = os.path.join(dataset_dir_path,'gw_250_renorm')
    dataset_multiband_path = os.path.join(dataset_dir_path,f'gw_250_nband{n_band}')
    dataset = h5py.File(dataset_path,mode='r')
    dataset_multiband = h5py.File(dataset_multiband_path,mode='w')
    return dataset, dataset_multiband

# create filter blocks
def create_w(n_band):
    w = create_n_block_w(n_band)
    w[0] = w[0][1:]
    w[-1] = w[-1][:1]
    return w

# create filtered verions of each dataset, save to multiband dataset
def create_filtered_data(dataset, dataset_multiband, w, n_iir, n_pad):
    for idx, _w in enumerate(w):
        print(f'Filter band {idx}:\t{_w}')
        # create sum-of-squares (sos) filter array
        if idx == 0:
            bandtype = 'lowpass'
        elif idx == len(w) - 1:
            bandtype = 'highpass'
        else:
            bandtype = 'bandpass'
        sos_filter = sps.iirfilter(n_iir, _w, output='sos', btype=bandtype)
        # loop across dataset partitions (and scalar values) in parent dataset, filter each and pack into target dataset
        for k in dataset.keys():
            print(k)
            if k in dataset_multiband.keys():
                print(f'{k} found in output dataset - skipping.')
            else:
                if k == 'dt':
                    dataset_multiband[k] = dataset[k][()]
                else:
                    print('filtering data...')
                    data_filt = torch.tensor(
                        sps.sosfiltfilt(sos_filter, dataset['train_ecog'], axis=1, padlen=n_pad).copy(),
                        dtype=torch.float32
                    )
                    data_filt = tensor_zscore(data_filt,dim=1)
                    k_out = f'band{idx}_' + k
                    dataset_multiband[k_out] = data_filt

# tidy up across open datasets
def close_datasets(*datasets):
    for ds in datasets:
        ds.close()

# main script function
def main(dataset_dir_path,n_band,n_iir,n_pad):
    dataset, dataset_multiband = get_datasets(dataset_dir_path,n_band)
    w = create_w(n_band)
    create_filtered_data(dataset, dataset_multiband, w, n_iir, n_pad)
    close_datasets(dataset, dataset_multiband)
    # :)

if __name__ == '__main__':
    args = parser.parse_args()
    main(
        dataset_dir_path = args.dataset_dir_path,
        n_band = args.n_band,
        n_iir = args.n_iir,
        n_pad = args.n_pad
        )