import h5py
import os
import json
import numpy as np
import yaml
import torch

def write_data(data_fname, data_dict, use_json=False, compression=None):
    """Write data in HD5F format.

    Args:
    data_fname: The filename of teh file in which to write the data.
    data_dict:  The dictionary of data to write. The keys are strings
      and the values are numpy arrays.
    use_json (optional): human readable format for simple items
    compression (optional): The compression to use for h5py (disabled by
      default because the library borks on scalars, otherwise try 'gzip').
    """

    dir_name = os.path.dirname(data_fname)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if use_json:
        the_file = open(data_fname,'w')
        json.dump(data_dict, the_file)
        the_file.close()
    else:
        try:
            with h5py.File(data_fname, 'w') as hf:
                for k, v in data_dict.items():
                    if type(k) is str:
                        clean_k = k.replace('/', '_')
                        if clean_k is not k:
                            print('Warning: saving variable with name: ', k, ' as ', clean_k)
                        else:
                            print('Saving variable with name: ', clean_k)
                    else:
                        clean_k = k

                    hf.create_dataset(clean_k, data=v, compression=compression)
        except IOError:
            print("Cannot open %s for writing.", data_fname)
                    

def read_data(data_fname):
    
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
            data_dict = {k: np.array(v) for k, v in hf.items()}
            return data_dict
    except IOError:
        print("Cannot open %s for reading." % data_fname)
        raise

def read_multiband_data(data_file_path,n_block):
    data_dir_path = os.path.dirname(data_file_path)
    filt_data_file_path = os.path.join(data_dir_path,f'gw_250_nband{n_block}')
    assert os.path.exists(data_file_path), f'Data file path not found: {data_file_path}'
    assert os.path.exists(filt_data_file_path), f'Filtered data file path not found: {filt_data_file_path}'
    full_data_record = h5py.File(data_file_path,mode='r')
    filt_data_record = h5py.File(filt_data_file_path,mode='r')
    return full_data_record, filt_data_record
        
def batchify_sample(sample, batch_size):
    if not torch.is_tensor(sample):
        sample = torch.Tensor(sample)
    batch = sample.unsqueeze(0).repeat(batch_size, *(1,)*(sample.ndimension()))
    return batch
        
def batchify_random_sample(data, batch_size, ix=None):
    
    """
    Randomly select sample from data, and turn into a batch of size batch_size to generate multiple samples
    from model to average over
    
    Args:
        data (torch.Tensor) : dataset to randomly select sample from
        batch_size (int) : number of sample repeats
    Optional:
        ix (int) : index to select sample. Randomly generated if None
    
    Returns:
        batch (torch.Tensor) : sample repeated batch
        ix (int) : index of repeated sample
    """
    
    num_trials = data.shape[0]
    if type(ix) is not int:
        ix = np.random.randint(num_trials)
    sample = data[ix]
    batch = batchify_sample(sample, batch_size)
    return batch, ix

def update_param_dict(prev_params, new_params):
    params = prev_params.copy()
    for k in prev_params.keys():
        if k in new_params.keys():
            params[k] = new_params[k]
    return params

def load_parameters(path):
    return yaml.load(open(path), Loader=yaml.FullLoader)

def save_parameters(save_loc, params):
    if not os.path.isdir(save_loc):
        os.makedirs(save_loc)
    yaml.dump(params, open(save_loc + 'hyperparameters.yaml', 'w'), default_flow_style=False)
