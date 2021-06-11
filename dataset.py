from typing import Iterable
import torch
import torch.nn.functional as F
# from torch._C import dtype, float32
import torchvision
from scipy.stats import zscore
import scipy.signal as sps
from math import floor
# from torch.utils.data.dataloader import _DataLoaderIter

class LFADS_MultiSession_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_list, device='cpu'):
        super(LFADS_MultiSession_Dataset, self).__init__()
        
        self.data_list   = data_list
        self.device      = device
        self.tensor_list = []
        
        for data in self.data_list:
            self.tensor_list.append(torch.Tensor(data).to(self.device))
            
    def __getitem__(self, ix):
        try:
            return self.tensor_list[ix], ix
        except KeyError:
            raise StopIteration
            
    def __len__(self):
        return len(self.tensor_list)
    
default_collate = torch.utils.data.dataloader._utils.collate.default_collate

class SessionLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, session_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        
        super(SessionLoader, self).__init__(dataset=dataset,
                                            batch_size=session_size,
                                            shuffle=shuffle,
                                            sampler=sampler,
                                            batch_sampler=batch_sampler,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn,
                                            pin_memory=pin_memory,
                                            drop_last=drop_last,
                                            timeout=timeout,
                                            worker_init_fn=worker_init_fn)
        
    def __iter__(self):
        return _SessionLoaderIter(self)
    
# class _SessionLoaderIter(_DataLoaderIter):
    
#     def __init__(self, loader):
#         super(_SessionLoaderIter, self).__init__(loader)
        
#     def __next__(self):
#         x, idx = super(_SessionLoaderIter, self).__next__()
#         x = x.squeeze()
#         setattr(x, 'session', idx)
#         return x,

#-------------------------------------------------------------------
#-------------------------------------------------------------------
class EcogTensorDataset(torch.utils.data.Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, device='cpu', transform=None, transform_mask=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.device = device
        self.transform = transform
        if transform_mask:
            assert len(self.tensors) == len(transform_mask), f'transform_mask length ({len(transform_mask)}) must match number of tensors ({len(tensors)}).'
        else:
            transform_mask = [True] * len(self.tensors) # all-hot mask
        self.transform_mask = transform_mask


    def __getitem__(self, index):
        # get samples
        sample = [tensor[index] for tensor in self.tensors]
        # apply transform
        if self.transform:
            for idx, s in enumerate(sample):
                if self.transform_mask[idx]:
                    sample[idx] = self.transform(s)
        # assign device
        sample = list_or_tuple_recursive_to(sample,self.device)
        return sample

    def __len__(self):
        return self.tensors[0].size(0)

class MultiblockEcogTensorDataset(torch.utils.data.Dataset):
    r'''
    Dataset wrapping: 
        (1) a full-band ECoG sample h5 record
        (2) a multi-band filtered ECoG sample h5 record

    Data samples are returned as a list. The first element is a list of the band-filtered data samples, while the second element is a tensor of the full-band data sample.

    Filtered samples are filtered from the full-band sample in each draw.

    Arguments:
        - data_path (str):      File path to full-band ECoG data record
        - filt_data_path (str): File path to filtered ECoG data record
    '''

    def __init__(self,data_record,filt_data_record,n_band,part_str,device='cpu'):
        self.data_record = data_record
        self.filt_data_record = filt_data_record
        self.n_band = n_band
        self.part_str = part_str
        self.device = device
        assert self.part_str in ['train','valid','test'], f'Invalid partition string. {self.part_str} not in [train,valid,test].'
        assert f'' in self.filt_data_record.keys(), f'(n_band = {self.n_band}) key {self.partition_band_key(self.n_band-1)} not found in filt_data_record.'

    def __getitem__(self,index):
        filt_sample_list = []
        for b_idx in range(self.n_band):
            b_key = self.partition_band_key(b_idx)
            filt_sample_list.append(
                torch.tensor(
                    self.filt_data_record[b_key][index,:,:],
                    dtype=torch.float32
                ).to(self.device)
            )
        full_sample = torch.tensor(
            self.data_record[f'{self.part_str}_ecog'][index,:,:],
            dtype=torch.float32
        ).to(self.device)
        return [filt_sample_list,full_sample]
    
    def __len__(self):
        return self.data_record.shape[0]

    def partition_band_key(self,idx):
        return f'band{idx}_{self.part_str}_ecog'

def list_or_tuple_recursive_to(x,device):
    if isinstance(x,(list,tuple)):
        x = [list_or_tuple_recursive_to(_x,device) for _x in x]
    else:
        x = x.to(device)
    return x

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
# data filtering transform
class FilterData(torch.nn.Module):
    '''
        Dataset transform to filter data samples. Any number of filter bands are allowed. Each filter is interpreted as a bandpass filter requiring 2 corner frequencies.
        A fixed-order IIR filter is used to filter data to each given window.
        Samples are interpreted as a [time x n_ch x ...] tensor and are filtered along the first dimension (pre-batch). 
        Dataset filtered samples are returned in a tuple.

        Inputs:
            - w:    List-like of normalized frequency windows defining each data filter. Each window is defined by a 2-element array or list.
            - n:    IIR filter order.

        Outputs:
            - samples:  List of filtered samples, as defined by w.
    '''

    def __init__(self,w,n,padlen=49,normalize=True):
        super(FilterData, self).__init__()
        self.w = w
        self.n = n
        self.padlen = padlen
        self.normalize = normalize

        # parse w
        # -- add this when you consider using variably-formatted 
        
        # create array of filters
        self.filters = []
        self.btypes = []
        for idx, _w in enumerate(self.w):
            if len(_w) < 2:
                if idx == 0:
                    btype = 'lowpass'
                elif idx == len(self.w) - 1:
                    btype = 'highpass'
            else:
                btype = 'bandpass'
            self.filters.append(
                sps.iirfilter(
                    self.n,
                    _w,
                    btype=btype,
                    rs=60,
                    ftype='butter',     # consider 'bessel' if group delays are an issue
                    output='sos'
                    )
                )
            self.btypes.append(btype)

    def forward(self,sample):
        samples_filt = []
        for idx, f in enumerate(self.filters):
            samples_filt.append(
                torch.tensor(sps.sosfiltfilt( # this type conversion may not be necessary
                    f,
                    sample,
                    axis=0,
                    padlen=self.padlen
                    ).copy(),
                    dtype=torch.float32
                ))
            if self.btypes[idx] == 'lowpass':
                None
            else:
                samples_filt[-1] -= samples_filt[-1].mean(dim=0)
            if self.normalize:
                samples_filt[-1] = tensor_zscore(samples_filt[-1],dim=0)
            
        return samples_filt

# create n uniform filter blocks for the FilterData transform shown above
def create_n_block_w(n_block):
    bandwidth = 1/n_block
    w_c = torch.arange(n_block) * bandwidth
    w_c[0] = 0.01
    w_c = torch.cat([w_c,torch.tensor([0.99])])
    w = []
    for idx in range(n_block):
        w.append([w_c[idx],w_c[idx+1]])
    return w

# z-scoring for tensors in pytorch.
def tensor_zscore(x,dim=0):
    mean = x.mean(dim=dim).expand([50,-1,-1]).permute(1,0,2)
    std = x.std(dim=dim).expand([50,-1,-1]).permute(1,0,2)
    return (x - mean) / std

#-------------------------------------------------------------------
#-------------------------------------------------------------------