import torch
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

    def __init__(self, *tensors, device='cpu', transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
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
# data filtering transform
class FilterData(object):
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

    def __call__(self,sample):
        samples_filt = []
        for idx, f in enumerate(self.filters):
            samples_filt.append(
                torch.tensor(sps.sosfiltfilt( # this type conversion may not be necessary
                    f,
                    sample,
                    axis=0,
                    padlen=self.padlen
                    ).copy()
                ))
            if self.btypes[idx] == 'lowpass':
                None
            else:
                samples_filt[-1] -= samples_filt[-1].mean(dim=0)
            if self.normalize:
                samples_filt[-1] = torch.tensor(zscore(samples_filt[-1],axis=0))
        return samples_filt