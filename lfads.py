import torch
import torch.nn as nn
import torch.nn.functional as F
from objective import kldiv_gaussian_gaussian
from rnn import LFADS_GenGRUCell
from math import log
import pdb
import time
# for debug
# from tqdm import tqdm

class LFADS_Net(nn.Module):
    '''
    LFADS_Net (Latent Factor Analysis via Dynamical Systems) neural network class.
    
    __init__(self, input_size, factor_size = 4,
                   g_encoder_size = 64, c_encoder_size = 64,
                   g_latent_size = 64, u_latent_size = 1,
                   controller_size= 64, generator_size = 64,
                   prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                    'var'  : {'value': 0.1, 'learnable' : False}},
                             'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                    'var'  : {'value': 0.1, 'learnable' : True},
                                     'tau'  : {'value': 10,  'learnable' : True}}},
                   clip_val=5.0, dropout=0.0, max_norm = 200,
                   do_normalize_factors=True, device='cpu')
                   
    Required Arguments:
        - input_size (int) : size of input dimensions (number of cells)
    Optional Arguments:
        - g_encoder_size     (int): size of generator encoder network
        - c_encoder_size     (int): size of controller encoder network
        - g_latent_size      (int): size of generator ic latent variable
        - u_latent_size      (int): size of generator input latent variable
        - controller_size    (int): size of controller network
        - generator_size     (int): size of generator network
        - prior             (dict): dictionary of prior distribution parameters
        - clip_val         (float): RNN hidden state value limit
        - dropout          (float): dropout probability
        - max_norm           (int): maximum gradient norm
        - do_normalize_factors (bool): whether to normalize factors
        - device          (string): device to use
    '''
    
    def __init__(self, input_size, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 attn = False,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):
        
        super(LFADS_Net, self).__init__()
        
        self.input_size           = input_size
#         self.output_size          = input_size if output_size is None else output_size
        self.g_encoder_size       = g_encoder_size
        self.c_encoder_size       = c_encoder_size
        self.g_latent_size        = g_latent_size
        self.u_latent_size        = u_latent_size
        self.controller_size      = controller_size
        self.generator_size       = generator_size
        self.factor_size          = factor_size
        
        self.clip_val             = clip_val
        self.max_norm             = max_norm
        self.do_normalize_factors = do_normalize_factors
        self.factor_bias          = factor_bias
        self.attn                 = attn
        self.device               = device
        self.deep_freeze          = deep_freeze
        
        self.dropout              = torch.nn.Dropout(dropout)

        # Initialize encoder RNN
        self.encoder     = LFADS_Encoder(input_size     = self.input_size,
                                         g_encoder_size = self.g_encoder_size,
                                         c_encoder_size = self.c_encoder_size,
                                         g_latent_size  = self.g_latent_size,
                                         clip_val       = self.clip_val,
                                         dropout        = dropout)
        
        # Initialize controller RNN
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.controller  = LFADS_ControllerCell(input_size      = self.c_encoder_size*2 + self.factor_size,
                                                    controller_size = self.controller_size,
                                                    u_latent_size   = self.u_latent_size,
                                                    clip_val        = self.clip_val,
                                                    dropout         = dropout)
        
        # Initialize generator RNN
        if self.attn:
            self.generator  = LFADS_Generator_SourceAttnCell(input_size     = self.u_latent_size,
                                                generator_size = self.generator_size,
                                                factor_size    = self.factor_size,
                                                source_size    = self.input_size,
                                                clip_val       = self.clip_val,
                                                factor_bias    = self.factor_bias,
                                                dropout        = dropout)
        else:
            self.generator   = LFADS_GeneratorCell(input_size     = self.u_latent_size,
                                                generator_size = self.generator_size,
                                                factor_size    = self.factor_size,
                                                clip_val       = self.clip_val,
                                                factor_bias    = self.factor_bias,
                                                dropout        = dropout)
        
        # Initialize dense layers
        if self.g_latent_size == self.generator_size:
            self.fc_genstate = Identity(in_features=self.g_latent_size, out_features=self.generator_size)
        else:
            self.fc_genstate = nn.Linear(in_features= self.g_latent_size, out_features= self.generator_size)
                
        # Initialize learnable biases
        self.g_encoder_init  = nn.Parameter(torch.zeros(2, self.g_encoder_size))
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.c_encoder_init  = nn.Parameter(torch.zeros(2, self.c_encoder_size))
            self.controller_init = nn.Parameter(torch.zeros(self.controller_size))
        
        # Initialize priors
        self.register_buffer('g_prior_mean',None)
        self.register_buffer('g_prior_logvar',None)
        self.register_buffer('g_posterior_mean',None)
        self.register_buffer('g_posterior_logvar',None) # these become large arrays and should not be singleton values.
        
        self.g_prior_mean = torch.ones(self.g_latent_size, device=device) * prior['g0']['mean']['value']
        
        if prior['g0']['mean']['learnable']:
            self.g_prior_mean = nn.Parameter(self.g_prior_mean)
        self.g_prior_logvar = torch.ones(self.g_latent_size, device=device) * log(prior['g0']['var']['value'])
        if prior['g0']['var']['learnable']:
            self.g_prior_logvar = nn.Parameter(self.g_prior_logvar)
            
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.u_prior_gp_mean = torch.ones(self.u_latent_size, device=device) * prior['u']['mean']['value']
            if prior['u']['mean']['learnable']:
                self.u_prior_gp_mean = nn.Parameter(self.u_prior_gp_mean)
            self.u_prior_gp_logvar = torch.ones(self.u_latent_size, device=device) * log(prior['u']['var']['value'])
            if prior['u']['var']['learnable']:
                self.u_prior_gp_logvar = nn.Parameter(self.u_prior_gp_logvar)
            self.u_prior_gp_logtau = torch.ones(self.u_latent_size, device=device) * log(prior['u']['tau']['value'])
            if prior['u']['tau']['learnable']:
                self.u_prior_gp_logtau = nn.Parameter(self.u_prior_gp_logtau)
        
        # Initialize weights
        self.initialize_weights()
        
    def forward(self, input):
        '''
        forward(input)
        
        Required Arguments:
            - input (torch.Tensor): input data with dimensions [time x batch x cells]
        '''
        # tic = time.time()

        # Initialize hidden states
        g_encoder_state, c_encoder_state, controller_state = self.initialize_hidden_states(input) 

        
        # Encode input and calculate and calculate generator initial condition variational posterior distribution
        self.g_posterior_mean, self.g_posterior_logvar, out_gru_g_enc, out_gru_c_enc = self.encoder(input, (g_encoder_state, c_encoder_state))

        # Sample generator state
        generator_state = self.fc_genstate(self.sample_gaussian(self.g_posterior_mean, self.g_posterior_logvar))
        
        # Initialize factor state
        factor_state = self.generator.fc_factors(self.dropout(generator_state))
        
        # Factors store
        factors = torch.empty(0, self.batch_size, self.factor_size, device=self.device)
        
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # initialize generator input store
            gen_inputs = torch.empty(0, self.batch_size, self.u_latent_size, device=self.device)
            
            # initialize u posterior store
            self.u_posterior_mean   = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
            self.u_posterior_logvar = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
        
        # tic = time.time()
        
        # Controller and Generator Loop
        for t in range(self.steps_size):
            if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
                # Update controller state and calculate generator input variational posterior distribution
                u_mean, u_logvar, controller_state = self.controller(torch.cat((out_gru_c_enc[t], factor_state), dim=1), controller_state)
                
                # Append u_posterior mean and logvar
                self.u_posterior_mean = torch.cat((self.u_posterior_mean, u_mean.unsqueeze(1)), dim=1)
                self.u_posterior_logvar = torch.cat((self.u_posterior_logvar, u_logvar.unsqueeze(1)), dim=1)

                # Sample generator input
                generator_input = self.sample_gaussian(u_mean, u_logvar)
                # Append generator input to store
                gen_inputs  = torch.cat((gen_inputs, generator_input.unsqueeze(0)), dim=0)
            else:
                generator_input = torch.empty(self.batch_size, self.u_latent_size, device=self.device)
                gen_inputs = None
                
            # Update generator and factor state
            if self.attn:
                generator_state, factor_state = self.generator(generator_input, generator_state, input)
            else:
                generator_state, factor_state = self.generator(generator_input, generator_state)
            # Store factor state
            factors = torch.cat((factors, factor_state.unsqueeze(0)), dim=0)
            
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # Instantiate AR1 process as mean and variance per time step
            self.u_prior_mean, self.u_prior_logvar = self._gp_to_normal(self.u_prior_gp_mean, self.u_prior_gp_logvar, self.u_prior_gp_logtau, gen_inputs)
        
        return (factors, gen_inputs)

    def forward_all(self, input):
        '''
        forward(input)
        
        Required Arguments:
            - input (torch.Tensor): input data with dimensions [time x batch x cells]
        '''
        import time
        tic = time.time()

        # Initialize hidden states
        g_encoder_state, c_encoder_state, controller_state = self.initialize_hidden_states(input) 

        
        # Encode input and calculate and calculate generator initial condition variational posterior distribution
        self.g_posterior_mean, self.g_posterior_logvar, out_gru_g_enc, out_gru_c_enc = self.encoder(input, (g_encoder_state, c_encoder_state))
        

        # Sample generator state
        generator_state = self.fc_genstate(self.sample_gaussian(self.g_posterior_mean, self.g_posterior_logvar))
        
        # Initialize factor state
        factor_state = self.generator.fc_factors(self.dropout(generator_state))
        
        # generator store
        generators = torch.empty(0,self.batch_size, self.generator_size, device=self.device)

        # Factors store
        factors = torch.empty(0, self.batch_size, self.factor_size, device=self.device)
        
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # initialize generator input store
            gen_inputs = torch.empty(0, self.batch_size, self.u_latent_size, device=self.device)
            
            # initialize u posterior store
            self.u_posterior_mean   = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
            self.u_posterior_logvar = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
        
        tic = time.time()
        
        # Controller and Generator Loop
        for t in range(self.steps_size):
            if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
                # Update controller state and calculate generator input variational posterior distribution
                u_mean, u_logvar, controller_state = self.controller(torch.cat((out_gru_c_enc[t], factor_state), dim=1), controller_state)
                
                # Append u_posterior mean and logvar
                self.u_posterior_mean = torch.cat((self.u_posterior_mean, u_mean.unsqueeze(1)), dim=1)
                self.u_posterior_logvar = torch.cat((self.u_posterior_logvar, u_logvar.unsqueeze(1)), dim=1)

                # Sample generator input
                generator_input = self.sample_gaussian(u_mean, u_logvar)
                # Append generator input to store
                gen_inputs  = torch.cat((gen_inputs, generator_input.unsqueeze(0)), dim=0)
            else:
                generator_input = torch.empty(self.batch_size, self.u_latent_size, device=self.device)
                gen_inputs = None
                
            # Update generator and factor state
            generator_state, factor_state = self.generator(generator_input, generator_state)
            # Store factor state
            factors = torch.cat((factors, factor_state.unsqueeze(0)), dim=0)
            generators = torch.cat((generators, generator_state.unsqueeze(0)), dim=0)
            
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # Instantiate AR1 process as mean and variance per time step
            self.u_prior_mean, self.u_prior_logvar = self._gp_to_normal(self.u_prior_gp_mean, self.u_prior_gp_logvar, self.u_prior_gp_logtau, gen_inputs)
        
        return (factors, generators, gen_inputs)
    
    def sample_gaussian(self, mean, logvar):
        '''
        sample_gaussian(mean, logvar)
        
        Sample from a diagonal gaussian with given mean and log-variance
        
        Required Arguments:
            - mean (torch.Tensor)   : mean of diagional gaussian
            - logvar (torch.Tensor) : log-variance of diagonal gaussian
        '''
        # Generate noise from standard gaussian
        eps = torch.randn(mean.shape, requires_grad=False, dtype=torch.float32).to(torch.get_default_dtype()).to(self.device)
        # Scale and shift by mean and standard deviation
        return torch.exp(logvar*0.5)*eps + mean
    
    def initialize_hidden_states(self, input):
        '''
        initialize_hidden_states()
        
        Initialize hidden states of recurrent networks
        '''
        
        # Check dimensions
        self.steps_size, self.batch_size, input_size = input.shape
        assert input_size == self.input_size, 'Input is expected to have dimensions [%i, %i, %i]'%(self.steps_size, self.batch_size, self.input_size)
        
        g_encoder_state  = (torch.ones(self.batch_size, 2,  self.g_encoder_size, device=self.device) * self.g_encoder_init).permute(1, 0, 2)
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            c_encoder_state  = (torch.ones(self.batch_size, 2,  self.c_encoder_size, device=self.device) * self.c_encoder_init).permute(1, 0, 2)
            controller_state = torch.ones(self.batch_size, self.controller_size, device=self.device) * self.controller_init
            return g_encoder_state, c_encoder_state, controller_state
        else:
            return g_encoder_state, None, None
    
    def _gp_to_normal(self, gp_mean, gp_logvar, gp_logtau, process):
        '''
        _gp_to_normal(gp_mean, gp_logvar, gp_logtau, process)
        
        Convert gaussian process with given process mean, process log-variance, process tau, and realized process
        to mean and log-variance of diagonal Gaussian for each time-step
        '''
        
        mean   = gp_mean * torch.ones(1, process.shape[1], process.shape[2], device=self.device)
        logvar = gp_logvar * torch.ones(1, process.shape[1], process.shape[2], device=self.device)
        
        mean   = torch.cat((mean, gp_mean + (process[:-1] - gp_mean) * torch.exp(-1/gp_logtau.exp())))
        logvar = torch.cat((logvar, torch.log(1 - torch.exp(-1/gp_logtau.exp()).pow(2)) + gp_logvar * torch.ones(process.shape[0]-1, process.shape[1], process.shape[2], device=self.device)))
        return mean.permute(1, 0, 2), logvar.permute(1, 0, 2)
    
    def initialize_weights(self):
        '''
        initialize_weights()
        
        Initialize weights of network
        '''
        
        def standard_init(weights):
            k = weights.shape[1] # dimensionality of inputs
            weights.data.normal_(std=k**-0.5) # inplace resetting W ~ N(0, 1/sqrt(K))
        
        with torch.no_grad():
            for name, p in self.named_parameters():
                if 'weight' in name:
                    standard_init(p)

            if self.do_normalize_factors:
                self.normalize_factors()
     
    def normalize_factors(self):
        self.generator.fc_factors.weight.data = F.normalize(self.generator.fc_factors.weight.data, dim=1)
    
    def change_parameter_grad_status(self, step, optimizer, scheduler, loading_checkpoint=False):
        return optimizer, scheduler
    
    def kl_div(self):
        kl = kldiv_gaussian_gaussian(post_mu  = self.g_posterior_mean,
                                     post_lv  = self.g_posterior_logvar,
                                     prior_mu = self.g_prior_mean,
                                     prior_lv = self.g_prior_logvar)
        if self.u_latent_size > 0:
            kl += kldiv_gaussian_gaussian(post_mu  = self.u_posterior_mean,
                                          post_lv  = self.u_posterior_logvar,
                                          prior_mu = self.u_prior_mean,
                                          prior_lv = self.u_prior_logvar)
        return kl

class LFADS_coRNN_Net(nn.Module):
    '''
    LFADS_Net (Latent Factor Analysis via Dynamical Systems) neural network class.
    
    __init__(self, input_size, factor_size = 4,
                   g_encoder_size = 64, c_encoder_size = 64,
                   g_latent_size = 64, u_latent_size = 1,
                   controller_size= 64, generator_size = 64,
                   prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                    'var'  : {'value': 0.1, 'learnable' : False}},
                             'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                    'var'  : {'value': 0.1, 'learnable' : True},
                                     'tau'  : {'value': 10,  'learnable' : True}}},
                   clip_val=5.0, dropout=0.0, max_norm = 200,
                   do_normalize_factors=True, device='cpu')
                   
    Required Arguments:
        - input_size (int) : size of input dimensions (number of cells)
    Optional Arguments:
        - g_encoder_size     (int): size of generator encoder network
        - c_encoder_size     (int): size of controller encoder network
        - g_latent_size      (int): size of generator ic latent variable
        - u_latent_size      (int): size of generator input latent variable
        - controller_size    (int): size of controller network
        - generator_size     (int): size of generator network
        - prior             (dict): dictionary of prior distribution parameters
        - clip_val         (float): RNN hidden state value limit
        - dropout          (float): dropout probability
        - max_norm           (int): maximum gradient norm
        - do_normalize_factors (bool): whether to normalize factors
        - device          (string): device to use
    '''
    
    def __init__(self, input_size, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):
        
        super(LFADS_coRNN_Net, self).__init__()
        
        self.input_size           = input_size
#         self.output_size          = input_size if output_size is None else output_size
        self.g_encoder_size       = g_encoder_size
        self.c_encoder_size       = c_encoder_size
        self.g_latent_size        = g_latent_size
        self.u_latent_size        = u_latent_size
        self.controller_size      = controller_size
        self.generator_size       = generator_size
        self.factor_size          = factor_size
        
        self.clip_val             = clip_val
        self.max_norm             = max_norm
        self.do_normalize_factors = do_normalize_factors
        self.factor_bias          = factor_bias
        self.device               = device
        self.deep_freeze          = deep_freeze
        
        self.dropout              = torch.nn.Dropout(dropout)

        # Initialize encoder RNN
        self.encoder     = LFADS_Encoder(input_size     = self.input_size,
                                         g_encoder_size = self.g_encoder_size,
                                         c_encoder_size = self.c_encoder_size,
                                         g_latent_size  = self.g_latent_size,
                                         clip_val       = self.clip_val,
                                         dropout        = dropout)
        
        # Initialize controller RNN
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.controller  = LFADS_ControllerCell(input_size      = self.c_encoder_size*2 + self.factor_size,
                                                    controller_size = self.controller_size,
                                                    u_latent_size   = self.u_latent_size,
                                                    clip_val        = self.clip_val,
                                                    dropout         = dropout)
        
        # Initialize generator RNN
        self.generator   = LFADS_CoRNN_GeneratorCell(input_size     = self.u_latent_size,
                                            generator_size = self.generator_size,
                                            factor_size    = self.factor_size,
                                            clip_val       = self.clip_val,
                                            factor_bias    = self.factor_bias,
                                            dropout        = dropout)
        
        
        # Initialize dense layers
        if self.g_latent_size == self.generator_size:
            self.fc_genstate = Identity(in_features=self.g_latent_size, out_features=self.generator_size)
        else:
            self.fc_genstate = nn.Linear(in_features= self.g_latent_size, out_features= self.generator_size)
                
        # Initialize learnable biases
        self.g_encoder_init  = nn.Parameter(torch.zeros(2, self.g_encoder_size))
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.c_encoder_init  = nn.Parameter(torch.zeros(2, self.c_encoder_size))
            self.controller_init = nn.Parameter(torch.zeros(self.controller_size))
        
        # Initialize priors
        self.register_buffer('g_prior_mean',None)
        self.register_buffer('g_prior_logvar',None)
        self.register_buffer('g_posterior_mean',None)
        self.register_buffer('g_posterior_logvar',None) # these become large arrays and should not be singleton values.
        
        self.g_prior_mean = torch.ones(self.g_latent_size, device=device) * prior['g0']['mean']['value']
        
        if prior['g0']['mean']['learnable']:
            self.g_prior_mean = nn.Parameter(self.g_prior_mean)
        self.g_prior_logvar = torch.ones(self.g_latent_size, device=device) * log(prior['g0']['var']['value'])
        if prior['g0']['var']['learnable']:
            self.g_prior_logvar = nn.Parameter(self.g_prior_logvar)
            
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.u_prior_gp_mean = torch.ones(self.u_latent_size, device=device) * prior['u']['mean']['value']
            if prior['u']['mean']['learnable']:
                self.u_prior_gp_mean = nn.Parameter(self.u_prior_gp_mean)
            self.u_prior_gp_logvar = torch.ones(self.u_latent_size, device=device) * log(prior['u']['var']['value'])
            if prior['u']['var']['learnable']:
                self.u_prior_gp_logvar = nn.Parameter(self.u_prior_gp_logvar)
            self.u_prior_gp_logtau = torch.ones(self.u_latent_size, device=device) * log(prior['u']['tau']['value'])
            if prior['u']['tau']['learnable']:
                self.u_prior_gp_logtau = nn.Parameter(self.u_prior_gp_logtau)
        
        # Initialize weights
        self.initialize_weights()
        
    def forward(self, input):
        '''
        forward(input)
        
        Required Arguments:
            - input (torch.Tensor): input data with dimensions [time x batch x cells]
        '''
        # tic = time.time()

        # Initialize hidden states
        g_encoder_state, c_encoder_state, controller_state = self.initialize_hidden_states(input) 

        
        # Encode input and calculate and calculate generator initial condition variational posterior distribution
        self.g_posterior_mean, self.g_posterior_logvar, out_gru_g_enc, out_gru_c_enc = self.encoder(input, (g_encoder_state, c_encoder_state))

        # Sample generator state
        generator_state = self.fc_genstate(self.sample_gaussian(self.g_posterior_mean, self.g_posterior_logvar))
        
        # Initialize factor state
        factor_state = self.generator.fc_factors(self.dropout(generator_state))
        
        # Factors store
        factors = torch.empty(0, self.batch_size, self.factor_size, device=self.device)
        
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # initialize generator input store
            gen_inputs = torch.empty(0, self.batch_size, self.u_latent_size, device=self.device)
            
            # initialize u posterior store
            self.u_posterior_mean   = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
            self.u_posterior_logvar = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
        
        # tic = time.time()
        
        # Controller and Generator Loop
        for t in range(self.steps_size):
            if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
                # Update controller state and calculate generator input variational posterior distribution
                u_mean, u_logvar, controller_state = self.controller(torch.cat((out_gru_c_enc[t], factor_state), dim=1), controller_state)
                
                # Append u_posterior mean and logvar
                self.u_posterior_mean = torch.cat((self.u_posterior_mean, u_mean.unsqueeze(1)), dim=1)
                self.u_posterior_logvar = torch.cat((self.u_posterior_logvar, u_logvar.unsqueeze(1)), dim=1)

                # Sample generator input
                generator_input = self.sample_gaussian(u_mean, u_logvar)
                # Append generator input to store
                gen_inputs  = torch.cat((gen_inputs, generator_input.unsqueeze(0)), dim=0)
            else:
                generator_input = torch.empty(self.batch_size, self.u_latent_size, device=self.device)
                gen_inputs = None
                
            # Update generator and factor state
            if self.attn:
                generator_state, factor_state = self.generator(generator_input, generator_state, input)
            else:
                generator_state, factor_state = self.generator(generator_input, generator_state)
            # Store factor state
            factors = torch.cat((factors, factor_state.unsqueeze(0)), dim=0)
            
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # Instantiate AR1 process as mean and variance per time step
            self.u_prior_mean, self.u_prior_logvar = self._gp_to_normal(self.u_prior_gp_mean, self.u_prior_gp_logvar, self.u_prior_gp_logtau, gen_inputs)
        
        return (factors, gen_inputs)

    def forward_all(self, input):
        '''
        forward(input)
        
        Required Arguments:
            - input (torch.Tensor): input data with dimensions [time x batch x cells]
        '''
        import time
        tic = time.time()

        # Initialize hidden states
        g_encoder_state, c_encoder_state, controller_state = self.initialize_hidden_states(input) 

        
        # Encode input and calculate and calculate generator initial condition variational posterior distribution
        self.g_posterior_mean, self.g_posterior_logvar, out_gru_g_enc, out_gru_c_enc = self.encoder(input, (g_encoder_state, c_encoder_state))
        

        # Sample generator state
        generator_state = self.fc_genstate(self.sample_gaussian(self.g_posterior_mean, self.g_posterior_logvar))
        
        # Initialize factor state
        factor_state = self.generator.fc_factors(self.dropout(generator_state))
        
        # generator store
        generators = torch.empty(0,self.batch_size, self.generator_size, device=self.device)

        # Factors store
        factors = torch.empty(0, self.batch_size, self.factor_size, device=self.device)
        
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # initialize generator input store
            gen_inputs = torch.empty(0, self.batch_size, self.u_latent_size, device=self.device)
            
            # initialize u posterior store
            self.u_posterior_mean   = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
            self.u_posterior_logvar = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
        
        tic = time.time()
        
        # Controller and Generator Loop
        for t in range(self.steps_size):
            if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
                # Update controller state and calculate generator input variational posterior distribution
                u_mean, u_logvar, controller_state = self.controller(torch.cat((out_gru_c_enc[t], factor_state), dim=1), controller_state)
                
                # Append u_posterior mean and logvar
                self.u_posterior_mean = torch.cat((self.u_posterior_mean, u_mean.unsqueeze(1)), dim=1)
                self.u_posterior_logvar = torch.cat((self.u_posterior_logvar, u_logvar.unsqueeze(1)), dim=1)

                # Sample generator input
                generator_input = self.sample_gaussian(u_mean, u_logvar)
                # Append generator input to store
                gen_inputs  = torch.cat((gen_inputs, generator_input.unsqueeze(0)), dim=0)
            else:
                generator_input = torch.empty(self.batch_size, self.u_latent_size, device=self.device)
                gen_inputs = None
                
            # Update generator and factor state
            generator_state, factor_state = self.generator(generator_input, generator_state)
            # Store factor state
            factors = torch.cat((factors, factor_state.unsqueeze(0)), dim=0)
            generators = torch.cat((generators, generator_state.unsqueeze(0)), dim=0)
            
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # Instantiate AR1 process as mean and variance per time step
            self.u_prior_mean, self.u_prior_logvar = self._gp_to_normal(self.u_prior_gp_mean, self.u_prior_gp_logvar, self.u_prior_gp_logtau, gen_inputs)
        
        return (factors, generators, gen_inputs)
    
    def sample_gaussian(self, mean, logvar):
        '''
        sample_gaussian(mean, logvar)
        
        Sample from a diagonal gaussian with given mean and log-variance
        
        Required Arguments:
            - mean (torch.Tensor)   : mean of diagional gaussian
            - logvar (torch.Tensor) : log-variance of diagonal gaussian
        '''
        # Generate noise from standard gaussian
        eps = torch.randn(mean.shape, requires_grad=False, dtype=torch.float32).to(torch.get_default_dtype()).to(self.device)
        # Scale and shift by mean and standard deviation
        return torch.exp(logvar*0.5)*eps + mean
    
    def initialize_hidden_states(self, input):
        '''
        initialize_hidden_states()
        
        Initialize hidden states of recurrent networks
        '''
        
        # Check dimensions
        self.steps_size, self.batch_size, input_size = input.shape
        assert input_size == self.input_size, 'Input is expected to have dimensions [%i, %i, %i]'%(self.steps_size, self.batch_size, self.input_size)
        
        g_encoder_state  = (torch.ones(self.batch_size, 2,  self.g_encoder_size, device=self.device) * self.g_encoder_init).permute(1, 0, 2)
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            c_encoder_state  = (torch.ones(self.batch_size, 2,  self.c_encoder_size, device=self.device) * self.c_encoder_init).permute(1, 0, 2)
            controller_state = torch.ones(self.batch_size, self.controller_size, device=self.device) * self.controller_init
            return g_encoder_state, c_encoder_state, controller_state
        else:
            return g_encoder_state, None, None
    
    def _gp_to_normal(self, gp_mean, gp_logvar, gp_logtau, process):
        '''
        _gp_to_normal(gp_mean, gp_logvar, gp_logtau, process)
        
        Convert gaussian process with given process mean, process log-variance, process tau, and realized process
        to mean and log-variance of diagonal Gaussian for each time-step
        '''
        
        mean   = gp_mean * torch.ones(1, process.shape[1], process.shape[2], device=self.device)
        logvar = gp_logvar * torch.ones(1, process.shape[1], process.shape[2], device=self.device)
        
        mean   = torch.cat((mean, gp_mean + (process[:-1] - gp_mean) * torch.exp(-1/gp_logtau.exp())))
        logvar = torch.cat((logvar, torch.log(1 - torch.exp(-1/gp_logtau.exp()).pow(2)) + gp_logvar * torch.ones(process.shape[0]-1, process.shape[1], process.shape[2], device=self.device)))
        return mean.permute(1, 0, 2), logvar.permute(1, 0, 2)
    
    def initialize_weights(self):
        '''
        initialize_weights()
        
        Initialize weights of network
        '''
        
        def standard_init(weights):
            k = weights.shape[1] # dimensionality of inputs
            weights.data.normal_(std=k**-0.5) # inplace resetting W ~ N(0, 1/sqrt(K))
        
        with torch.no_grad():
            for name, p in self.named_parameters():
                if 'weight' in name:
                    standard_init(p)

            if self.do_normalize_factors:
                self.normalize_factors()
     
    def normalize_factors(self):
        self.generator.fc_factors.weight.data = F.normalize(self.generator.fc_factors.weight.data, dim=1)
    
    def change_parameter_grad_status(self, step, optimizer, scheduler, loading_checkpoint=False):
        return optimizer, scheduler
    
    def kl_div(self):
        kl = kldiv_gaussian_gaussian(post_mu  = self.g_posterior_mean,
                                     post_lv  = self.g_posterior_logvar,
                                     prior_mu = self.g_prior_mean,
                                     prior_lv = self.g_prior_logvar)
        if self.u_latent_size > 0:
            kl += kldiv_gaussian_gaussian(post_mu  = self.u_posterior_mean,
                                          post_lv  = self.u_posterior_logvar,
                                          prior_mu = self.u_prior_mean,
                                          prior_lv = self.u_prior_logvar)
        return kl

class LFADS_Multiblock_Net(nn.Module):
    '''
        Similar to the LFADS_Net, but implementing multiple encoder/generator pair RNN cells in parallel for multiband processing.
    '''
    def __init__(self,input_size, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 n_block = 2,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):

        super(LFADS_Multiblock_Net, self).__init__()

        self.max_norm               = max_norm
        self.do_normalize_factors   = do_normalize_factors
        self.device                 = device

        # create n_block different LFADS_Net objects, define the forward pass to give a different input to each.
        lfads_blocks = []
        for block_idx in range(n_block):
            lfads_blocks.append(
                LFADS_Ecog_SingleSession_Net(
                    input_size=input_size, 
                    factor_size=factor_size,
                    g_encoder_size=g_encoder_size, 
                    c_encoder_size=c_encoder_size,
                    g_latent_size=g_latent_size, 
                    u_latent_size=u_latent_size,
                    controller_size=controller_size, 
                    generator_size=generator_size,
                    prior=prior,
                    clip_val=clip_val, 
                    dropout=dropout, 
                    max_norm=max_norm, 
                    deep_freeze=deep_freeze,
                    do_normalize_factors=do_normalize_factors, 
                    factor_bias=factor_bias, 
                    device=device
                )
            )
        self.lfads_blocks = nn.ModuleList(lfads_blocks)
        # create linear net to mix LFADS block outputs
        # self.out_mix = nn.Linear(in_features = n_block * input_size, out_features = input_size, bias=False)
        self.out_mix = nn.Linear(in_features=n_block, out_features=1, bias=False) # stack block outputs in a new dimension then mix along that axis then squeeze.

    def forward(self,src):
        assert len(src) == len(self.lfads_blocks), f'input sample must be a list-like of length equal to the number of model blocks. {len(src)} sample elements found, {len(self.lfads_blocks)} expected.'
        block_outputs = []
        for idx, lb in enumerate(self.lfads_blocks):
            _recon, _ = lb(src[idx])
            block_outputs.append(_recon['data'])
        block_outputs = torch.stack(block_outputs,dim=-1) # too much data to hold onto, overwriting
        pred = self.out_mix(block_outputs)
        pred = pred.squeeze(dim=-1)
        recon = {}
        recon['data'] = pred
        # omit factor returns for now, return block_outputs later.
        return recon, block_outputs # consider routing the latent states here instead of the block outputs

    def forward_all(self,src):
        assert len(src) == len(self.lfads_blocks), f'input sample must be a list-like of length equal to the number of model blocks. {len(src)} sample elements found, {len(self.lfads_blocks)} expected.'
        block_outputs = []
        block_gen = []
        for idx, lb in enumerate(self.lfads_blocks):
            _recon, (_factors, _gen, _gen_ic) = lb.forward_all(src[idx])
            block_outputs.append(_recon['data'])
            block_gen.append(_gen)
        block_outputs = torch.stack(block_outputs,dim=-1) # too much data to hold onto, overwriting
        pred = self.out_mix(block_outputs)
        pred = pred.squeeze(dim=-1)
        recon = {}
        recon['data'] = pred
        # omit factor returns for now, return block_outputs later.
        return recon, (None, block_outputs, block_gen) # I've got to find a better way to pack these elements together. Dict?

    def normalize_factors(self):
        for lb in self.lfads_blocks:
            lb.generator.fc_factors.weight.data = F.normalize(lb.generator.fc_factors.weight.data, dim=1)

    # this does nothing, I don't know why it's here. If the optimizer were to change, this would make sense, but it's just an ID transform. Odd.
    def change_parameter_grad_status(self, step, optimizer, scheduler, loading_checkpoint=False):
        return optimizer, scheduler

    
class LFADS_SingleSession_Net(LFADS_Net):
    
    def __init__(self, input_size, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):
        
        super(LFADS_SingleSession_Net, self).__init__(input_size = input_size, factor_size = factor_size, prior = prior,
                                                      g_encoder_size   = g_encoder_size, c_encoder_size = c_encoder_size,
                                                      g_latent_size    = g_latent_size, u_latent_size = u_latent_size,
                                                      controller_size  = controller_size, generator_size = generator_size,
                                                      clip_val=clip_val, dropout=dropout, max_norm = max_norm, deep_freeze = deep_freeze,
                                                      do_normalize_factors=do_normalize_factors, factor_bias = factor_bias, device=device)
        
        self.fc_logrates = nn.Linear(in_features= self.factor_size, out_features= self.input_size)
        
        self.initialize_weights()
        
    def forward(self, input):
        factors, gen_inputs = super(LFADS_SingleSession_Net, self).forward(input.permute(1, 0, 2))
        recon = {'rates' : self.fc_logrates(factors).exp()}
        recon['data'] = recon['rates'].clone().permute(1, 0, 2)
        return recon, (factors, gen_inputs)

    def forward_all(self, input):
        factors, generators, gen_inputs = super(LFADS_SingleSession_Net, self).forward_all(input.permute(1, 0, 2))
        recon = {'rates' : self.fc_logrates(factors).exp()}
        recon['data'] = recon['rates'].clone().permute(1, 0, 2)
        return recon, (factors, generators, gen_inputs)

class LFADS_Ecog_SingleSession_Net(LFADS_Net):

    def __init__(self, input_size, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 attention = False,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):

        super(LFADS_Ecog_SingleSession_Net, self).__init__(input_size = input_size, factor_size = factor_size, prior = prior,
                                                      g_encoder_size   = g_encoder_size, c_encoder_size = c_encoder_size,
                                                      g_latent_size    = g_latent_size, u_latent_size = u_latent_size,
                                                      controller_size  = controller_size, generator_size = generator_size,
                                                      clip_val=clip_val, dropout=dropout, max_norm = max_norm, deep_freeze = deep_freeze,
                                                      do_normalize_factors=do_normalize_factors, factor_bias = factor_bias, attn=attention, device=device)
        
        self.fc_logrates = nn.Linear(in_features= self.factor_size, out_features= self.input_size)
        
        self.initialize_weights()
        
    def forward(self, input):
        factors, gen_inputs = super(LFADS_Ecog_SingleSession_Net, self).forward(input.permute(1, 0, 2))
        recon = {'rates' : self.fc_logrates(factors)}
        recon['data'] = recon['rates'].clone().permute(1, 0, 2)
        return recon, (factors, gen_inputs)

    def forward_all(self, input):
        factors, generators, gen_inputs = super(LFADS_Ecog_SingleSession_Net, self).forward_all(input.permute(1, 0, 2))
        recon = {'rates' : self.fc_logrates(factors)}
        recon['data'] = recon['rates'].clone().permute(1, 0, 2)
        return recon, (factors, generators, gen_inputs)

class LFADS_Ecog_CoRNN_Net(LFADS_coRNN_Net):

    def __init__(self, input_size, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):
        
        super(LFADS_Ecog_CoRNN_Net, self).__init__(
            input_size=input_size, factor_size = factor_size, prior = prior,
            g_encoder_size   = g_encoder_size, c_encoder_size = c_encoder_size,
            g_latent_size    = g_latent_size, u_latent_size = u_latent_size,
            controller_size  = controller_size, generator_size = generator_size,
            clip_val=clip_val, dropout=dropout, max_norm = max_norm, deep_freeze = deep_freeze,
            do_normalize_factors=do_normalize_factors, factor_bias = factor_bias, device=device
        )
    
        self.initialize_weights()

    def forward(self, input):
        factors, gen_inputs = super(LFADS_Ecog_CoRNN_Net, self).forward(input.perumte(1, 0, 2))
        recon = {
            'rates':    factors,
            'data':     factors.clone().permute(1,0,2)
        }
        return recon, (factors, gen_inputs)

    def forward_all(self, input):
        factors, generators, gen_inputs = super(LFADS_Ecog_CoRNN_Net, self).forward(input.perumte(1, 0, 2))
        recon = {
            'rates':    factors,
            'data':     factors.clone().permute(1,0,2)
        }
        return recon, (factors, generators, gen_inputs)
    
class LFADS_MultiSession_Net(LFADS_Net): # for aligning multiple subjects/recordings where things drift.
    
    def __init__(self, W_in_list, W_out_list, b_in_list, b_out_list, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):
        
        super(LFADS_MultiSession_Net, self).__init__(input_size = factor_size, factor_size = factor_size, prior = prior,
                                                     g_encoder_size   = g_encoder_size, c_encoder_size = c_encoder_size,
                                                     g_latent_size    = g_latent_size, u_latent_size = u_latent_size,
                                                     controller_size  = controller_size, generator_size = generator_size,
                                                     clip_val=clip_val, dropout=dropout, max_norm = max_norm, deep_freeze = deep_freeze,
                                                     do_normalize_factors=do_normalize_factors, factor_bias = factor_bias, device=device)
        
        for idx, (W_in, b_in, W_out, b_out) in enumerate(zip(W_in_list, b_in_list, W_out_list, b_out_list)):
            assert W_in.shape[1] == self.factor_size, 'Read in matrix should have dim 1 = %i, but has dims [%i, %i]'%(self.factor_size, W_in.shape[0], W_in.shape[1])
            assert W_out.shape[0] == self.factor_size, 'Read out matrix should have dim 0 = %i, but has dims [%i, %i]'%(self.factor_size, W_out.shape[0], W_out.shape[1])
            setattr(self, 'fc_input_%i'%idx, nn.Linear(in_features=W_in.shape[0], out_features=self.factor_size))
            setattr(self, 'fc_logrates_%i'%idx, nn.Linear(in_features=self.factor_size, out_features=W_in.shape[0]))
            
            getattr(self, 'fc_input_%i'%idx).weight.data = W_in.permute(1, 0)
            getattr(self, 'fc_input_%i'%idx).bias.data = b_in
            getattr(self, 'fc_logrates_%i'%idx).weight.data = W_out.permute(1, 0)
            getattr(self, 'fc_logrates_%i'%idx).bias.data = b_out
            
#             getattr(self, 'fc_input_%i'%idx).weight.requires_grad = False
#             getattr(self, 'fc_logrates_%i'%idx).weight.requires_grad = False
#             getattr(self, 'fc_input_%i'%idx).bias.requires_grad = False
#             getattr(self, 'fc_logrates_%i'%idx).bias.requires_grad = False
            
    def forward(self, input):
        aligned_input = getattr(self, 'fc_input_%i'%input.session)(input).permute(1, 0, 2)
        factors, gen_inputs = super(LFADS_MultiSession_Net, self).forward(aligned_input)
        recon = {'rates' : getattr(self, 'fc_logrates_%i'%input.session)(factors).exp().permute(1, 0, 2)}
        recon['data'] = recon['rates'].clone()
        return recon, (factors, gen_inputs)
        
    
class LFADS_Encoder(nn.Module):
    '''
    LFADS_Encoder
    
    LFADS Encoder Network 
    
    __init__(self, input_size, g_encoder_size, g_latent_size, c_encoder_size= 0, dropout= 0.0, clip_val= 5.0)
    
    Required Arguments:
        - input_size (int):  size of input dimensions
        - g_encoder_size (int):  size of generator encoder network
        - g_latent_size (int): size of generator ic latent variable
        
    Optional Arguments:
        - c_encoder_size (int): size of controller encoder network
        - dropout (float): dropout probability
        - clip_val (float): RNN hidden state value limit
        
    '''
    def __init__(self, input_size, g_encoder_size, g_latent_size, c_encoder_size= 0, dropout= 0.0, clip_val= 5.0):
        super(LFADS_Encoder, self).__init__()
        self.input_size      = input_size
        self.g_encoder_size  = g_encoder_size
        self.c_encoder_size  = c_encoder_size
        self.g_latent_size   = g_latent_size
        self.clip_val        = clip_val

        self.dropout = nn.Dropout(dropout)
        
        # g Encoder BiRNN
        self.gru_g_encoder  = nn.GRU(input_size=self.input_size, hidden_size=self.g_encoder_size, bidirectional=True)
        # g Linear mapping
        self.fc_g0_theta    = nn.Linear(in_features= 2 * self.g_encoder_size, out_features= self.g_latent_size * 2)
        
        if self.c_encoder_size > 0:
            # c encoder BiRNN
            self.gru_c_encoder  = nn.GRU(input_size=self.input_size, hidden_size=self.c_encoder_size, bidirectional=True)
            
    def forward(self, input, hidden):
        self.gru_g_encoder.flatten_parameters()
        if self.c_encoder_size > 0:
            self.gru_c_encoder.flatten_parameters()
        gru_g_encoder_init, gru_c_encoder_init = hidden
        
        # Run bidirectional RNN over data
        out_gru_g_enc, hidden_gru_g_enc = self.gru_g_encoder(self.dropout(input), gru_g_encoder_init.contiguous())
        hidden_gru_g_enc = self.dropout(hidden_gru_g_enc.clamp(min=-self.clip_val, max=self.clip_val))
        hidden_gru_g_enc = torch.cat((hidden_gru_g_enc[0], hidden_gru_g_enc[1]), dim=1)
        
        g0_mean, g0_logvar = torch.split(self.fc_g0_theta(hidden_gru_g_enc), self.g_latent_size, dim=1)
        
        if self.c_encoder_size > 0:
            out_gru_c_enc, hidden_gru_c_enc = self.gru_c_encoder(self.dropout(input), gru_c_encoder_init.contiguous())
            out_gru_c_enc = out_gru_c_enc.clamp(min=-self.clip_val, max=self.clip_val)
        
            return g0_mean, g0_logvar, out_gru_g_enc, out_gru_c_enc
        
        else:
            
            return g0_mean, g0_logvar, out_gru_g_enc, None
        
class LFADS_ControllerCell(nn.Module):
    
    def __init__(self, input_size, controller_size, u_latent_size, dropout = 0.0, clip_val=5.0):
        super(LFADS_ControllerCell, self).__init__()
        self.input_size      = input_size
        self.controller_size = controller_size
        self.u_latent_size   = u_latent_size
        self.clip_val        = clip_val
        
        self.dropout = nn.Dropout(dropout)
        
        self.gru_controller  = LFADS_GenGRUCell(input_size  = self.input_size, hidden_size = self.controller_size)
        self.fc_u_theta = nn.Linear(in_features = self.controller_size, out_features=self.u_latent_size * 2)
        
    def forward(self, input, hidden):
        controller_state = hidden
        controller_state = self.gru_controller(self.dropout(input), controller_state)
        controller_state = controller_state.clamp(-self.clip_val, self.clip_val)
        u_mean, u_logvar = torch.split(self.fc_u_theta(controller_state), self.u_latent_size, dim=1)
        return u_mean, u_logvar, controller_state
    
class LFADS_GeneratorCell(nn.Module):
    
    def __init__(self, input_size, generator_size, factor_size, dropout = 0.0, clip_val = 5.0, factor_bias = False):
        super(LFADS_GeneratorCell, self).__init__()
        self.input_size = input_size
        self.generator_size = generator_size
        self.factor_size = factor_size
        
        self.dropout = nn.Dropout(dropout)
        self.clip_val = clip_val
        
        self.gru_generator = LFADS_GenGRUCell(input_size=input_size, hidden_size=generator_size)
        self.fc_factors = nn.Linear(in_features=generator_size, out_features=factor_size, bias=factor_bias)
        
    def forward(self, input, hidden):
        
        generator_state = hidden
        generator_state = self.gru_generator(input, generator_state)
        generator_state = generator_state.clamp(min=-self.clip_val, max=self.clip_val)
        factor_state    = self.fc_factors(self.dropout(generator_state))
        
        return generator_state, factor_state

from coRNN import coRNN
class LFADS_CoRNN_GeneratorCell(nn.Module):

    def __init__(self,input_size, generator_size, factor_size, dropout=0.0, clip_val=5.0, factor_bias=False):
        super(LFADS_CoRNN_GeneratorCell, self).__init__()
        self.input_size = input_size
        self.generator_size = generator_size
        self.factor_size = factor_size

        self.dropout = dropout
        self.clip_val = clip_val

        self.cornn_generator = coRNN(
            input_size = input_size,
            hidden_size = generator_size,
            output_size = factor_size,
            n_layer = 1,
            dt = 1, # how important is this? not routed through though (yet)
        )
        None

    def forward(self,input,hidden):
        generator_state = hidden
        generator_state = self.cornn_generator(input,generator_state)
        # the coRNN generator is inherently bound, so this shouldn't be necessary, but why not (remove if shown true)
        factor_state, generator_state = generator_state.clamp(min=-self.clip_val,max=self.clip_val)
        return generator_state, factor_state

class LFADS_Generator_SourceAttnCell(nn.Module):

    def __init__(self, input_size, generator_size, factor_size, source_size, dropout = 0.0, clip_val = 5.0, factor_bias = False):
        super(LFADS_Generator_SourceAttnCell, self).__init__()
        self.input_size = input_size
        self.generator_size = generator_size
        self.factor_size = factor_size
        self.dropout = nn.Dropout(dropout)
        self.clip_val = clip_val
        self.gru_generator = LFADS_GenGRUCell(input_size=input_size, hidden_size=generator_size)
        self.fc_factors = nn.Linear(in_features=generator_size, out_features=factor_size, bias=factor_bias)
        self.attn_alpha = nn.Linear(in_features=generator_size+source_size, out_features=source_size, bias=False) # the new thing
        self.attn_out = nn.Linear(in_features=generator_size+source_size, out_features=generator_size, bias=False)

    def forward(self, input, hidden, src):
        '''
            One pass through the generator loop.
        '''
        # # attention mapping
        # attn_weights = F.softmax(self.attn(torch.cat((input,hidden),1)),dim=1)
        # breakpoint()
        # attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # input = nn.Softmax(self.attn_combine(torch.cat((input[0],attn_applied[0]), 1)).unsqueeze(0))

        generator_state = hidden

        generator_state = self.gru_generator(input, generator_state).clamp(min=-self.clip_val, max=self.clip_val)
        # concatenate generator output to hidden states, project to obtain attention weights
        # attn_w = torch.zeros((encoder_outputs.shape[0],encoder_outputs.shape[1])).to(generator_state.device) # time x batch
        # for idx in range(encoder_outputs.shape[0]): # slow way, dumb way. Figure out how to broadcast this
        #     attn_w[idx] = torch.relu(self.attn_alpha(torch.cat((generator_state,encoder_outputs.permute(1,0,2)[:,idx,:]),dim=-1))).squeeze()
        attn_w = torch.relu(self.attn_alpha(
            torch.cat(
                (
                generator_state.unsqueeze(1).repeat(1,src.shape[0],1), # repeat the generator output and tack it onto the encoder sequence at each point
                src.permute(1,0,2)
                ),
                dim=-1)
            ))
        attn_w = torch.softmax(attn_w,dim=1) # softmax across time
        # take weighted sum of encoder states, augment generator state (query), output to generator_state size
        attn_out = torch.tanh(self.attn_out(torch.cat((generator_state,(src.permute(1,0,2)*attn_w).sum(dim=1)),dim=-1)))
        factor_state    = self.fc_factors(self.dropout(attn_out))

        return generator_state, factor_state

class LFADS_Generator_EncoderAttnCell(nn.Module):

    def __init__(self, input_size, generator_size, factor_size, encoder_size, dropout = 0.0, clip_val = 5.0, factor_bias = False):
        super(LFADS_Generator_EncoderAttnCell, self).__init__()
        self.input_size = input_size
        self.generator_size = generator_size
        self.factor_size = factor_size
        self.dropout = nn.Dropout(dropout)
        self.clip_val = clip_val
        self.gru_generator = LFADS_GenGRUCell(input_size=input_size, hidden_size=generator_size)
        self.fc_factors = nn.Linear(in_features=generator_size, out_features=factor_size, bias=factor_bias)
        self.attn_alpha = nn.Linear(in_features=generator_size+2*encoder_size, out_features=encoder_size*2, bias=False) # the new thing
        self.attn_out = nn.Linear(in_features=generator_size+2*encoder_size, out_features=generator_size, bias=False)

        # self.attn_combine = nn.Linear(in_features=input_size+generator_size,out_features=generator_size)

    def forward(self, input, hidden, encoder_outputs):
        '''
            One pass through the generator loop.
        '''
        # # attention mapping
        # attn_weights = F.softmax(self.attn(torch.cat((input,hidden),1)),dim=1)
        # breakpoint()
        # attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # input = nn.Softmax(self.attn_combine(torch.cat((input[0],attn_applied[0]), 1)).unsqueeze(0))

        generator_state = hidden

        generator_state = self.gru_generator(input, generator_state).clamp(min=-self.clip_val, max=self.clip_val)
        # concatenate generator output to hidden states, project to obtain attention weights
        # attn_w = torch.zeros((encoder_outputs.shape[0],encoder_outputs.shape[1])).to(generator_state.device) # time x batch
        # for idx in range(encoder_outputs.shape[0]): # slow way, dumb way. Figure out how to broadcast this
        #     attn_w[idx] = torch.relu(self.attn_alpha(torch.cat((generator_state,encoder_outputs.permute(1,0,2)[:,idx,:]),dim=-1))).squeeze()
        attn_w = torch.relu(self.attn_alpha(
            torch.cat(
                (
                generator_state.unsqueeze(1).repeat(1,encoder_outputs.shape[0],1), # repeat the generator output and tack it onto the encoder sequence at each point
                encoder_outputs.permute(1,0,2)
                ),
                dim=-1)
            ))
        attn_w = torch.softmax(attn_w,dim=1) # softmax across time, dim 1
        # take weighted sum of encoder states, augment generator state (query), output to generator_state size
        attn_out = torch.tanh(self.attn_out(torch.cat((generator_state,(encoder_outputs.permute(1,0,2)*attn_w).sum(dim=1)),dim=-1)))
        factor_state    = self.fc_factors(self.dropout(attn_out))

        return generator_state, factor_state
    
class Identity(nn.Module):
    def __init__(self, in_features, out_features):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x
