import torch
from torch import nn
from torch.autograd import Variable

# practice:
#
# Cell classes implement the single forward pass of the network block. Think of this as a very stripped-down ODE model.
# The RNN unit class then can wrap together several instances of the cell class, thread them together and create a n-layer unit.
# In short, the unit is several layers of cells, one cell per layer. Add a readout layer to the end of that and voila! An RNN.

# Coupled Oscillatory Recurrent Neural Network (coRNN) - Rusch, Mishra; ICLR 2021
class coRNNCell(nn.Module):

    def __init__(self,input_size,hidden_size,dt,gamma,epsilon):
        super(coRNNCell,self).__init__()
        self.dt         = dt
        self.gamma      = gamma
        self.epsilon    = epsilon
        
        self.y2z        = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=False)
        self.z2z        = nn.Linear(in_features=hidden_size,out_features=hidden_size,bias=False)
        self.u2z        = nn.Linear(in_features=input_size,out_features=hidden_size,bias=True)

    def forward(self,u,y,z):
        y_out = y + self.dt * z
        z_out = z + self.dt * (torch.tanh(self.y2z(y) + self.z2z(z) + self.u2z(u)) - self.gamma * y - self.epsilon * z)   # (3)

        return y_out, z_out

class coRNN(nn.Module):

    def __init__(self,input_size,hidden_size,output_size,n_layer,dt,gamma,epsilon):
        super(coRNN,self).__init__()

        self.hidden_size    = hidden_size
        self.n_layer        = n_layer
        cell = []
        for n in range(n_layer):
            if n == 0:
                in_size = input_size
            else:
                in_size = hidden_size
            cell.append(coRNNCell(input_size=in_size,hidden_size=hidden_size,dt=dt,gamma=gamma,epsilon=epsilon))
        self.cell       = nn.ModuleList(cell)
        self.readout    = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self,u):
        # init hidden states as zero, assuming batch-first
        y = Variable(torch.zeros(u.size(0),self.hidden_size,dtype=u.dtype)).to(u.device)
        z = Variable(torch.zeros(u.size(0),self.hidden_size,dtype=u.dtype)).to(u.device)

        for t in range(u.size(0)):
            y, z = self.cell[0](u[:,t,:],y,z)
        output = self.readout(y)

        return output

# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #
# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #

def main():
    '''
    Test script for coRNN model class.
    Not sure 
    '''
    # create model
    n_in = 10
    n_hid = 20
    n_out = 5

    dt = 0.01
    gam = 1.0
    eps = 1.0

    model = coRNN(
        input_size = n_in,
        hidden_size = n_hid,
        output_size = n_out,
        n_layer = 1,
        dt = dt,
        gamma = gam,
        epsilon = eps
    )

    # create sample input
    n_sample = 100 # 1s
    n_batch = 10
    input = torch.randn(n_batch,n_sample,n_in)

    # comput output 
    output = model(input)

    # print input, model, outputs
    print('model:')
    print(model)
    print(f'input:\t{input.shape}')
    print(input)
    print(f'output:\t{output.shape}')
    print(output)

    return None


if __name__ == '__main__':
    main()
