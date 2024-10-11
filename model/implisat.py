import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pdb
import torch.utils.checkpoint as cp

class FourierNetwork2DPhase(nn.Module):
    def __init__(self, n_hidden_dims: int, n_fourier_bases: int, n_layers: int, n_output_dims: int = None):
        super().__init__()

        self.n_hidden_dims = n_hidden_dims
        self.n_fourier_bases = n_fourier_bases
        self.n_layers = n_layers
        self.n_output_dims = n_output_dims if n_output_dims is not None else n_hidden_dims
        self.square_dim = int(self.n_fourier_bases ** (1/2))

        self.encoder = nn.Sequential(
            nn.Linear(14, self.n_hidden_dims // 4),
            nn.ReLU(),
            nn.Linear(self.n_hidden_dims // 4, self.n_hidden_dims // 2),
            nn.ReLU(),
            nn.Linear(self.n_hidden_dims // 2, self.n_hidden_dims),
        )

        self.decoder = []
        for _ in range(n_layers):
            self.decoder.append(nn.Linear(self.n_hidden_dims, self.n_fourier_bases * 2))
        self.decoder = nn.ModuleList(self.decoder)
        
        # Sampled points vector Z
        self.register_buffer('z_vector', torch.Tensor(np.linspace(-2 * math.pi, 2 * math.pi, self.square_dim)))
    
    def forward(self, channel: torch.Tensor, resolution: torch.Tensor):
        # channel (13, 13) one hot encoding of the channel
        # resolution (13, 1) resolution of the image, e.g. 0.1, 0.2, 0.6
        B = channel.shape[0]
        f_input = torch.cat([channel, resolution], dim=1) # (13, 14)
        f_output = self.encoder(f_input)

        freq_phase_output = []
        # Generate the frequency and phase for each layers
        for i in range(self.n_layers):
            freq_phase_output.append(F.tanh(self.decoder[i](f_output))) # (13, n_fourier_bases * 2)

        # each freq and phase is (13, square_dim, square_dim)
        freq_vec = []
        phase_vec = []
        for i in range(self.n_layers):
            freq_vec_i, phase_vec_i = freq_phase_output[i][:, :self.n_fourier_bases], freq_phase_output[i][:, self.n_fourier_bases:]
            freq_vec.append(freq_vec_i.unsqueeze(2).reshape(-1, self.square_dim, self.square_dim))
            phase_vec.append(phase_vec_i.unsqueeze(2).reshape(-1, self.square_dim, self.square_dim))
        
        z_matrix = self.z_vector.unsqueeze(0).unsqueeze(0).expand(B, self.square_dim, -1) # (13, square_dim, square_dim)
        
        fourier_mod = []
        for i in range(self.n_layers):
            fourier_mod_i = torch.cos(freq_vec[i] * z_matrix + phase_vec[i]) 
            fourier_mod.append(fourier_mod_i)

        fourier_mods = torch.stack(fourier_mod, dim=1)
        return fourier_mods # (13, n_layer, square_dim, square_dim)

class FourierNetwork1DPhase(FourierNetwork2DPhase):
    def __init__(self, n_hidden_dims: int, n_fourier_bases: int, n_layers: int, n_output_dims: int = None):
        super().__init__(n_hidden_dims, n_fourier_bases, n_layers, n_output_dims)
        del self.decoder
        self.encoder = nn.Sequential(
            nn.Linear(14, self.n_layers * (self.n_fourier_bases + int(self.n_fourier_bases **(1/2))))
        )
    
    def forward(self, channel: torch.Tensor, resolution: torch.Tensor):
        # channel (13, 13) one hot encoding of the channel
        # resolution (13, 1) resolution of the image, e.g. 0.1, 0.2, 0.6
        B = channel.shape[0]
        f_input = torch.cat([channel, resolution], dim=1) # (13, 14)
        f_output = self.encoder(f_input)

        freq_phase_output = []
        # Generate the frequency and phase for each layers
        for i in range(self.n_layers):
            freq_phase_output.append(F.tanh((f_output[:, i * (self.n_fourier_bases + self.square_dim): (i+1) * (self.n_fourier_bases + self.square_dim)])))

        # each freq and phase is (13, square_dim, square_dim)
        freq_vec = []
        phase_vec = []
        for i in range(self.n_layers):
            freq_vec_i, phase_vec_i = freq_phase_output[i][:, :self.n_fourier_bases], freq_phase_output[i][:, self.n_fourier_bases:]
            freq_vec.append(freq_vec_i.unsqueeze(2).reshape(-1, self.square_dim, self.square_dim))
            phase_vec.append(phase_vec_i.unsqueeze(2).expand(-1, self.square_dim, self.square_dim))
        
        z_matrix = self.z_vector.unsqueeze(0).unsqueeze(0).expand(B, self.square_dim, -1) # (13, square_dim, square_dim)
        
        fourier_mod = []
        for i in range(self.n_layers):
            # pdb.set_trace()
            fourier_mod_i = torch.cos(freq_vec[i] * z_matrix + phase_vec[i]) 
            fourier_mod.append(fourier_mod_i)

        fourier_mods = torch.stack(fourier_mod, dim=1)
        return fourier_mods # (13, n_layer, square_dim, square_dim)

class SinFRLayer(nn.Module):
    def __init__(
        self, 
        n_input_dims: int, 
        n_output_dims: int,
        square_dim: int = None, 
        bias: bool = True, 
        is_reparam: bool = False,
        is_first: bool = False, 
        is_linear: bool = False, 
        omega_0: float = 30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_linear = is_linear
        self.is_reparam = is_reparam
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.square_dim = square_dim

        if is_reparam:
            self.weights_alpha = nn.Parameter(torch.empty(n_input_dims, square_dim, requires_grad=True))
            self.weights_beta = nn.Parameter(torch.empty(square_dim, n_output_dims, requires_grad=True))
        else:
            self.weights = nn.Parameter(torch.empty(n_output_dims, n_input_dims, requires_grad=True))
            
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(n_output_dims, requires_grad=True))
            self.init_bias()

        self.init_weights()
    
    def init_bias(self):
        stdv = 1. / math.sqrt(self.n_input_dims)
        with torch.no_grad():
            self.bias.uniform_(-stdv, stdv)

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.weights.uniform_(-1 / self.n_input_dims, 1 / self.n_input_dims)      
            else:
                if self.is_reparam:
                    self.weights_alpha.uniform_(-np.sqrt(6 / self.n_input_dims) / self.omega_0, np.sqrt(6 / self.n_input_dims) / self.omega_0)
                    self.weights_beta.uniform_(-np.sqrt(6 / self.n_input_dims) / self.omega_0, np.sqrt(6 / self.n_input_dims) / self.omega_0)
                else:
                    self.weights.uniform_(-np.sqrt(6 / self.n_input_dims) / self.omega_0, np.sqrt(6 / self.n_input_dims) / self.omega_0)
        
    def forward(self, input: torch.Tensor, fourier_mod: torch.Tensor = None):
        # input (B, n_input_dims)
        # fourier_mod (square_dim, n_output_dims)
        if not self.is_reparam:
            out = F.linear(input, self.weights, self.bias) # (B, n_output_dims)
            if self.is_linear:
                return out
            return torch.sin(self.omega_0 * out)
        else:
            weights = self.weights_alpha @ fourier_mod @ self.weights_beta
            out = F.linear(input, weights.T, self.bias)
            return torch.sin(self.omega_0 * out) # (B, n_output_dims)

class BackboneINR(nn.Module):
    def __init__(
        self, 
        n_input_dims: int, 
        n_output_dims: int,
        n_fourier_bases: int,
        n_neurons: int,
        n_layers: int, 
        outermost_linear: bool=False, 
        first_omega_0: float=50, 
        hidden_omega_0: float=30,
        **kwargs
    ):
        super().__init__()

        self.input_layer = SinFRLayer(n_input_dims, n_neurons, omega_0=first_omega_0, is_first=True)
        self.output_layer = SinFRLayer(n_neurons, n_output_dims, omega_0=hidden_omega_0, is_linear=outermost_linear)

        self.net = []
        for _ in range(n_layers):
            self.net.append(SinFRLayer(n_neurons, n_neurons, omega_0=hidden_omega_0, square_dim=int(n_fourier_bases**(1/2)), is_reparam=True))
        self.net = nn.ModuleList(self.net)
        
    def forward(self, coords: torch.Tensor, fourier_mods: torch.Tensor):
        # coords (B, n_input_dims)
        # fourier_mods (n_layer, square_dim, square_dim)
        out = self.input_layer(coords, fourier_mods[0])
        for i, layer in enumerate(self.net):
            out = layer(out, fourier_mods[i])
        out = self.output_layer(out)

        return out # (B, n_output_dims)
    
class Implisat(nn.Module):
    def __init__(
        self, 
        n_input_dims: int, 
        n_output_dims: int, # this should be 1
        n_fourier_bases: int,
        n_neurons: int,
        n_layers: int, 
        outermost_linear: bool = False, 
        first_omega_0: float = 30, 
        hidden_omega_0: float = 30,
        output_activation: str = 'None',
        is_phase1d: bool = False,
        **kwargs
    ):
        super().__init__()

        assert n_output_dims == 1, 'n_output_dims should be 1'
        assert int(n_fourier_bases ** (1/2)) ** 2 == n_fourier_bases, 'n_fourier_bases should be a square number'
        if is_phase1d:
            self.fourier_net = FourierNetwork1DPhase(n_neurons, n_fourier_bases, n_layers)
        else:
            self.fourier_net = FourierNetwork2DPhase(n_neurons, n_fourier_bases, n_layers)
        self.inr = BackboneINR(n_input_dims, n_output_dims, n_fourier_bases, n_neurons, n_layers, outermost_linear, first_omega_0, hidden_omega_0)
        self.output_activation = (getattr(torch.nn, output_activation) if output_activation != 'None' else torch.nn.Identity)()
    
    def forward_fourier(self, channel: torch.Tensor, resolution: torch.Tensor):
        return self.fourier_net(channel, resolution)
    
    def forward_inr(self, coords: torch.Tensor, fourier_mods: torch.Tensor):
        out = []
        for i in range(fourier_mods.shape[0]): # iterate for each channel in the image
            out.append(self.inr(coords, fourier_mods[i]))
        out = torch.cat(out, dim=1)
        out = self.output_activation(out)
        return out # (B, n_channels)

    def forward_inr_with_cp(self, coords: torch.Tensor, fourier_mods: torch.Tensor):
        out = []
        for i in range(fourier_mods.shape[0]): # iterate for each channel in the image
            x = cp.checkpoint(self.inr, coords, fourier_mods[i], use_reentrant=False)
            out.append(x)
        out = torch.cat(out, dim=1)
        out = self.output_activation(out)
        return out # (B, n_channels)

if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)
    model = Implisat(2, 1, 1024, 128, 3, True, 30, 30)
    channel = F.one_hot(torch.arange(13), num_classes=13).float()
    resolution = torch.tensor([0.1] * 4 + [0.2] * 6 + [0.6] * 3).unsqueeze(1)
    coords = torch.randn(258*258, 2)
    print('coords', coords.shape)
    print('channel', channel.shape)
    print('resolution', resolution.shape)

    fourier_mods = model.forward_fourier(channel, resolution)
    print('fourier mods', fourier_mods.shape)

    out = model.forward_inr(coords, fourier_mods)
    print(out.shape)