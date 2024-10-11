# import tinycudann as tcnn
import torch
import math

class GridEncoderTcnn(torch.nn.Module):
    def __init__(
        self,
        n_input_dims: int,
        n_levels: int = 16, # L in paper
        n_features_per_level: int = 2, # F in paper
        log2_hashmap_size: int = 15, # log T in paper
        base_resolution: int = 16, # N_min in paper
        per_level_scale: float = 1.5, # b in paper
        max_resolution: int = None, # overrides per_level_scale if set
        full_precision: bool = False,
    ):
        super().__init__()
        if max_resolution is not None:
            per_level_scale = math.exp(math.log(max_resolution / base_resolution) / (n_levels - 1))

        self.model = tcnn.Encoding(n_input_dims, {
            'otype': 'HashGrid',
            'n_levels': n_levels,
            'n_features_per_level': n_features_per_level,
            'log2_hashmap_size': log2_hashmap_size,
            'base_resolution': base_resolution,
            'per_level_scale': per_level_scale,
        }, dtype=torch.float32 if full_precision else torch.float16)
        
        # for i in range(n_levels):
        #     print(f'Resolution at level-{i}:', math.floor(base_resolution * (per_level_scale ** i)))

    def forward(self, x: torch.Tensor):
        return self.model(x)

if __name__ == '__main__':
    encoding = GridEncoderTcnn(2, max_resolution=1024)
    input_tensor = torch.randn(100000, 2).to('cuda')
    encoding.to('cuda')
    output_tensor = encoding(input_tensor)
    for param in encoding.parameters():
        print(param.shape)