import torch
import math

class ActLinear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int = 1, out_features: int = 1,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ActLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(in_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv=1/math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,0)
        self.bias.data.uniform_(-stdv/4,0)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.add(torch.mul(input, self.weight),self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ActLinear_2(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int = 1, out_features: int = 1,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ActLinear_2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((in_features), **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty((in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv=1/math.sqrt(self.weight.size(0))*0.01
        self.weight.data.uniform_(-stdv,stdv)
        # self.bias.data.uniform_(-stdv,stdv)
        torch.nn.init.zeros_(self.bias)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.add(torch.mul(input, self.weight),self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
