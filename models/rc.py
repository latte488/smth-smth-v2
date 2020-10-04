import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import random


class ESNCell(nn.Module):
    def __init__(self, size_in, size_res):
        super(ESNCell, self).__init__()
        self.size_in = size_in
        self.size_res = size_res
        self.w_in = Parameter(torch.Tensor(size_in, size_res))   
        self.register_buffer('w_res', torch.Tensor(size_res, size_res))
        self.b = Parameter(torch.Tensor(size_res))
        self.reset_parameters()

    def reset_parameters(self):
        self._reset_weight_in()
        self._reset_weight_res()
        self._reset_bias()

    def _reset_weight_in(self):
        init.kaiming_uniform_(self.w_in, a=math.sqrt(5))

    def _reset_weight_res(self):
        adjency = torch.Tensor([random.randint(0, 1) for _ in range(self.size_res**2)])
        init.kaiming_uniform_(self.w_res, a=math.sqrt(5))
        self.w_res *= adjency.view(self.size_res, self.size_res)

    def _reset_bias(self):
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_in)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def _reset_state(self):
        init.zeros_(self.x)

    def forward(self, inputs, states):
        return torch.tanh(inputs @ self.w_in \
            + states @ self.w_res \
            + self.b)

class ESN(nn.Module):
    def __init__(self, size_in, size_res, batch_first=False):
        super(ESN, self).__init__()
        self.cell = ESNCell(size_in, size_res)
        self.batch_first = batch_first

    def forward(self, xs):
        if self.batch_first:
            b, t, f = 0, 1, 2
            xs = xs.permute(t, b, f)
            xs = self._forward(xs)
            xs = xs.permute(t, b, f)
        else:
            xs = self._forward(xs)
        return xs


    def _forward(self, xs):
        states = torch.Tensor(xs.size(1), self.cell.size_res).to(xs.device)
        init.zeros_(states)
        states_list = []
        for x in xs:
            states = self.cell(x, states.detach())
            states_list.append(states)
        return torch.stack(states_list)

if __name__ == '__main__':
    model = ESN(4, 10)
    inputs = torch.randn(6, 8, 4)
    outputs = model(inputs)
    print(outputs.shape)
