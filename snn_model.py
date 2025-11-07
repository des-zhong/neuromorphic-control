import torch
import torch.nn as nn
import torch.nn.functional as F
from uuid import uuid1
from wrap_load_save import rand


class lynchip_state_wrapper():
    def __init__(self, mem, on_apu):
        self.mem = mem
        self.id = uuid1()
        self.on_apu = on_apu

    @staticmethod
    def lynload(tensor, uuid):
        from torch import ops
        from custom_op_in_lyn.custom_op_load_save import load
        ops.load_library('./custom_op_in_pytorch/build/libcustom_ops.so')
        from wrap_load_save import load
        return load(tensor, f'{uuid}')

    @staticmethod
    def lynsave(tensor, uuid):
        from torch import ops
        from custom_op_in_lyn.custom_op_load_save import save
        ops.load_library('./custom_op_in_pytorch/build/libcustom_ops.so')
        from wrap_load_save import save
        save(tensor, f'{uuid}')

    def get(self):
        if not self.on_apu:
            return self.mem
        else:
            return self.lynload(self.mem.clone(), self.id)

    def set(self, mem):
        if not self.on_apu:
            self.mem = mem
        else:
            self.lynsave(mem.clone(), self.id)

class Spike_Act_with_Surrogate_Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vth):
        ctx.save_for_backward(input)
        ctx.vth = vth
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        lens = 0.5
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - ctx.vth) < lens
        return grad_input * temp.float() / (lens * 2), None

class Spiking_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vth, decay, activation, on_apu, init_batch=1):
        super().__init__()
        self.Spike_Act = activation
        self.on_apu = on_apu
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.vth = nn.Parameter(vth * torch.ones(1), requires_grad=False)
        self.decay = nn.Parameter(decay * torch.ones(1), requires_grad=False)

        self.fc = [nn.Linear(input_dim, hidden_dim[0])]
        for idx in range(len(hidden_dim)):
            self.fc.append(nn.Linear(hidden_dim[idx], hidden_dim[idx + 1] if idx + 1 < len(hidden_dim) else output_dim))
        
        self.init_mem(batch_size=init_batch)

    # def init_mem(self):
    #     self.mem1 = lynchip_state_wrapper(torch.zeros(1, self.num_channels, self.img_size, self.img_size), self.on_apu)
    #     self.mem2 = lynchip_state_wrapper(torch.zeros(1, self.num_channels, self.img_size // 2, self.img_size // 2), self.on_apu)
    #     self.mem_fc = lynchip_state_wrapper(torch.zeros(1, self.num_classes), self.on_apu)

    def init_mem(self, batch_size=1):
        # 根据 batch_size 动态初始化记忆单元
        self.mem = [
            lynchip_state_wrapper(torch.zeros(batch_size, self.input_dim), self.on_apu)
        ]
        for idx in range(len(self.hidden_dim)):
            self.mem.append(lynchip_state_wrapper(torch.zeros(batch_size, self.hidden_dim[idx]), self.on_apu))
        self.mem.append(lynchip_state_wrapper(torch.zeros(batch_size, self.output_dim), self.on_apu))
    

    
    def mem_update(self, op, x, idx, return_mem=False):
        mem_wrapper = self.mem[idx]
        mem = mem_wrapper.get().to(x.device)
        # print(mem)
        spk = self.Spike_Act(mem, self.vth).float()
        if return_mem:
            new_mem = mem * (1 - spk) * self.decay + op(x)
        else:
            new_mem = mem * (1 - spk) * self.decay*8 + op(x)
        mem_wrapper.set(new_mem)
        if return_mem:
            return new_mem
        return spk


    # def forward(self, x):
    #     if self.on_apu:
    #         rr = rand(x[0,0,0,0], x.size(), 0x4000, mode=0)
    #         x = (rr < x).float()
    #     else:
    #         x = torch.bernoulli(x)
        
    #     x = self.mem_update(self.conv1, x, self.mem1, self.vth, self.decay)
    #     x = nn.functional.max_pool2d(x, 2)
    #     x = self.mem_update(self.conv2, x, self.mem2, self.vth, self.decay)
    #     x = nn.functional.max_pool2d(x, 2)
    #     x = x.view(x.size(0), -1)
    #     x = self.mem_update(self.fc, x, self.mem_fc, self.vth, self.decay)
    #     return x
    
    # def forward(self, x):
    #     
            
    #     x = self.mem_update(self.conv1, x, 0)
    #     x = F.max_pool2d(x, 2)
    #     x = self.mem_update(self.conv2, x, 1)
    #     x = F.max_pool2d(x, 2)
    #     x = self.mem_update(self.conv3, x, 2)
    #     x = F.max_pool2d(x, 2)
    #     x = self.mem_update(self.conv4, x, 3)
    #     x = F.max_pool2d(x, 2)
        
    #     x = x.view(x.size(0), -1)
    #     x = self.mem_update(self.fc1, x, 4)
    #     x = self.mem_update(self.fc2, x, 5)
    #     x = self.mem_update(self.fc3, x, 6, return_mem=True)
    #     return x
    
    def forward(self, x):
        for idx in range(len(self.hidden_dim) + 2):
            x = self.mem_update(self.fc[idx], x, idx)
        x = self.mem_update(self.fc1, x, 0)
        return x