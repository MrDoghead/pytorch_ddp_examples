"""
without ddp
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import cpu_mem_stats, MB

# define a simple model
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        # layer1
        self.weight1 = nn.Parameter(torch.randn(hidden_size, 2*hidden_size, dtype=torch.float32))
        self.bias1   = nn.Parameter(torch.zeros((2*hidden_size,), dtype=torch.float32))
        # layer2
        self.weight2 = nn.Parameter(torch.randn(2*hidden_size, hidden_size, dtype=torch.float32))
        self.bias2 = nn.Parameter(torch.zeros((hidden_size,), dtype=torch.float32))
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = F.linear(x, self.weight1.T, self.bias1)
        x = self.activation(x)
        x = F.linear(x, self.weight2.T, self.bias2)
        return x
    
    def mem_stats(self):
        assert self.weight1.device == self.weight2.device
        if self.weight1.is_cuda:
            dev = self.weight1.device
            cur_mem = torch.cuda.memory_allocated(dev)
            peak_mem = torch.cuda.max_memory_allocated(dev)
        elif self.weight1.device == torch.device("cpu"):
            cur_mem = cpu_mem_stats()
            peak_mem = 0
        else:
            raise NotImplementedError()

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        cur_mem, peak_mem = self.mem_stats()

        if output_file is not None:
            with open(output_file, "w") as f:
                f.write(f"TorchDevice: {self.weight1.device}\n")
                f.write(f"  cur_mem: {cur_mem/MB:.4f} MB, "
                        f" peak_mem: {peak_mem/MB:.4f} MB\n")
        else:
            print(f"TorchDevice: {self.weight1.device}")
            print(f"  cur_mem: {cur_mem/MB:.4f} MB, "
                  f" peak_mem: {peak_mem/MB:.4f} MB")

        return cur_mem, peak_mem


if __name__ == "__main__":
    torch.manual_seed(55)

    # init inputs and devices
    batch_size = 4
    seq_len = 32
    hidden_size = 256
    inputs = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    model = MLP(hidden_size)
    model.print_stats()

    inputs = inputs.cuda()
    model = model.cuda()
    model.print_stats()

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    outputs = model(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    model.print_stats()

    print("Outputs:", outputs, outputs.shape)
    print("Time:", end_time-start_time)