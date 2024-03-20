"""
Example: pipeline parallel
CUDA_VISIBLE_DEVICES=0,1 python demo_pp.py --pp 2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import time
from utils import set_seed, MB

def add_parser_arguments(parser):
    parser.add_argument("--n_samples", type=int, default=4, help="number of samples")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--seq_len", type=int, default=32, help="sequence length")
    parser.add_argument("--hidden_size", type=int, default=256, help="hidden size")
    parser.add_argument("--dp", type=int, default=1, help="data parallel number")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel number")
    parser.add_argument("--pp", type=int, default=1, help="pipeline parallel number")

# define a simple model
class MLP(nn.Module):
    def __init__(self, hidden_size, pp_num=1):
        super(MLP, self).__init__()
        self.pp_num = pp_num
        self.device_ids = list(range(pp_num))
        # layer1
        self.weight1 = nn.Parameter(torch.randn(hidden_size, 2*hidden_size, dtype=torch.float32)).cuda(self.device_ids[0])
        self.bias1 = nn.Parameter(torch.zeros((2*hidden_size,), dtype=torch.float32)).cuda(self.device_ids[0])
        # layer2
        self.weight2 = nn.Parameter(torch.randn(2*hidden_size, hidden_size, dtype=torch.float32)).cuda(self.device_ids[1])
        self.bias2 = nn.Parameter(torch.zeros((hidden_size,), dtype=torch.float32)).cuda(self.device_ids[1])
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = F.linear(x.to(self.weight1.device), self.weight1.T, self.bias1) # cuda0
        x = self.activation(x) # cuda0
        # self.print_stats()
        x = F.linear(x.to(self.weight2.device), self.weight2.T, self.bias2) # cuda1
        self.print_stats()
        return x
    
    def mem_stats(self, dev):
        # only for cuda
        cur_mem = torch.cuda.memory_allocated(dev)
        peak_mem = torch.cuda.max_memory_allocated(dev)

        return cur_mem, peak_mem

    def print_stats(self, output_file=None):
        torch.cuda.synchronize()
        for i in range(self.pp_num):
            cur_mem, peak_mem = self.mem_stats(i)

            if output_file is not None:
                with open(output_file, "w") as f:
                    f.write(f"TorchDevice: cuda {i}\n")
                    f.write(f"  cur_mem: {cur_mem/MB:.4f} MB, "
                            f" peak_mem: {peak_mem/MB:.4f} MB\n")
            else:
                print(f"TorchDevice: cuda {i}")
                print(f"  cur_mem: {cur_mem/MB:.4f} MB, "
                    f" peak_mem: {peak_mem/MB:.4f} MB")

        return cur_mem, peak_mem
    
class MyDataset(Dataset):
    def __init__(self, samples, n_gpu=1) -> None:
        super().__init__()
        self.seq_len = samples[0].shape[0]
        self.samples = samples
        self.n_samples = len(samples)
        self.n_padded_samples = 0
        if self.n_samples % n_gpu != 0:
            self.n_padded_samples = n_gpu - self.n_samples % n_gpu
            for _ in range(self.n_padded_samples):
                self.samples.append(torch.zeros_like(samples[0]))

    def __len__(self):
        return self.n_samples + self.n_padded_samples
    
    def __getitem__(self, index):
        data = self.samples[index]
        label = torch.tensor(index) # to align the all-gather results
        return data, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    print("Args:", args)
    set_seed(55)

    # randomly initialize samples
    n_samples = args.n_samples
    batch_size = args.batch_size
    seq_len = args.seq_len
    hidden_size = args.hidden_size
    samples = [torch.randn(seq_len, hidden_size, dtype=torch.float32) for _ in range(n_samples)]

    test_dataset = MyDataset(samples)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False
                             )
    model = MLP(hidden_size, pp_num=args.pp)

    outputs = []
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for inp, label in test_loader:
        inp = inp.cuda(0)
        out = model(inp)
        outputs.append(out)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    outputs = torch.cat(outputs, dim=0)
    print("Outputs:", outputs, outputs.shape)
    print("Time:", end_time-start_time)

