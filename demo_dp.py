"""
Example: data parallel
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 demo_dp.py --dp 4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import argparse
import time
from utils import cpu_mem_stats, MB, set_seed

def add_parser_arguments(parser):
    parser.add_argument("--n_samples", type=int, default=4, help="number of samples")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--seq_len", type=int, default=32, help="sequence length")
    parser.add_argument("--hidden_size", type=int, default=256, help="hidden size")
    parser.add_argument("--dp", type=int, default=1, help="data parallel number")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel number")
    parser.add_argument("--pp", type=int, default=1, help="pipeline parallel number")

def _initialize_distributed(args):
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
    # Call the init process
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank)

# define a simple model
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        # layer1
        self.weight1 = nn.Parameter(torch.randn(hidden_size, 2*hidden_size, dtype=torch.float32))
        self.bias1 = nn.Parameter(torch.zeros((2*hidden_size,), dtype=torch.float32))
        # layer2
        self.weight2 = nn.Parameter(torch.randn(2*hidden_size, hidden_size, dtype=torch.float32))
        self.bias2 = nn.Parameter(torch.zeros((hidden_size,), dtype=torch.float32))
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = F.linear(x, self.weight1.T, self.bias1)
        x = self.activation(x)
        x = F.linear(x, self.weight2.T, self.bias2)
        self.print_stats()
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

    args.local_rank = None
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    args.distributed_backend = "nccl"
    _initialize_distributed(args)

    set_seed(55)
    if torch.distributed.get_rank() == 0:
        print("Args:", args)

    # randomly initialize samples
    n_samples = args.n_samples
    batch_size = args.batch_size
    seq_len = args.seq_len
    hidden_size = args.hidden_size
    samples = [torch.randn(seq_len, hidden_size, dtype=torch.float32) for _ in range(n_samples)]

    model = MLP(hidden_size)

    test_dataset = MyDataset(samples, torch.distributed.get_world_size())
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             sampler=DistributedSampler(test_dataset),
                             shuffle=False
                             )
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                      broadcast_buffers=False, 
                                                    #   find_unused_parameters=True
                                                      )
    
    outputs = []
    orders = []
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for inp, label in test_loader:
        inp = inp.cuda()
        label = label.cuda()
        out = model(inp)
        gather_outputs = [torch.zeros_like(out) for _ in range(torch.distributed.get_world_size())]
        gather_orders = [torch.zeros_like(label) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gather_outputs, out) # not ordered
        torch.distributed.all_gather(gather_orders, label)
        outputs.extend(gather_outputs)
        orders.extend(gather_orders)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    outputs = torch.cat(outputs, dim=0)
    orders = [idx.item() for idx in orders]
    outputs = outputs[orders, :, :]
    outputs = outputs[:test_dataset.n_samples, :, :]
    if torch.distributed.get_rank() == 0:
        print("Outputs:", outputs, outputs.shape)
        print(orders)
        print("Time:", end_time-start_time)
