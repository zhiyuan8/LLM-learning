import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def example(rank, world_size):
    # Setup for GPU or CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size) if torch.cuda.is_available() else dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=[rank] if device.type == 'cuda' else None)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    outputs = ddp_model(torch.randn(64, 1, 28, 28).to(device))
    labels = torch.randint(0, 10, (64,)).to(device)
    loss_fn(outputs, labels).backward()
    optimizer.step()

def main():
    world_size = 2
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
