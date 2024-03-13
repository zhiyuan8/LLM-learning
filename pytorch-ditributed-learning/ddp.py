import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F


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
    # Setup for device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # Initialize process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Model and optimizer
    model = SimpleCNN().to(device)
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # Example data for forward pass
    inputs = torch.randn(64, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (64,)).to(device)
    
    optimizer.zero_grad()  # Clear gradients
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # Saving model state (only by rank 0)
    if rank == 0:
        torch.save(model.state_dict(), 'simple_cnn_model.pt')  # Save underlying model state


def load_model_for_inference(model_path='simple_cnn_model.pt', device=torch.device('cpu')):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Example of performing inference
def perform_inference(model, input_tensor):
    with torch.no_grad():  # No need to track gradients for inference
        output = model(input_tensor)
        return output

def main():
    world_size = 2
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)
    
    # Loading model
    model_for_inference = load_model_for_inference()
    
    # Assuming you have an input tensor for inference
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor
    output = perform_inference(model_for_inference, input_tensor)
    print(output)


if __name__ == "__main__":
    main()
