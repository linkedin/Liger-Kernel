import torch
import torch.nn as nn
import torch.distributed as dist

import os
import datetime
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        # Replace BatchNorm with SyncBatchNorm for distributed training
        self.bn1 = nn.SyncBatchNorm(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.SyncBatchNorm(20)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(1440, 10)  # Assuming output size after convolutions

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 1440)  # Flatten for fully-connected layer
        x = self.fc(x)
        return x


def main():
    # Initialize distributed training (replace with your specific initialization logic)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)  # Example for 4 processes
    print('main')
    model = MyCNN()
    model.to('cuda')
    # Wrap the model with DistributedDataParallel (DDP) for distributed training
    model = nn.parallel.DistributedDataParallel(model)

    # Define your optimizer, loss function, and training loop (omitted for brevity)

    # ... training code ...

    dist.destroy_process_group()  # Clean up after training


if __name__ == "__main__":
    main()
