
---
title: PyTorch Cheat Sheet
description: A comprehensive reference guide for PyTorch, covering tensors, neural networks, training, CUDA, deployment, and advanced techniques.
---

# PyTorch Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of the PyTorch deep learning library, covering essential concepts, code snippets, and best practices for efficient model building, training, and deployment. It aims to be a one-stop reference for common tasks.

## Getting Started

### Installation

```bash
# CPU-only version
pip install torch torchvision torchaudio

# CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version. Check the [PyTorch website](https://pytorch.org/get-started/locally/) for the most up-to-date installation instructions.

### Importing PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

# Check version
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Tensors

### Tensor Creation Flow

```
    ┌────────────────────────────────────────┐
    │     Tensor Creation Methods            │
    └───┬────────────────────────────────┬───┘
        │                                │
        ↓                                ↓
    ┌─────────┐                    ┌─────────┐
    │  From   │                    │  From   │
    │  Data   │                    │ Scratch │
    └────┬────┘                    └────┬────┘
         │                              │
         ↓                              ↓
   ┌──────────┐                   ┌──────────┐
   │ .tensor()│                   │ .zeros() │
   │ .from_   │                   │ .ones()  │
   │  numpy() │                   │ .rand()  │
   │ .as_     │                   │ .randn() │
   │  tensor()│                   │ .empty() │
   └──────────┘                   └──────────┘
```

### Creating Tensors

From a List:

```python
# Direct creation from list
data = [1, 2, 3, 4, 5]
tensor = torch.tensor(data)
print(tensor)  # tensor([1, 2, 3, 4, 5])

# 2D tensor
data_2d = [[1, 2, 3], [4, 5, 6]]
tensor_2d = torch.tensor(data_2d)
print(tensor_2d.shape)  # torch.Size([2, 3])
```

From a NumPy Array:

```python
import numpy as np

# Convert NumPy to tensor (shares memory)
data = np.array([1, 2, 3, 4, 5])
tensor = torch.from_numpy(data)

# Changes in NumPy array affect tensor
data[0] = 100
print(tensor)  # tensor([100, 2, 3, 4, 5])

# Create independent copy
tensor_copy = torch.tensor(data)
```

Zeros, Ones, and Filled Tensors:

```python
# Create tensor filled with zeros
zeros = torch.zeros(3, 4)
print(zeros.shape)  # torch.Size([3, 4])

# Create tensor filled with ones
ones = torch.ones(3, 4)

# Create tensor filled with specific value
full = torch.full((3, 4), fill_value=7)

# Create tensor like another tensor
x = torch.tensor([[1, 2], [3, 4]])
zeros_like = torch.zeros_like(x)
ones_like = torch.ones_like(x)

# Empty tensor (uninitialized memory)
empty = torch.empty(3, 4)
```

Ranges:

```python
# Create range of integers
arange = torch.arange(start=0, end=10, step=2)
print(arange)  # tensor([0, 2, 4, 6, 8])

# Create linearly spaced values
linspace = torch.linspace(start=0, end=1, steps=5)
print(linspace)  # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

# Logarithmically spaced values
logspace = torch.logspace(start=0, end=2, steps=5)
print(logspace)  # tensor([1., 3.1623, 10., 31.6228, 100.])
```

Random Number Generation:

```python
# Set random seed for reproducibility
torch.manual_seed(42)

# Uniform distribution [0, 1)
rand = torch.rand(3, 4)

# Standard normal distribution (mean=0, std=1)
randn = torch.randn(3, 4)

# Random integers in range [low, high)
randint = torch.randint(low=0, high=10, size=(3, 4))

# Random permutation
perm = torch.randperm(10)  # tensor([2, 5, 1, 9, 0, 3, 7, 4, 6, 8])

# Random sampling from normal distribution
normal = torch.normal(mean=0.0, std=1.0, size=(3, 4))
```

### Tensor Attributes

```python
tensor.shape       # Shape of the tensor
tensor.size()      # Same as shape
tensor.ndim        # Number of dimensions
tensor.dtype       # Data type of the tensor
tensor.device      # Device where the tensor is stored (CPU or GPU)
tensor.requires_grad # Whether gradients are tracked
tensor.layout      # Memory layout (torch.strided, torch.sparse_coo)
```

### Tensor Operations

```
    ┌──────────────────────────────────┐
    │    Tensor Operation Types        │
    └────┬────────────────────────┬────┘
         │                        │
         ↓                        ↓
    ┌─────────┐             ┌─────────┐
    │Arithmetic│            │ Shape   │
    │Operations│            │Operations│
    └────┬────┘             └────┬────┘
         │                       │
    ┌────┴────┐             ┌────┴────┐
    │ +, -, *,│             │ reshape │
    │ /, **, @│             │ view    │
    │ add_()  │             │ squeeze │
    │ sub_()  │             │ unsqueeze│
    └─────────┘             └─────────┘
```

Arithmetic Operations:

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
c = a + b       # tensor([5., 7., 9.])
d = a - b       # tensor([-3., -3., -3.])
e = a * b       # tensor([4., 10., 18.])
f = a / b       # tensor([0.25, 0.4, 0.5])
g = a ** 2      # tensor([1., 4., 9.])

# In-place operations (modify original tensor)
a.add_(b)       # a = tensor([5., 7., 9.])
a.sub_(b)       # Subtract in-place
a.mul_(b)       # Multiply in-place
a.div_(b)       # Divide in-place

# Functional form
result = torch.add(a, b)
result = torch.mul(a, b)

# Scalar operations
a = torch.tensor([1.0, 2.0, 3.0])
scaled = a * 2  # tensor([2., 4., 6.])
shifted = a + 5  # tensor([6., 7., 8.])
```

Slicing and Indexing:

```python
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Basic indexing
first_row = tensor[0]           # tensor([1, 2, 3])
second_col = tensor[:, 1]       # tensor([2, 5])
element = tensor[0, 1]          # tensor(2)

# Slicing
sliced = tensor[0:2, 1:3]       # tensor([[2, 3], [5, 6]])

# Boolean indexing
mask = tensor > 3
filtered = tensor[mask]         # tensor([4, 5, 6])

# Advanced indexing
indices = torch.tensor([0, 1, 0])
selected = tensor[:, indices]   # Select columns

# Fancy indexing
rows = torch.tensor([0, 1])
cols = torch.tensor([1, 2])
elements = tensor[rows, cols]   # tensor([2, 6])
```

Reshaping:

```python
tensor = torch.arange(12)
print(tensor)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# Reshape to 2D
reshaped = tensor.reshape(3, 4)
print(reshaped.shape)  # torch.Size([3, 4])

# View (shares memory with original)
viewed = tensor.view(3, 4)  # Same as reshape but requires contiguous tensor

# Transpose (2D tensors)
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
transposed = tensor_2d.T
print(transposed.shape)  # torch.Size([3, 2])

# Permute dimensions (generalized transpose)
tensor_3d = torch.randn(2, 3, 4)
permuted = tensor_3d.permute(2, 0, 1)  # New shape: (4, 2, 3)

# Flatten to 1D
flattened = tensor_2d.flatten()
print(flattened)  # tensor([1, 2, 3, 4, 5, 6])

# Squeeze (remove dimensions of size 1)
tensor_with_ones = torch.randn(1, 3, 1, 4)
squeezed = tensor_with_ones.squeeze()  # Shape: (3, 4)

# Unsqueeze (add dimension of size 1)
tensor_1d = torch.tensor([1, 2, 3])
unsqueezed = tensor_1d.unsqueeze(0)  # Shape: (1, 3)
unsqueezed = tensor_1d.unsqueeze(1)  # Shape: (3, 1)

# Contiguous (ensure tensor is contiguous in memory)
contiguous = transposed.contiguous()
```

Concatenation and Stacking:

```python
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# Concatenate along existing dimension
cat_rows = torch.cat((tensor1, tensor2), dim=0)  # Shape: (4, 2)
print(cat_rows)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

cat_cols = torch.cat((tensor1, tensor2), dim=1)  # Shape: (2, 4)
print(cat_cols)
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])

# Stack creates new dimension
stacked = torch.stack((tensor1, tensor2), dim=0)  # Shape: (2, 2, 2)
print(stacked.shape)  # torch.Size([2, 2, 2])

# Split tensor into chunks
tensor = torch.arange(10)
chunks = torch.chunk(tensor, 3)  # Split into 3 chunks
print(chunks)  # (tensor([0, 1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))

# Split with specific sizes
split = torch.split(tensor, [2, 3, 5])  # Sizes: 2, 3, 5
```

Matrix Multiplication:

```python
# Matrix multiplication
a = torch.randn(3, 4)
b = torch.randn(4, 5)

# Three equivalent ways
c = torch.matmul(a, b)     # Functional form
d = a @ b                  # Operator form (preferred)
e = torch.mm(a, b)         # Explicit matrix multiplication

print(c.shape)  # torch.Size([3, 5])

# Batch matrix multiplication
batch1 = torch.randn(10, 3, 4)  # 10 matrices of shape (3, 4)
batch2 = torch.randn(10, 4, 5)  # 10 matrices of shape (4, 5)
batch_result = torch.bmm(batch1, batch2)  # Shape: (10, 3, 5)

# Element-wise multiplication (Hadamard product)
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a * b  # tensor([4, 10, 18])

# Dot product (1D tensors)
dot_product = torch.dot(a, b)  # tensor(32)

# Outer product
outer = torch.outer(a, b)  # Shape: (3, 3)

# Matrix-vector multiplication
matrix = torch.randn(3, 4)
vector = torch.randn(4)
result = torch.mv(matrix, vector)  # Shape: (3,)
```

### Data Types

*   `torch.float32` or `torch.float`: 32-bit floating point
*   `torch.float64` or `torch.double`: 64-bit floating point
*   `torch.float16` or `torch.half`: 16-bit floating point
*   `torch.bfloat16`: BFloat16 floating point (useful for mixed precision)
*   `torch.int8`: 8-bit integer (signed)
*   `torch.int16` or `torch.short`: 16-bit integer (signed)
*   `torch.int32` or `torch.int`: 32-bit integer (signed)
*   `torch.int64` or `torch.long`: 64-bit integer (signed)
*   `torch.uint8`: 8-bit integer (unsigned)
*   `torch.bool`: Boolean

### Device Management

```
    ┌─────────────────────────┐
    │   Device Management     │
    └──────────┬──────────────┘
               │
        ┌──────┴──────┐
        │             │
        ↓             ↓
    ┌───────┐    ┌────────┐
    │  CPU  │    │  GPU   │
    │ tensor│←──→│ tensor │
    └───────┘    └────────┘
        │             │
        ↓             ↓
    .cpu()        .cuda()
    .to('cpu')    .to('cuda')
```

```python
# Check device availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create tensor on specific device
tensor_gpu = torch.randn(3, 4, device=device)
print(tensor_gpu.device)  # cuda:0 or cpu

# Move tensor between devices
tensor_cpu = torch.randn(3, 4)
tensor_gpu = tensor_cpu.to(device)      # Move to GPU
tensor_gpu = tensor_cpu.cuda()          # Alternative
tensor_cpu_back = tensor_gpu.cpu()      # Move back to CPU

# Check tensor device
print(f"Tensor is on CUDA: {tensor_gpu.is_cuda}")

# Multiple GPU support
if torch.cuda.device_count() > 1:
    tensor_gpu0 = tensor.to('cuda:0')
    tensor_gpu1 = tensor.to('cuda:1')

# Set active GPU
torch.cuda.set_device(0)  # Use GPU 0
```

## Neural Networks

### Neural Network Architecture

```
    ┌──────────┐
    │  Input   │
    │  Layer   │
    └────┬─────┘
         │
         ↓
    ┌──────────┐
    │  Hidden  │
    │  Layer 1 │──→ Activation (ReLU)
    └────┬─────┘
         │
         ↓
    ┌──────────┐
    │  Hidden  │
    │  Layer 2 │──→ Activation (ReLU)
    └────┬─────┘
         │
         ↓
    ┌──────────┐
    │  Output  │
    │  Layer   │──→ Softmax/Sigmoid
    └──────────┘
```

### Defining a Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple feedforward network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation for logits
        return x

# Instantiate model
model = SimpleNet(input_size=784, hidden_size=128, num_classes=10)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

### Layers

*   `nn.Linear`: Fully connected layer.
*   `nn.Conv1d`: 1D convolution layer.
*   `nn.Conv2d`: 2D convolution layer.
*   `nn.Conv3d`: 3D convolution layer.
*   `nn.ConvTranspose2d`: Transposed convolution layer (deconvolution).
*   `nn.MaxPool1d`, `nn.MaxPool2d`, `nn.MaxPool3d`: Max pooling layers.
*   `nn.AvgPool1d`, `nn.AvgPool2d`, `nn.AvgPool3d`: Average pooling layers.
*   `nn.AdaptiveAvgPool2d`: Adaptive average pooling layer.
*   `nn.ReLU`: ReLU activation function.
*   `nn.Sigmoid`: Sigmoid activation function.
*   `nn.Tanh`: Tanh activation function.
*   `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.BatchNorm3d`: Batch normalization layers.
*   `nn.LayerNorm`: Layer normalization layer.
*   `nn.Dropout`: Dropout layer.
*   `nn.Embedding`: Embedding layer.
*   `nn.LSTM`: LSTM layer.
*   `nn.GRU`: GRU layer.
*   `nn.Transformer`: Transformer layer.
*   `nn.TransformerEncoder`, `nn.TransformerDecoder`: Transformer encoder and decoder layers.
*   `nn.MultiheadAttention`: Multi-head attention layer.

### Activation Functions

*   `torch.relu`: Rectified Linear Unit.
*   `torch.sigmoid`: Sigmoid function.
*   `torch.tanh`: Hyperbolic tangent function.
*   `torch.softmax`: Softmax function (for multi-class classification).
*   `torch.elu`: Exponential Linear Unit.
*   `torch.selu`: Scaled Exponential Linear Unit.
*   `torch.leaky_relu`: Leaky Rectified Linear Unit.
*   `torch.gelu`: Gaussian Error Linear Unit (GELU).
*   `torch.silu`: SiLU (Sigmoid Linear Unit) or Swish.

### Loss Functions

*   `nn.CrossEntropyLoss`: Cross-entropy loss (for multi-class classification).
*   `nn.BCELoss`: Binary cross-entropy loss (for binary classification).
*   `nn.BCEWithLogitsLoss`: Binary cross-entropy with logits (more stable).
*   `nn.MSELoss`: Mean squared error loss (for regression).
*   `nn.L1Loss`: Mean absolute error loss (for regression).
*   `nn.SmoothL1Loss`: Huber loss (for robust regression).
*   `nn.CTCLoss`: Connectionist Temporal Classification loss (for sequence labeling).
*   `nn.TripletMarginLoss`: Triplet margin loss (for learning embeddings).
*   `nn.CosineEmbeddingLoss`: Cosine embedding loss.

### Optimizers

*   `optim.SGD`: Stochastic Gradient Descent.
*   `optim.Adam`: Adaptive Moment Estimation.
*   `optim.RMSprop`: Root Mean Square Propagation.
*   `optim.Adagrad`: Adaptive Gradient Algorithm.
*   `optim.Adadelta`: Adaptive Delta.
*   `optim.AdamW`: Adam with weight decay regularization.
*   `optim.SparseAdam`: Adam optimizer for sparse tensors.

### Optimizer Configuration

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
```

### Learning Rate Schedulers

```python
from torch.optim.lr_scheduler import StepLR

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    # Training loop
    scheduler.step()
```

Common Schedulers:

*   `StepLR`: Decays the learning rate by a factor every few steps.
*   `MultiStepLR`: Decays the learning rate at specified milestones.
*   `ExponentialLR`: Decays the learning rate exponentially.
*   `CosineAnnealingLR`: Uses a cosine annealing schedule.
*   `ReduceLROnPlateau`: Reduces the learning rate when a metric has stopped improving.
*   `CyclicLR`: Sets the learning rate cyclically.
*   `OneCycleLR`: Sets the learning rate according to the 1cycle policy.
*   `CosineAnnealingWarmRestarts`: Cosine annealing with warm restarts.

### Metrics

*   Accuracy
*   Precision
*   Recall
*   F1-Score
*   AUC (Area Under the Curve)
*   IoU (Intersection over Union)

## Training

### Training Pipeline

```
    ┌─────────────────────────────────────────┐
    │         Training Pipeline               │
    └───────────────┬─────────────────────────┘
                    │
                    ↓
    ┌───────────────────────────────┐
    │  1. Load Data (DataLoader)    │
    └──────────────┬────────────────┘
                   │
                   ↓
    ┌───────────────────────────────┐
    │  2. Zero Gradients            │
    │     optimizer.zero_grad()     │
    └──────────────┬────────────────┘
                   │
                   ↓
    ┌───────────────────────────────┐
    │  3. Forward Pass              │
    │     outputs = model(inputs)   │
    └──────────────┬────────────────┘
                   │
                   ↓
    ┌───────────────────────────────┐
    │  4. Compute Loss              │
    │     loss = criterion(...)     │
    └──────────────┬────────────────┘
                   │
                   ↓
    ┌───────────────────────────────┐
    │  5. Backward Pass             │
    │     loss.backward()           │
    └──────────────┬────────────────┘
                   │
                   ↓
    ┌───────────────────────────────┐
    │  6. Update Weights            │
    │     optimizer.step()          │
    └──────────────┬────────────────┘
                   │
                   └──→ Repeat for all batches
```

### Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Sample data
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
X_val = torch.randn(200, 784)
y_val = torch.randint(0, 10, (200,))

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss, optimizer
model = SimpleNet(784, 128, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # Calculate average training metrics
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = 100 * train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = 100 * val_correct / val_total
    
    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
```

### Custom Datasets and DataLoaders

```python
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: Input data (numpy array or list)
            labels: Target labels (numpy array or list)
            transform: Optional transform to be applied
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# CSV Dataset example
class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Assuming last column is label
        features = self.data.iloc[idx, :-1].values.astype('float32')
        label = self.data.iloc[idx, -1]
        
        if self.transform:
            features = self.transform(features)
        
        return torch.FloatTensor(features), torch.LongTensor([label])

# Create dataset and dataloader
dataset = CustomDataset(data, labels)

# DataLoader parameters
dataloader = DataLoader(
    dataset,
    batch_size=32,         # Number of samples per batch
    shuffle=True,          # Shuffle data at every epoch
    num_workers=4,         # Number of subprocesses for data loading
    pin_memory=True,       # Pin memory for faster data transfer to CUDA
    drop_last=False,       # Drop last incomplete batch
    persistent_workers=True # Keep workers alive between epochs
)

# Iterate through batches
for batch_idx, (inputs, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: inputs shape {inputs.shape}, labels shape {labels.shape}")
```

### Transforms

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4, pin_memory=True)
```

Common Augmentations:

*   `transforms.RandomHorizontalFlip`: Horizontally flips the image.
*   `transforms.RandomVerticalFlip`: Vertically flips the image.
*   `transforms.RandomRotation`: Rotates the image by a random angle.
*   `transforms.RandomAffine`: Applies random affine transformations.
*   `transforms.RandomPerspective`: Performs perspective transformation of the given image randomly with a given magnitude.
*   `transforms.RandomCrop`: Crops a random portion of the image.
*   `transforms.CenterCrop`: Crops the image from the center.
*   `transforms.ColorJitter`: Randomly changes the brightness, contrast, saturation, and hue of an image.
*   `transforms.RandomGrayscale`: Converts the image to grayscale with a certain probability.
*   `transforms.RandomErasing`: Randomly erases a rectangular region in the image.

### Mixed Precision Training

```
    ┌────────────────────────────────────┐
    │   Mixed Precision Training Flow    │
    └─────────────┬──────────────────────┘
                  │
                  ↓
    ┌────────────────────────────┐
    │  Forward Pass (FP16)       │
    │  with autocast()           │
    └──────────┬─────────────────┘
               │
               ↓
    ┌────────────────────────────┐
    │  Compute Loss (FP16)       │
    └──────────┬─────────────────┘
               │
               ↓
    ┌────────────────────────────┐
    │  Scale Loss                │
    │  scaler.scale(loss)        │
    └──────────┬─────────────────┘
               │
               ↓
    ┌────────────────────────────┐
    │  Backward Pass (FP32)      │
    │  scaled_loss.backward()    │
    └──────────┬─────────────────┘
               │
               ↓
    ┌────────────────────────────┐
    │  Unscale & Update          │
    │  scaler.step(optimizer)    │
    └────────────────────────────┘
```

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Initialize GradScaler
scaler = GradScaler()

# Training loop with mixed precision
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        
        # Unscale gradients and perform optimizer step
        scaler.step(optimizer)
        
        # Update scaler for next iteration
        scaler.update()

# Benefits:
# - 2-3x faster training
# - ~50% memory reduction
# - Maintains model accuracy
```

## Evaluation

```
    ┌────────────────────────────┐
    │   Evaluation Mode          │
    └────────┬───────────────────┘
             │
             ↓
    ┌─────────────────┐
    │  model.eval()   │──→ Disable dropout
    └────────┬────────┘     Freeze batch norm
             │
             ↓
    ┌─────────────────────┐
    │  torch.no_grad()    │──→ Disable gradient
    └────────┬────────────┘     computation
             │
             ↓
    ┌─────────────────────┐
    │  Forward Pass       │
    │  Compute Metrics    │
    └─────────────────────┘
```

```python
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item() * inputs.size(0)
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    return accuracy, avg_loss

# Quick evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
```

## Prediction and Inference

```python
import torch
import torch.nn.functional as F

# Single prediction
model.eval()
with torch.no_grad():
    # Prepare input
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # Get prediction
    output = model(input_tensor)
    
    # For classification
    probabilities = F.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)
    
    print(f'Predicted class: {predicted_class.item()}')
    print(f'Confidence: {confidence.item():.4f}')
    print(f'All probabilities: {probabilities[0].cpu().numpy()}')

# Batch prediction
def predict_batch(model, inputs, device):
    """Make predictions for a batch of inputs"""
    model.eval()
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
    
    return predictions.cpu().numpy(), confidences.cpu().numpy()

# Example usage
batch_inputs = torch.randn(10, 3, 224, 224)
predictions, confidences = predict_batch(model, batch_inputs, device)

for i, (pred, conf) in enumerate(zip(predictions, confidences)):
    print(f"Sample {i}: Class {pred}, Confidence {conf:.4f}")

# Top-k predictions
def get_top_k_predictions(model, input_tensor, k=5):
    """Get top-k predictions with probabilities"""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.to(device))
        probabilities = F.softmax(output, dim=1)
        top_probs, top_classes = torch.topk(probabilities, k, dim=1)
    
    return top_classes[0].cpu().numpy(), top_probs[0].cpu().numpy()

# Usage
input_tensor = torch.randn(1, 3, 224, 224)
classes, probs = get_top_k_predictions(model, input_tensor, k=5)

print("Top 5 predictions:")
for cls, prob in zip(classes, probs):
    print(f"  Class {cls}: {prob:.4f}")
```

## Saving and Loading Models

```
    ┌──────────────────────────────────┐
    │    Model Saving Strategies       │
    └────────┬─────────────────────────┘
             │
        ┌────┴─────┐
        │          │
        ↓          ↓
    ┌────────┐  ┌──────────────┐
    │ Full   │  │ State Dict   │
    │ Model  │  │ (Preferred)  │
    └────────┘  └──────────────┘
```

### Save Model State Dictionary (Recommended)

```python
import torch
import os

# Save only state dictionary (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Save checkpoint with additional info
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'accuracy': accuracy
}
torch.save(checkpoint, 'checkpoint.pth')

# Save best model
best_acc = 0.0
if val_acc > best_acc:
    best_acc = val_acc
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, 'best_model.pth')
```

### Load Model State Dictionary

```python
# Load state dictionary
model = SimpleNet(784, 128, 10)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Resume training
model.train()

# Load for inference on different device
device = torch.device('cpu')
model = SimpleNet(784, 128, 10)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()
```

### Save Entire Model (Not Recommended)

```python
# Save entire model
torch.save(model, 'full_model.pth')

# Load entire model
model = torch.load('full_model.pth')
model.eval()

# Note: This is less flexible and may break with PyTorch version changes
```

### Model Versioning and Management

```python
import os
from datetime import datetime

def save_model_with_metadata(model, optimizer, epoch, metrics, save_dir='models'):
    """Save model with comprehensive metadata"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'model_epoch{epoch}_{timestamp}.pth'
    filepath = os.path.join(save_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': timestamp,
        'pytorch_version': torch.__version__
    }, filepath)
    
    print(f"Model saved to {filepath}")
    return filepath

# Usage
metrics = {'train_loss': 0.5, 'train_acc': 0.85, 'val_loss': 0.6, 'val_acc': 0.82}
save_model_with_metadata(model, optimizer, epoch=10, metrics=metrics)
```

## CUDA (GPU Support)

### Check CUDA Availability

```python
torch.cuda.is_available()
```

### Set Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### Move Tensors to GPU

```python
tensor = tensor.to(device)
```

### CUDA Best Practices

*   Use pinned memory for data transfer: `torch.utils.data.DataLoader(..., pin_memory=True)`
*   Use asynchronous data transfer: `torch.cuda.Stream()`
*   Use mixed precision training: `torch.cuda.amp.autocast()` and `torch.cuda.amp.GradScaler()`
*   Use `torch.backends.cudnn.benchmark = True` for faster convolutions when input sizes are fixed.

## Distributed Training

```
    ┌─────────────────────────────────────┐
    │   Distributed Training Methods        │
    └───────────────┬─────────────────────┘
                    │
        ┌───────────┴──────────┐
        │                       │
        ↓                       ↓
    ┌────────────┐       ┌───────────────┐
    │DataParallel│       │DistributedData│
    │    (DP)    │       │Parallel (DDP) │
    │            │       │                │
    │ • Single   │       │ • Multi-node │
    │   node     │       │ • Faster     │
    │ • Easier   │       │ • Scalable   │
    └────────────┘       └───────────────┘
```

### DataParallel (Simple but limited)

```python
import torch
import torch.nn as nn

# Check available GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    
    # Wrap model with DataParallel
    model = nn.DataParallel(model)
    model.to('cuda')
    
    # Training works the same way
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Note: DataParallel splits batches across GPUs but has limitations:
# - Single process (GIL bottleneck)
# - Slower than DDP
# - Uneven GPU utilization
```

### DistributedDataParallel (Recommended)

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',      # Use 'gloo' for CPU, 'nccl' for GPU
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()

def train(rank, world_size, epochs):
    """Training function for each process"""
    print(f"Running DDP on rank {rank}")
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = SimpleNet(784, 128, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create dataset and sampler
    train_dataset = YourDataset()
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(rank)
    
    # Training loop
    for epoch in range(epochs):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        ddp_model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss across all processes
        avg_loss = epoch_loss / len(train_loader)
        
        if rank == 0:  # Only print from main process
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint from main process
        if rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': ddp_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
    
    cleanup()

def main():
    """Main function to spawn processes"""
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Need at least 2 GPUs for DDP")
        return
    
    print(f"Training on {world_size} GPUs")
    
    # Spawn processes
    mp.spawn(
        train,
        args=(world_size, 10),  # world_size, epochs
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```

### Multi-Node DDP

```python
import torch.distributed as dist

# On each node, set these environment variables:
# MASTER_ADDR: IP address of rank 0 node
# MASTER_PORT: Free port on rank 0 node
# WORLD_SIZE: Total number of processes across all nodes
# RANK: Global rank of this process

def setup_multinode(rank, world_size):
    """Setup for multi-node training"""
    # These should be set via environment variables
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )

# Launch on node 0 (4 GPUs):
# MASTER_ADDR=node0 MASTER_PORT=12355 WORLD_SIZE=8 RANK=0 python train.py

# Launch on node 1 (4 GPUs):
# MASTER_ADDR=node0 MASTER_PORT=12355 WORLD_SIZE=8 RANK=4 python train.py
```

### DDP with Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_amp(rank, world_size):
    setup(rank, world_size)
    
    model = SimpleNet(784, 128, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(rank), labels.to(rank)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    cleanup()
```

## Autograd (Automatic Differentiation)

### Computational Graph

```
    ┌─────────────────────────────────┐
    │   Computational Graph          │
    └───────────────┬─────────────────┘
                    │
                    ↓ Forward
    ┌───────┐     ┌───────┐     ┌───────┐
    │   x   │─────→│  y=x+2│─────→│z=y*y*3│
    │(leaf) │     │       │     │       │
    └───────┘     └───────┘     └───┬────┘
        ↑                            │
        │                            ↓
        │       Backward         ┌────────┐
        └────────────────────────│out.mean│
                                    └────────┘
```

### Tracking Gradients

```python
import torch

# Create tensor with gradient tracking
x = torch.randn(3, requires_grad=True)
print(f"x: {x}")
print(f"requires_grad: {x.requires_grad}")

# Perform operations
y = x + 2
z = y * y * 3
out = z.mean()

print(f"\nout: {out}")

# Compute gradients
out.backward()

# Access gradients
print(f"\nx.grad: {x.grad}")
print(f"\nGradient of out with respect to x: {x.grad}")

# Gradient accumulation (multiple backward passes)
x = torch.randn(3, requires_grad=True)
for i in range(3):
    y = (x ** 2).sum()
    y.backward()
    print(f"Iteration {i+1}, x.grad: {x.grad}")

# Zero gradients manually
x.grad.zero_()
print(f"After zeroing: {x.grad}")
```

### Controlling Gradient Tracking

```python
# Disable gradient tracking (for inference)
with torch.no_grad():
    y = x + 2
    z = y * y
    print(f"z.requires_grad: {z.requires_grad}")  # False

# Alternative: use inference mode (faster than no_grad)
with torch.inference_mode():
    y = x + 2
    z = y * y

# Temporarily disable gradient
x = torch.randn(3, requires_grad=True)
with torch.set_grad_enabled(False):
    y = x * 2

# Detach tensor from computation graph
x = torch.randn(3, requires_grad=True)
y = x.detach()  # y shares data with x but has no gradient
z = x.detach().clone()  # Create independent copy without gradient

print(f"x.requires_grad: {x.requires_grad}")  # True
print(f"y.requires_grad: {y.requires_grad}")  # False
print(f"z.requires_grad: {z.requires_grad}")  # False
```

### Computing Higher-Order Derivatives

```python
# Second derivative
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# First derivative
first_grad = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"First derivative (3x^2): {first_grad}")  # 12.0

# Second derivative
second_grad = torch.autograd.grad(first_grad, x)[0]
print(f"Second derivative (6x): {second_grad}")  # 12.0
```

### Gradient for Non-Scalar Outputs

```python
# For non-scalar outputs, provide gradient argument
x = torch.randn(3, requires_grad=True)
y = x * 2

# Create gradient tensor matching y's shape
grad_output = torch.ones_like(y)
y.backward(grad_output)
print(f"x.grad: {x.grad}")
```

### Custom Autograd Functions

```python
import torch

class CustomReLU(torch.autograd.Function):
    """
    Custom ReLU implementation with autograd support
    """
    @staticmethod
    def forward(ctx, input):
        # Save input for backward pass
        ctx.save_for_backward(input)
        # Apply ReLU: max(0, x)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        input, = ctx.saved_tensors
        # Compute gradient
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0  # Gradient is 0 where input < 0
        return grad_input

# Use custom function
custom_relu = CustomReLU.apply

x = torch.randn(5, requires_grad=True)
y = custom_relu(x)
loss = y.sum()
loss.backward()
print(f"Input: {x}")
print(f"Output: {y}")
print(f"Gradient: {x.grad}")

# Custom function with multiple inputs/outputs
class CustomMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, constant):
        ctx.save_for_backward(input1, input2)
        ctx.constant = constant
        return input1 * input2 * constant
    
    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        constant = ctx.constant
        
        grad_input1 = grad_output * input2 * constant
        grad_input2 = grad_output * input1 * constant
        # Return gradient for each input (constant has no gradient)
        return grad_input1, grad_input2, None

# Usage
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)
c = CustomMultiply.apply(a, b, 5.0)
c.backward()
print(f"a.grad: {a.grad}")  # 15.0 (3 * 5)
print(f"b.grad: {b.grad}")  # 10.0 (2 * 5)
```

## Data Augmentation

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
```

Common Augmentations:

*   `transforms.RandomHorizontalFlip`: Horizontally flips the image.
*   `transforms.RandomVerticalFlip`: Vertically flips the image.
*   `transforms.RandomRotation`: Rotates the image by a random angle.
*   `transforms.RandomAffine`: Applies random affine transformations.
*   `transforms.RandomPerspective`: Performs perspective transformation of the given image randomly with a given magnitude.
*   `transforms.RandomCrop`: Crops a random portion of the image.
*   `transforms.CenterCrop`: Crops the image from the center.
*   `transforms.ColorJitter`: Randomly changes the brightness, contrast, saturation, and hue of an image.
*   `transforms.RandomGrayscale`: Converts the image to grayscale with a certain probability.
*   `transforms.RandomErasing`: Randomly erases a rectangular region in the image.
*   `transforms.RandomResizedCrop`: Crops a random portion of the image and resizes it.

## Learning Rate Schedulers

```
    Learning Rate Schedules Visualization
    
    StepLR:                CosineAnnealing:
    lr │                    lr │
       ├─────┐               ├───┐
       │     └─────┐          │    ╮╮
       │           └────       │    ╲ ╯┐
       │                       │       ╯┐╲
       └──────────────────     └───────────────
              epochs                    epochs
```

### Common Schedulers

```python
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

model = SimpleNet(784, 128, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 1. StepLR: Decay LR by gamma every step_size epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# LR: 0.001 → 0.0001 (epoch 30) → 0.00001 (epoch 60)

# 2. MultiStepLR: Decay at specific milestones
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

# 3. ExponentialLR: Exponential decay
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# LR = initial_lr * (gamma ** epoch)

# 4. CosineAnnealingLR: Cosine annealing
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# 5. ReduceLROnPlateau: Reduce when metric plateaus
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',        # 'min' for loss, 'max' for accuracy
    factor=0.1,        # Multiply LR by factor
    patience=10,       # Wait 10 epochs before reducing
    verbose=True,
    min_lr=1e-6
)

# 6. CyclicLR: Cycle between two boundaries
scheduler = lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.001,
    max_lr=0.01,
    step_size_up=2000,
    mode='triangular'
)

# 7. OneCycleLR: 1cycle policy (very effective)
scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=100,
    steps_per_epoch=len(train_loader)
)

# 8. CosineAnnealingWarmRestarts: Cosine with restarts
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # First restart after 10 epochs
    T_mult=2,    # Double period after each restart
    eta_min=0
)

# 9. LambdaLR: Custom schedule
scheduler = lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: 0.95 ** epoch
)
```

### Usage in Training Loop

```python
# For most schedulers
for epoch in range(num_epochs):
    # Training
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Step scheduler after each epoch
    scheduler.step()
    
    # Print current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, LR: {current_lr:.6f}")

# For ReduceLROnPlateau (needs validation metric)
for epoch in range(num_epochs):
    # Training
    train_loss = train_one_epoch(model, train_loader)
    
    # Validation
    val_loss = validate(model, val_loader)
    
    # Step with validation loss
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# For OneCycleLR (step after each batch)
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step after each batch

# Get learning rate history
lr_history = []
for epoch in range(100):
    lr_history.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# Plot learning rate schedule
import matplotlib.pyplot as plt
plt.plot(lr_history)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()
```

## TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")

# Log scalar values
writer.add_scalar('Loss/train', loss.item(), epoch)
writer.add_scalar('Accuracy/train', accuracy, epoch)

# Log model graph
writer.add_graph(model, images)

# Log images
writer.add_image('Image', img_grid, epoch)

# Log histograms
writer.add_histogram('fc1.weight', model.fc1.weight, epoch)

# Log embeddings
writer.add_embedding(features, metadata=labels, tag='my_embedding')

writer.close()
```

Run TensorBoard:

```bash
tensorboard --logdir=runs
```

## ONNX Export

```
    ┌─────────────────────────────────┐
    │   ONNX Export Pipeline          │
    └─────────────┬───────────────────┘
                 │
                 ↓
    ┌─────────────────────────────────┐
    │  PyTorch Model (.pth)         │
    └─────────────┬───────────────────┘
                 │
                 ↓
    ┌─────────────────────────────────┐
    │  torch.onnx.export()          │
    └─────────────┬───────────────────┘
                 │
                 ↓
    ┌─────────────────────────────────┐
    │  ONNX Model (.onnx)           │
    │  (Cross-platform inference)   │
    └─────────────────────────────────┘
```

```python
import torch
import onnx
import onnxruntime as ort
import numpy as np

# Export model to ONNX
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Export with dynamic axes for variable batch size
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,          # ONNX opset version
    do_constant_folding=True,  # Optimize constant folding
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},   # Variable batch size
        'output': {0: 'batch_size'}
    },
    verbose=False
)

print("ONNX model exported successfully!")

# Verify ONNX model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Print model info
print(f"\nONNX Model Info:")
print(f"Inputs: {[(i.name, i.type) for i in onnx_model.graph.input]}")
print(f"Outputs: {[(o.name, o.type) for o in onnx_model.graph.output]}")

# Test ONNX model with ONNX Runtime
ort_session = ort.InferenceSession("model.onnx")

# Prepare input
test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
ort_inputs = {ort_session.get_inputs()[0].name: test_input}
ort_outputs = ort_session.run(None, ort_inputs)

print(f"\nONNX Runtime output shape: {ort_outputs[0].shape}")

# Compare PyTorch and ONNX outputs
with torch.no_grad():
    pytorch_output = model(torch.from_numpy(test_input).to(device))
    pytorch_output = pytorch_output.cpu().numpy()

difference = np.abs(pytorch_output - ort_outputs[0])
print(f"Max difference between PyTorch and ONNX: {np.max(difference)}")
print(f"Mean difference: {np.mean(difference)}")

# Optimize ONNX model
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization (INT8)
quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QInt8
)
print("\nQuantized ONNX model created!")

# Compare model sizes
import os
original_size = os.path.getsize("model.onnx") / (1024 * 1024)
quantized_size = os.path.getsize("model_quantized.onnx") / (1024 * 1024)
print(f"Original model size: {original_size:.2f} MB")
print(f"Quantized model size: {quantized_size:.2f} MB")
print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")
```

## TorchScript

```
    ┌──────────────────────────────┐
    │   TorchScript Methods        │
    └──────────┬───────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ↓             ↓
    ┌────────┐   ┌──────────┐
    │ Tracing│   │ Scripting│
    │ (Record │   │ (Analyze │
    │  ops)   │   │  source) │
    └────────┘   └──────────┘
```

### Tracing (Recommended for most models)

```python
import torch

# Set model to evaluation mode
model.eval()

# Create example input
example_input = torch.rand(1, 3, 224, 224).to(device)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save traced model
traced_model.save("model_traced.pt")
print("Traced model saved!")

# Load traced model
loaded_model = torch.jit.load("model_traced.pt")
loaded_model.eval()

# Test traced model
with torch.no_grad():
    test_input = torch.randn(1, 3, 224, 224).to(device)
    output = loaded_model(test_input)
    print(f"Output shape: {output.shape}")

# Optimize for mobile deployment
optimized_model = torch.jit.optimize_for_inference(traced_model)
optimized_model.save("model_optimized.pt")
```

### Scripting (For models with control flow)

```python
import torch
import torch.nn as nn

# Script entire model
class ScriptableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        
        # Control flow is preserved
        if x.sum() > 0:
            x = self.fc2(x)
        else:
            x = torch.zeros(x.size(0), 10)
        
        return x

model = ScriptableModel()
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Script individual functions
@torch.jit.script
def custom_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Custom function with type annotations"""
    if x.sum() > y.sum():
        return x + y
    else:
        return x - y

# Use scripted function
result = custom_function(torch.randn(3, 4), torch.randn(3, 4))

# Combine tracing and scripting
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 10)
    
    @torch.jit.script_method
    def forward(self, x):
        # This method will be scripted
        x = torch.relu(self.conv(x))
        x = x.mean(dim=[2, 3])
        return self.fc(x)

# Mobile deployment
from torch.utils.mobile_optimizer import optimize_for_mobile

model.eval()
traced = torch.jit.trace(model, example_input)
optimized = optimize_for_mobile(traced)
optimized._save_for_lite_interpreter("model_mobile.ptl")
print("Mobile model saved!")
```

### Performance Comparison

```python
import time
import torch

# Benchmark function
def benchmark(model, input_tensor, num_runs=100):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
        
        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            _ = model(input_tensor)
        end = time.time()
    
    avg_time = (end - start) / num_runs * 1000  # ms
    return avg_time

# Compare models
input_tensor = torch.randn(1, 3, 224, 224).to(device)

original_time = benchmark(model, input_tensor)
traced_time = benchmark(traced_model, input_tensor)

print(f"Original model: {original_time:.3f} ms")
print(f"Traced model: {traced_time:.3f} ms")
print(f"Speedup: {original_time/traced_time:.2f}x")
```

## Deployment

```
    ┌───────────────────────────────────┐
    │     Deployment Pipeline            │
    └──────────────┬────────────────────┘
                   │
                   ↓
    ┌───────────────────────────────────┐
    │  1. Train Model                   │
    └──────────────┬────────────────────┘
                   │
                   ↓
    ┌───────────────────────────────────┐
    │  2. Save Model (.pth/.onnx)      │
    └──────────────┬────────────────────┘
                   │
                   ↓
    ┌───────────────────────────────────┐
    │  3. Create API (Flask/FastAPI)   │
    └──────────────┬────────────────────┘
                   │
                   ↓
    ┌───────────────────────────────────┐
    │  4. Containerize (Docker)        │
    └──────────────┬────────────────────┘
                   │
                   ↓
    ┌───────────────────────────────────┐
    │  5. Deploy (Cloud/On-Premise)    │
    └───────────────────────────────────┘
```

### Serving with Flask

```python
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet(784, 128, 10)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.to(device)
model.eval()
logger.info(f"Model loaded successfully on {device}")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def transform_image(image_bytes):
    """Transform image bytes to tensor"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return transform(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error transforming image: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'device': str(device)})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read and transform image
        image_bytes = request.files['image'].read()
        img_tensor = transform_image(image_bytes)
        img_tensor = img_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Get top 3 predictions
            top3_probs, top3_classes = torch.topk(probabilities, 3, dim=1)
        
        # Prepare response
        response = {
            'predicted_class': predicted_class.item(),
            'confidence': confidence.item(),
            'top3_predictions': [
                {
                    'class': top3_classes[0][i].item(),
                    'probability': top3_probs[0][i].item()
                }
                for i in range(3)
            ]
        }
        
        logger.info(f"Prediction: {response}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    try:
        images = request.files.getlist('images')
        batch_tensors = []
        
        for img in images:
            img_bytes = img.read()
            img_tensor = transform_image(img_bytes)
            batch_tensors.append(img_tensor)
        
        # Stack tensors into batch
        batch = torch.cat(batch_tensors, dim=0).to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(batch)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
        
        # Prepare response
        results = [
            {
                'index': i,
                'predicted_class': predictions[i].item(),
                'confidence': confidences[i].item()
            }
            for i in range(len(predictions))
        ]
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Test API

```python
import requests

# Test single prediction
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f}
    )
    print(response.json())

# Test batch prediction
files = [
    ('images', open('image1.jpg', 'rb')),
    ('images', open('image2.jpg', 'rb')),
    ('images', open('image3.jpg', 'rb'))
]
response = requests.post(
    'http://localhost:5000/batch_predict',
    files=files
)
print(response.json())
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application
COPY model_weights.pth .
COPY app.py .
COPY model.py .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV MODEL_PATH=model_weights.pth

# Run application
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  pytorch-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MODEL_PATH=/app/model_weights.pth
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Build and run Docker container
docker build -t pytorch-api:latest .
docker run -p 5000:5000 pytorch-api:latest

# Or use docker-compose
docker-compose up -d

# Scale service
docker-compose up -d --scale pytorch-api=3
```

### Serving with TorchServe

```bash
# Install TorchServe
pip install torchserve torch-model-archiver torch-workflow-archiver

# Create handler.py
cat > handler.py << 'EOF'
import torch
import torch.nn.functional as F
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io

class ImageClassifier(BaseHandler):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = Image.open(io.BytesIO(image))
            image = self.transform(image)
            images.append(image)
        return torch.stack(images).to(self.device)
    
    def postprocess(self, inference_output):
        probabilities = F.softmax(inference_output, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
        return [
            {
                "class": pred.item(),
                "confidence": conf.item()
            }
            for pred, conf in zip(predictions, confidences)
        ]
EOF

# Create model archive
torch-model-archiver \
  --model-name image_classifier \
  --version 1.0 \
  --model-file model.py \
  --serialized-file model_weights.pth \
  --handler handler.py \
  --extra-files index_to_name.json

# Create model store directory
mkdir -p model_store
mv image_classifier.mar model_store/

# Start TorchServe
torchserve --start \
  --model-store model_store \
  --models image_classifier=image_classifier.mar \
  --ncs

# Test inference
curl -X POST http://localhost:8080/predictions/image_classifier \
  -T test_image.jpg

# Management API
curl http://localhost:8081/models

# Stop TorchServe
torchserve --stop
```

## Distributed Training

### DataParallel

```python
model = nn.DataParallel(model)
```

### DistributedDataParallel (DDP)

```python
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    model = Net().to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Training loop
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

### Gradient Clipping

```
    ┌────────────────────────────┐
    │   Gradient Clipping Flow     │
    └───────────┬────────────────┘
                │
                ↓
    ┌────────────────────────────┐
    │  Compute Gradients          │
    │  loss.backward()            │
    └───────────┬────────────────┘
                │
                ↓
    ┌────────────────────────────┐
    │  Check Gradient Norm        │
    └───────────┬────────────────┘
                │
        ┌───────┴───────┐
        │               │
        ↓ >max_norm    ↓ <=max_norm
    ┌─────────┐     ┌─────────┐
    │  Scale  │     │   Use   │
    │Gradients│     │  As-Is  │
    └────┬────┘     └────┬────┘
         └────────┬───────┘
                  │
                  ↓
    ┌────────────────────────────┐
    │  optimizer.step()           │
    └────────────────────────────┘
```

```python
import torch
import torch.nn as nn

# Gradient clipping by norm (most common)
for inputs, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

# Gradient clipping by value
for inputs, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    
    # Clip each gradient to [-clip_value, clip_value]
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
    
    optimizer.step()

# Monitor gradient norms
def get_gradient_norm(model):
    """Calculate total gradient norm"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

# Usage in training loop
for inputs, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    
    # Monitor gradients
    grad_norm = get_gradient_norm(model)
    print(f"Gradient norm: {grad_norm:.4f}")
    
    # Clip if needed
    if grad_norm > 5.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    
    optimizer.step()
```

### Weight Decay

Weight decay (L2 regularization) is often included directly in the optimizer:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

### Early Stopping

```python
import torch
import numpy as np

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=7, min_delta=0.0, verbose=True, path='best_model.pth'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
            path: Path to save best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Usage in training loop
early_stopping = EarlyStopping(patience=10, verbose=True)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Check early stopping
    early_stopping(avg_val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
```

### Learning Rate Finders

```python
# Requires a separate library like `torch_lr_finder`
from torch_lr_finder import LRFinder

optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=0.01)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, end_lr=1, num_iter=100)
lr_finder.plot()
lr_finder.reset()
```

### Gradient Accumulation

Gradient accumulation allows you to simulate larger batch sizes when you are limited by GPU memory. It works by accumulating gradients over multiple smaller batches before performing the optimization step.

```
    Effective Batch Size = batch_size × accumulation_steps
    
    ┌───────────────────────────────────────┐
    │    Gradient Accumulation Process       │
    └─────────────────┬─────────────────────┘
                     │
    ┌─────────────┴─────────────┐
    │  optimizer.zero_grad()      │
    └─────────────┬──────────────┘
                  │
                  ↓
    ┌────────────────────────────┐
    │  Batch 1: Forward + Back   │
    │  (gradients accumulate)    │
    └─────────────┬───────────────┘
                  │
                  ↓
    ┌────────────────────────────┐
    │  Batch 2: Forward + Back   │
    │  (gradients accumulate)    │
    └─────────────┬───────────────┘
                  │
                  ↓
    ┌────────────────────────────┐
    │  Batch N: Forward + Back   │
    │  (gradients accumulate)    │
    └─────────────┬───────────────┘
                  │
                  ↓
    ┌────────────────────────────┐
    │  optimizer.step()          │
    │  (update weights)          │
    └────────────────────────────┘
```

```python
import torch

# Configuration
accumulation_steps = 4  # Simulate 4x larger batch size

# Training loop with gradient accumulation
model.train()
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(train_loader):
    inputs, labels = inputs.to(device), labels.to(device)
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Normalize loss (important!)
    loss = loss / accumulation_steps
    
    # Backward pass (gradients accumulate)
    loss.backward()
    
    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Step {(i+1)//accumulation_steps}: Loss = {loss.item() * accumulation_steps:.4f}")

# Handle remaining batches if dataset size not divisible by accumulation_steps
if (i + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()

# Example: Compare memory usage
# Without accumulation: batch_size = 64 → GPU memory = X GB
# With accumulation: batch_size = 16, accumulation_steps = 4 → GPU memory = X/4 GB
# But effective batch size is still 64
```

## Common Neural Network Architectures

### Convolutional Neural Network (CNN)

```
    ┌────────────────────────────────────┐
    │      CNN Architecture Flow          │
    └────────────────┬───────────────────┘
                     │
    Input Image      ↓
    (3, 224, 224)    │
                     │
    ┌────────────────┴──────────────────┐
    │  Conv2d + BatchNorm + ReLU         │
    │  (64 filters)                      │
    └────────────────┬──────────────────┘
                     │
                     ↓
    ┌──────────────────────────────────┐
    │  MaxPool2d (2x2)                  │
    └────────────────┬─────────────────┘
                     │
                     ↓ (Repeat blocks)
    ┌──────────────────────────────────┐
    │  Flatten                          │
    └────────────────┬─────────────────┘
                     │
                     ↓
    ┌──────────────────────────────────┐
    │  Fully Connected Layers           │
    └────────────────┬─────────────────┘
                     │
                     ↓
    ┌──────────────────────────────────┐
    │  Output (num_classes)             │
    └──────────────────────────────────┘
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Usage
model = SimpleCNN(num_classes=10)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

### ResNet-style Residual Block

```python
class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
```

### Recurrent Neural Network (LSTM)

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        # *2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM output
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch_size, seq_length, hidden_dim * 2)
        
        # Use last output or pooling
        output = lstm_out[:, -1, :]  # Last time step
        # Or use mean pooling: output = torch.mean(lstm_out, dim=1)
        
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

# Usage
model = LSTMClassifier(
    vocab_size=10000,
    embedding_dim=300,
    hidden_dim=256,
    num_classes=5
)
```

### Transformer Encoder

```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, src, src_mask=None):
        # src: (batch_size, seq_length)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer expects (seq_length, batch_size, d_model)
        src = src.transpose(0, 1)
        
        output = self.transformer_encoder(src, src_mask)
        
        # Use CLS token or mean pooling
        output = output.mean(dim=0)  # Average over sequence
        output = self.fc(output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

### Autoencoder

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # For normalized inputs [0, 1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
```

## Common Issues and Debugging

```
    ┌─────────────────────────────────────┐
    │      Debugging Workflow              │
    └────────────────┬─────────────────────┘
                     │
                     ↓
    ┌─────────────────────────────────────┐
    │  1. Check Tensor Shapes/Devices   │
    └────────────────┬─────────────────────┘
                     │
                     ↓
    ┌─────────────────────────────────────┐
    │  2. Verify Gradients Flow         │
    └────────────────┬─────────────────────┘
                     │
                     ↓
    ┌─────────────────────────────────────┐
    │  3. Monitor Loss/Metrics          │
    └────────────────┬─────────────────────┘
                     │
                     ↓
    ┌─────────────────────────────────────┐
    │  4. Profile Performance           │
    └─────────────────────────────────────┘
```

### Debugging Tools

```python
import torch
import torch.nn as nn

# 1. Register hooks to monitor gradients
def print_grad_hook(name):
    def hook(grad):
        print(f"{name} gradient: {grad.norm():.4f}")
    return hook

# Register hooks
for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(print_grad_hook(name))

# 2. Check for NaN/Inf values
def check_nan_inf(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

# 3. Print model summary
def print_model_summary(model, input_size):
    from torchsummary import summary
    summary(model, input_size)

# Usage
print_model_summary(model, (3, 224, 224))

# 4. Visualize gradient flow
def plot_grad_flow(named_parameters):
    import matplotlib.pyplot as plt
    
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    
    plt.bar(range(len(max_grads)), max_grads, alpha=0.5, label="max")
    plt.bar(range(len(ave_grads)), ave_grads, alpha=0.5, label="mean")
    plt.xticks(range(len(ave_grads)), layers, rotation="vertical")
    plt.legend()
    plt.show()

# 5. Memory profiling
def profile_memory():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 6. Torch profiler
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Common Issues and Solutions

*   **CUDA Out of Memory:**
    - Reduce batch size
    - Use mixed precision training (`torch.cuda.amp`)
    - Use gradient checkpointing
    - Clear cache: `torch.cuda.empty_cache()`
    - Use smaller model or fewer layers

*   **NaN/Inf Losses:**
    - Reduce learning rate
    - Use gradient clipping: `torch.nn.utils.clip_grad_norm_()`
    - Check for division by zero
    - Normalize input data
    - Use stable loss functions (e.g., `BCEWithLogitsLoss` instead of `BCELoss`)

*   **Slow Training:**
    - Profile code to find bottlenecks
    - Use GPU acceleration
    - Increase `num_workers` in DataLoader
    - Use `pin_memory=True` in DataLoader
    - Enable `torch.backends.cudnn.benchmark = True` for fixed input sizes
    - Use mixed precision training

*   **Overfitting:**
    - Add dropout layers
    - Use data augmentation
    - Implement early stopping
    - Reduce model complexity
    - Add L2 regularization (weight decay)
    - Increase training data

*   **Underfitting:**
    - Increase model capacity (more layers/neurons)
    - Train for more epochs
    - Reduce regularization
    - Check if data preprocessing is correct
    - Use better optimizer (Adam instead of SGD)

*   **Incorrect Tensor Shapes:**
    ```python
    # Debug tensor shapes
    print(f"Input shape: {x.shape}")
    print(f"Expected shape: (batch, channels, height, width)")
    
    # Use assertions
    assert x.shape[1] == 3, f"Expected 3 channels, got {x.shape[1]}"
    ```

*   **Device Mismatch:**
    ```python
    # Check device
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {inputs.device}")
    
    # Move everything to same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = inputs.to(device)
    ```

*   **Gradients Not Flowing:**
    ```python
    # Check requires_grad
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, grad={param.grad is not None}")
    
    # Check for detached tensors
    # Make sure not using .detach() or .data unintentionally
    ```

*   **Dead Neurons (ReLU):**
    - Use Leaky ReLU: `nn.LeakyReLU(0.01)`
    - Use ELU: `nn.ELU()`
    - Reduce learning rate
    - Better weight initialization: `nn.init.kaiming_normal_()`

*   **Data Loading Bottlenecks:**
    ```python
    # Increase workers and use prefetching
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    ```

## Best Practices

### Code Organization

```python
# Organize code into modular components

# model.py
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Forward pass
        return x

# dataset.py
class MyDataset(Dataset):
    def __init__(self, data_path):
        # Load data
        pass
    
    def __getitem__(self, idx):
        # Return sample
        pass

# train.py
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# config.py
class Config:
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Performance Optimization

```python
# 1. Enable cuDNN autotuner for fixed input sizes
torch.backends.cudnn.benchmark = True

# 2. Use appropriate data types
# Use fp16 for training when possible
from torch.cuda.amp import autocast, GradScaler

# 3. Optimize DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,           # Multi-process data loading
    pin_memory=True,         # Fast data transfer to GPU
    persistent_workers=True, # Keep workers alive
    prefetch_factor=2        # Prefetch batches
)

# 4. Use in-place operations when possible
x.add_(1)  # In-place
x.relu_()  # In-place

# 5. Avoid unnecessary CPU-GPU transfers
# Keep data on GPU as much as possible

# 6. Use torch.no_grad() for inference
with torch.no_grad():
    predictions = model(inputs)

# 7. Clear unused variables
del intermediate_tensor
torch.cuda.empty_cache()
```

### Reproducibility

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Model Initialization

```python
import torch.nn as nn

def init_weights(m):
    """Initialize model weights"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# Apply initialization
model.apply(init_weights)
```

### Logging and Monitoring

```python
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log training progress
logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# Use TensorBoard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment1')
for epoch in range(num_epochs):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
writer.close()
```

### Model Deployment Checklist

```python
# 1. Model validation
model.eval()
with torch.no_grad():
    test_acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")

# 2. Save model with metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'test_accuracy': test_acc,
    'config': config,
    'pytorch_version': torch.__version__
}, 'model_final.pth')

# 3. Export to ONNX for production
torch.onnx.export(model, dummy_input, 'model.onnx')

# 4. Test exported model
ort_session = ort.InferenceSession('model.onnx')

# 5. Create API endpoint
# See deployment section above

# 6. Containerize with Docker
# See Docker section above

# 7. Set up monitoring and logging
# Track inference time, memory usage, error rates

# 8. Implement versioning
# Use model registry (MLflow, DVC, etc.)
```

### Development Tips

*   **Virtual Environments:** Use `conda` or `venv` to isolate dependencies
*   **Code Style:** Follow PEP 8, use `black` for formatting
*   **Version Control:** Use Git, commit frequently with meaningful messages
*   **Testing:** Write unit tests for data loading, model forward pass, etc.
*   **Documentation:** Add docstrings to classes and functions
*   **GPU Memory:** Monitor with `nvidia-smi` or `torch.cuda.memory_summary()`
*   **Hyperparameter Tuning:** Use Optuna, Ray Tune, or Weights & Biases
*   **Model Compression:** Quantization, pruning, knowledge distillation
*   **Regular Updates:** Keep PyTorch and dependencies up to date
*   **Experiment Tracking:** Use MLflow, Weights & Biases, or TensorBoard

## Quick Reference

### Essential Operations

```python
# Tensor creation
x = torch.tensor([1, 2, 3])
x = torch.zeros(3, 4)
x = torch.randn(3, 4)

# Tensor operations
y = x.view(12)           # Reshape
y = x.permute(1, 0)      # Transpose
y = x.unsqueeze(0)       # Add dimension
y = x.squeeze()          # Remove dimensions of size 1

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)

# Gradient tracking
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # dy/dx

# No gradient context
with torch.no_grad():
    y = x * 2
```

### Training Template

```python
# Setup
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            # Calculate metrics
```

### Common Layer Patterns

```python
# Conv block
nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2)
)

# Dense block
nn.Sequential(
    nn.Linear(in_features, out_features),
    nn.ReLU(),
    nn.Dropout(0.5)
)

# Residual connection
out = F.relu(self.conv(x))
out = out + x  # Skip connection
```

### Useful Commands

```python
# Model info
print(model)
total_params = sum(p.numel() for p in model.parameters())

# Save/Load
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# Learning rate
for param_group in optimizer.param_groups:
    print(param_group['lr'])
    param_group['lr'] = 0.0001  # Update LR

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# GPU memory
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

### Performance Tips

```python
# Speed up training
torch.backends.cudnn.benchmark = True

# Mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# DataLoader optimization
DataLoader(dataset, batch_size=32, num_workers=4, 
          pin_memory=True, persistent_workers=True)
```

---

**Happy PyTorch Coding!** 🔥
