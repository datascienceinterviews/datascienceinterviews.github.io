
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
pip install torch torchvision torchaudio
```

For CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version. Check the [PyTorch website](https://pytorch.org/get-started/locally/) for the most up-to-date installation instructions.

### Importing PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
```

## Tensors

### Creating Tensors

From a List:

```python
data = [1, 2, 3, 4, 5]
tensor = torch.tensor(data)
```

From a NumPy Array:

```python
import numpy as np

data = np.array([1, 2, 3, 4, 5])
tensor = torch.from_numpy(data)
```

Zeros and Ones:

```python
zeros = torch.zeros(size=(3, 4))
ones = torch.ones(size=(3, 4))
```

Full (fill with a specific value):

```python
full = torch.full(size=(3, 4), fill_value=7)
```

Ranges:

```python
arange = torch.arange(start=0, end=10, step=2) # 0, 2, 4, 6, 8
linspace = torch.linspace(start=0, end=1, steps=5) # 0.0, 0.25, 0.5, 0.75, 1.0
```

Random Numbers:

```python
rand = torch.rand(size=(3, 4))  # Uniform distribution [0, 1)
randn = torch.randn(size=(3, 4)) # Standard normal distribution
randint = torch.randint(low=0, high=10, size=(3, 4)) # Integer values
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

Arithmetic:

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

c = a + b       # Element-wise addition
d = a * b       # Element-wise multiplication
e = a.add(b)    # In-place addition
f = a.mul(b)    # In-place multiplication
g = torch.add(a, b) # Functional form
h = torch.mul(a, b) # Functional form
```

Slicing and Indexing:

```python
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor[0]       # First row
tensor[:, 1]     # Second column
tensor[0, 1]    # Element at row 0, column 1
tensor[0:2, 1:3] # Slicing
```

Reshaping:

```python
tensor = torch.arange(12)
reshaped_tensor = tensor.reshape(3, 4)
transposed_tensor = tensor.T # For 2D tensors
flattened_tensor = tensor.flatten() # Flatten to 1D
viewed_tensor = tensor.view(3, 4) # Similar to reshape, but shares memory
```

Concatenation:

```python
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

concatenated_tensor = torch.cat((tensor1, tensor2), dim=0) # Concatenate along rows
stacked_tensor = torch.stack((tensor1, tensor2), dim=0) # Stack along a new dimension
```

Matrix Multiplication:

```python
a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = torch.matmul(a, b) # Matrix multiplication
d = a @ b # Matrix multiplication (shorthand)
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

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = tensor.to(device)
```

### Moving Data Between CPU and GPU

```python
cpu_tensor = tensor.cpu()
gpu_tensor = tensor.cuda() # or tensor.to('cuda')
```

## Neural Networks

### Defining a Model

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
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

### Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Sample data
X = torch.randn(100, 784)
y = torch.randint(0, 10, (100,))

# Create dataset and dataloader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = nn.Linear(784, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for inputs, labels in dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

### DataLoaders

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
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

```python
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

## Evaluation

```python
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    correct = 0
    total = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')
```

## Prediction

```python
model.eval()
with torch.no_grad():
    input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Example input
    output = model(input_tensor)
    predicted_class = torch.argmax(output).item()
    print(f'Predicted class: {predicted_class}')
```

## Saving and Loading Models

### Save the Entire Model

```python
torch.save(model, 'my_model.pth') # Saves the entire model object
```

### Load the Entire Model

```python
model = torch.load('my_model.pth')
model.eval()
```

### Save Model State Dictionary

```python
torch.save(model.state_dict(), 'model_state_dict.pth') # Saves only the model's learned parameters
```

### Load Model State Dictionary

```python
model = Net()  # Instantiate the model
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()
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

### Distributed Data Loading

When using DDP, you'll want to use a `DistributedSampler` to ensure each process gets a unique subset of the data:

```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
trainloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=train_sampler) # shuffle=False is important here
```

## Autograd

### Tracking Gradients

```python
x = torch.randn(3, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print(x.grad)
```

### Disabling Gradient Tracking

```python
with torch.no_grad():
    y = x + 2
```

### Detaching Tensors

```python
y = x.detach()  # Creates a new tensor with the same content but no gradient history
```

### Custom Autograd Functions

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

my_relu = MyReLU.apply
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
*   `LambdaLR`: Allows defining a custom learning rate schedule using a lambda function.

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

```python
dummy_input = torch.randn(1, 3, 224, 224).to(device) # Example input
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                                                                                                             'output' : {0 : 'batch_size'}})
```

## TorchScript

### Tracing

```python
model.eval()
example = torch.rand(1, 3, 224, 224).to(device)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model_traced.pt")
```

### Scripting

```python
@torch.jit.script
def scripted_function(x, y):
    return x + y
```

## Deployment

### Serving with Flask

```python
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
model = torch.load('my_model.pth')
model.eval()

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image).convert('RGB')
    return transform(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.files.get('image'):
            try:
                img_tensor = transform_image(request.files['image'])
                with torch.no_grad():
                    prediction = model(img_tensor)
                predicted_class = torch.argmax(prediction).item()
                return jsonify({'prediction': str(predicted_class)})
            except Exception as e:
                return jsonify({'error': str(e)})
        return jsonify({'error': 'No image found'})
    return jsonify({'message': 'Use POST method'})
```

### Serving with TorchServe

1.  Install TorchServe:

```bash
pip install torchserve torch-model-archiver
```

2.  Create a model archive:

```bash
torch-model-archiver --model-name my_model --version 1.0 --model-file model.py --serialized-file model.pth --handler handler.py --extra-files index_to_name.json
```

3.  Start TorchServe:

```bash
torchserve --start --model-store model_store --models my_model=my_model.mar
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

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
```

### Weight Decay

Weight decay (L2 regularization) is often included directly in the optimizer:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

### Early Stopping

```python
patience = 10
best_val_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    # Training and validation steps
    val_loss = validate(model, validation_loader, criterion)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break
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

```python
accumulation_steps = 4 # Accumulate gradients over 4 batches

optimizer.zero_grad() # Reset gradients before starting

for i, (inputs, labels) in enumerate(trainloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps # Normalize the loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0: # Every accumulation_steps batches
        optimizer.step()        # Perform optimization step
        optimizer.zero_grad()   # Reset gradients
```

## Common Issues and Debugging

*   **CUDA Out of Memory Errors:** Reduce batch size, use mixed precision training, use gradient checkpointing, use a smaller model, or use multiple GPUs.
*   **Slow Training:** Profile your code to identify bottlenecks, use a GPU, use data parallelism or distributed training, optimize data loading, or use faster operations.
*   **NaN Losses:** Reduce learning rate, use gradient clipping, use a different optimizer, check for numerical instability, or normalize your data.
*   **Overfitting:** Use regularization techniques, data augmentation, early stopping, or reduce model complexity.
*   **Underfitting:** Increase model capacity, train for longer, use a more complex optimizer, or add more features.
*   **Incorrect Tensor Shapes:** Carefully check the shapes of your tensors and ensure they are compatible with the operations you are performing. Use `tensor.shape` to inspect tensor shapes.
*   **Incorrect Data Types:** Ensure that your tensors have the correct data types (e.g., `torch.float32` for floating-point operations, `torch.long` for indices). Use `tensor.dtype` to inspect tensor data types.
*   **Device Mismatch:** Ensure that all tensors and models are on the same device (CPU or GPU). Use `tensor.to(device)` and `model.to(device)` to move tensors and models to the correct device.
*   **Gradients Not Flowing:** Check that `requires_grad=True` is set for the tensors you want to compute gradients for. Check that you are not detaching tensors from the computation graph unintentionally.
*   **Dead Neurons (ReLU):** Use Leaky ReLU or other activation functions that allow a small gradient to flow even when the input is negative.
*   **Exploding Gradients:** Use gradient clipping to limit the magnitude of gradients.
*   **Vanishing Gradients:** Use skip connections (e.g., ResNet), batch normalization, or different activation functions.
*   **Data Loading Bottlenecks:** Increase the number of worker processes in your `DataLoader` and use pinned memory.
*   **Incorrect Loss Function:** Ensure that you are using the appropriate loss function for your task (e.g., `CrossEntropyLoss` for multi-class classification, `MSELoss` for regression).
*   **Incorrect Optimizer:** Experiment with different optimizers and learning rates to find the best configuration for your task.
*   **Unstable Training:** Use a smaller learning rate, increase the batch size, or use a more stable optimizer.
*   **Model Not Learning:** Check your data for errors, ensure that your model is complex enough to learn the task, and try different hyperparameters.

## Tips and Best Practices

*   Use virtual environments to isolate project dependencies.
*   Use meaningful names for variables and functions.
*   Follow the DRY (Don't Repeat Yourself) principle.
*   Write unit tests to ensure code quality.
*   Use a consistent coding style (e.g., PEP 8).
*   Document your code.
*   Use a version control system (e.g., Git).
*   Use appropriate data types for your data.
*   Optimize your PyTorch configuration for your workload.
*   Use caching to improve performance.
*   Use a logging framework to log events and errors.
*   Use a security framework to protect your data.
*   Use a resource manager (e.g., YARN, Mesos, Kubernetes) to manage your cluster.
*   Use a deployment tool to deploy your application to production.
*   Monitor your application for performance issues.
*   Use a CDN (Content Delivery Network) for static files.
*   Optimize database queries.
*   Use asynchronous tasks for long-running operations.
*   Implement proper logging and error handling.
*   Regularly update PyTorch and its dependencies.
*   Use a security scanner to identify potential vulnerabilities.
*   Follow security best practices.
*   Use a reverse proxy like Nginx or Apache in front of your PyTorch application.
*   Use a load balancer for high availability.
*   Automate deployments using tools like Fabric or Ansible.
*   Use a monitoring tool like Prometheus or Grafana.
*   Implement health checks for your application.
*   Use a CDN for static assets.
*   Cache frequently accessed data.
*   Use a database connection pool.
*   Optimize your database queries.
*   Use a task queue for long-running tasks.
*   Use a background worker for asynchronous tasks.
*   Use a message queue for inter-process communication.
*   Use a service discovery tool for microservices.
*   Use a containerization tool like Docker.
*   Use an orchestration tool like Kubernetes.
*   Use a model compression technique to reduce model size.
*   Use quantization to reduce model size and improve inference speed.
*   Use pruning to remove unnecessary connections from the model.
*   Use knowledge distillation to transfer knowledge from a large model to a smaller model.
*   Use a model deployment framework like TorchServe, TensorFlow Serving, or ONNX Runtime.
*   Use a model monitoring tool to track model performance in production.
*   Implement A/B testing to compare different model versions.
*   Use a CI/CD pipeline to automate the model deployment process.
*   Use a feature store to manage your features.
*   Use a data catalog to manage your data.
*   Use a data lineage tool to track the flow of data through your system.
*   Use a data governance tool to ensure data quality and compliance.
*   Use a model registry to manage your models.
*   Use a model versioning tool to track changes to your models.
*   Use a model explainability tool to understand why your model is making certain predictions.
*   Use a model fairness tool to ensure that your model is not biased against certain groups of people.
*   Use a model security tool to protect your model from adversarial attacks.
*   Use a model privacy tool to protect the privacy of your data.

