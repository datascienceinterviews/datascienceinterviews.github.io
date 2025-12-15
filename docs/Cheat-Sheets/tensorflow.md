
---
title: TensorFlow Cheat Sheet
description: A comprehensive reference guide for TensorFlow 2.x, covering tensors, operations, models, layers, training, and more.
---

# TensorFlow Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of TensorFlow 2.x, covering essential concepts, validated examples, and best practices for efficient deep learning model building, training, evaluation, and deployment.

## Quick Reference

| Task | Code | Use Case |
|------|------|----------|
| **Create Tensor** | `tf.constant([[1, 2]])` | Fixed values |
| **Create Variable** | `tf.Variable([1.0, 2.0])` | Trainable parameters |
| **Matrix Multiply** | `tf.matmul(a, b)` | Linear transformations |
| **Reduce Sum** | `tf.reduce_sum(tensor, axis=1)` | Aggregate along dimension |
| **Gradient** | `tape.gradient(loss, vars)` | Backpropagation |
| **Dense Layer** | `Dense(128, activation='relu')` | Fully connected layer |
| **Conv2D Layer** | `Conv2D(32, (3,3), activation='relu')` | Image feature extraction |
| **LSTM Layer** | `LSTM(64, return_sequences=True)` | Sequential data processing |
| **Compile Model** | `model.compile(optimizer='adam', loss='mse')` | Setup training |
| **Train Model** | `model.fit(X, y, epochs=10)` | Train on data |
| **Save Model** | `model.save('model.h5')` | Persist trained model |
| **Load Model** | `tf.keras.models.load_model('model.h5')` | Load saved model |
| **Predict** | `model.predict(X_test)` | Inference |
| **Create Dataset** | `tf.data.Dataset.from_tensor_slices((X, y))` | Efficient data pipeline |
| **Batch Dataset** | `dataset.batch(32).prefetch(tf.data.AUTOTUNE)` | Optimize throughput |

### Activation Functions Quick Reference

```
relu:     f(x) = max(0, x)         → Most common, fast
sigmoid:  f(x) = 1/(1+e^(-x))      → Binary classification output
tanh:     f(x) = (e^x-e^(-x))/(e^x+e^(-x)) → Range [-1,1]
softmax:  f(x) = e^xi / Σe^xj      → Multi-class classification
elu:      f(x) = x if x>0 else α(e^x-1) → Smooth approximation
swish:    f(x) = x * sigmoid(x)    → Self-gated, smooth
```

### Loss Functions Quick Reference

```
Binary Classification:    binary_crossentropy
Multi-class (one-hot):    categorical_crossentropy  
Multi-class (integers):   sparse_categorical_crossentropy
Regression:               mean_squared_error, mean_absolute_error
Ranking/Metric Learning:  hinge, triplet_loss
```

### Optimizer Quick Reference

```
Adam:     Adaptive learning, momentum → Default choice
SGD:      Simple, stable → Fine-tuning, convergence
RMSprop:  Adaptive learning → RNNs, non-stationary
Adagrad:  Adaptive per-parameter → Sparse data
```

## Getting Started

### Installation

```bash
# Install TensorFlow (CPU and GPU support included)
pip install tensorflow

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Importing TensorFlow

```python
import tensorflow as tf
import numpy as np
```

### Environment Setup

```python
# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpus)}")

# Enable memory growth for GPUs (prevents OOM errors)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
```

### TensorFlow Execution Flow

```
    ┌──────────────────┐
    │   Load Data      │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Preprocess      │───→ tf.data pipeline
    │  & Augment       │     (map, batch, prefetch)
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Build Model     │───→ Sequential, Functional,
    │                  │     or Subclassing API
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Compile Model   │───→ optimizer, loss, metrics
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Train Model     │───→ model.fit() or custom loop
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Evaluate        │───→ model.evaluate()
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Predict         │───→ model.predict()
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Save Model      │───→ SavedModel or HDF5
    └──────────────────┘
```

## Tensors

### Tensor Hierarchy

```
    TensorFlow Tensors
           │
    ┌──────┴──────┐
    ↓             ↓
Constant      Variable
(Immutable)   (Mutable)
    │             │
    └──────┬──────┘
           ↓
    GPU/CPU Memory
```

### Creating Tensors

```python
# Constant tensors
a = tf.constant([[1, 2], [3, 4]])                    # From list
b = tf.zeros((2, 3))                                  # All zeros
c = tf.ones((3, 2))                                   # All ones
d = tf.eye(3)                                         # Identity matrix
e = tf.fill((2, 3), 9)                               # Fill with specific value

# Random tensors
f = tf.random.normal((2, 2), mean=0.0, stddev=1.0)   # Normal distribution
g = tf.random.uniform((2, 2), minval=0, maxval=10)   # Uniform distribution
h = tf.random.truncated_normal((2, 2))               # Truncated normal

# From NumPy arrays
arr = np.array([[1, 2], [3, 4]])
tensor_from_np = tf.convert_to_tensor(arr)

# Sequences
range_tensor = tf.range(start=0, limit=10, delta=2)  # [0, 2, 4, 6, 8]
linspace_tensor = tf.linspace(0.0, 1.0, num=5)       # [0.0, 0.25, 0.5, 0.75, 1.0]

# One-hot encoding
labels = tf.constant([0, 1, 2, 0])
one_hot = tf.one_hot(labels, depth=3)                # Shape: (4, 3)
```

### Tensor Attributes

```python
tensor = tf.constant([[1, 2], [3, 4]])
print(tensor.shape)       # Shape of the tensor
print(tensor.dtype)       # Data type of the tensor
print(tensor.device)      # Device where the tensor is stored (CPU or GPU)
print(tensor.numpy())     # Convert to a NumPy array
```

### Tensor Operations

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Element-wise operations
add_result = a + b                    # [[6, 8], [10, 12]]
sub_result = a - b                    # [[-4, -4], [-4, -4]]
mul_result = a * b                    # [[5, 12], [21, 32]]
div_result = a / b                    # [[0.2, 0.33], [0.43, 0.5]]
pow_result = tf.pow(a, 2)            # [[1, 4], [9, 16]]

# Matrix operations
matmul_result = tf.matmul(a, b)      # [[19, 22], [43, 50]]
transpose_result = tf.transpose(a)    # [[1, 3], [2, 4]]
determinant = tf.linalg.det(a)       # -2.0
inverse = tf.linalg.inv(a)           # [[-2, 1], [1.5, -0.5]]

# Shape operations
reshape_result = tf.reshape(a, [1, 4])         # [[1, 2, 3, 4]]
squeeze_result = tf.squeeze([[[1], [2]]])      # [1, 2]
expand_result = tf.expand_dims([1, 2], axis=0) # [[1, 2]]

# Concatenation and stacking
concat_result = tf.concat([a, b], axis=0)      # Shape: (4, 2)
stack_result = tf.stack([a, b], axis=0)        # Shape: (2, 2, 2)

# Reduction operations
sum_all = tf.reduce_sum(a)                     # 10.0
sum_axis0 = tf.reduce_sum(a, axis=0)          # [4, 6]
mean_all = tf.reduce_mean(a)                   # 2.5
max_val = tf.reduce_max(a)                     # 4.0
min_val = tf.reduce_min(a)                     # 1.0

# Argmax and Argmin
argmax_axis1 = tf.argmax(a, axis=1)           # [1, 1] (indices)
argmin_axis0 = tf.argmin(a, axis=0)           # [0, 0] (indices)

# Comparison operations
greater = tf.greater(a, 2.0)                  # [[False, False], [True, True]]
equal = tf.equal(a, b)                        # [[False, False], [False, False]]

# Clipping
clipped = tf.clip_by_value(a, 2.0, 3.0)      # [[2, 2], [3, 3]]

# Casting
float_tensor = tf.cast(tf.constant([1, 2, 3]), tf.float32)
int_tensor = tf.cast(tf.constant([1.5, 2.8]), tf.int32)  # [1, 2]
```

### Tensor Operation Flow

```
    Input Tensors (a, b)
           │
    ┌──────┴──────┐
    ↓             ↓
Element-wise    Matrix Ops
Operations      (matmul)
    │             │
    ├─→ +, -, *, / │
    ├─→ pow, sqrt  │
    └─→ log, exp   ↓
           │    Transpose,
           │    Inverse
           ↓
    Reduction Ops
    (sum, mean, max)
           │
           ↓
    Output Tensor
```

### Indexing and Slicing

```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(tensor[0])       # First row
print(tensor[:, 1])     # Second column
print(tensor[0:2, 1:3]) # Slicing
```

### Data Types

*   `tf.float16`, `tf.float32`, `tf.float64`: Floating-point numbers.
*   `tf.int8`, `tf.int16`, `tf.int32`, `tf.int64`: Signed integers.
*   `tf.uint8`, `tf.uint16`, `tf.uint32`, `tf.uint64`: Unsigned integers.
*   `tf.bool`: Boolean.
*   `tf.string`: String.
*   `tf.complex64`, `tf.complex128`: Complex numbers.
*   `tf.qint8`, `tf.qint32`, `tf.quint8`: Quantized integers.

## Variables

Variables are mutable tensors used to store model parameters (weights and biases).

```python
# Create a variable
var = tf.Variable([1.0, 2.0], name='my_variable')

# Assign new values
var.assign([3.0, 4.0])          # Replace entire value
var.assign_add([1.0, 1.0])      # Add to current value: [4.0, 5.0]
var.assign_sub([0.5, 0.5])      # Subtract from current value: [3.5, 4.5]

# Accessing properties
print(var.numpy())               # Convert to NumPy array
print(var.shape)                 # Shape of the variable
print(var.dtype)                 # Data type
print(var.trainable)             # Whether variable is trainable

# Non-trainable variables (e.g., for batch norm moving averages)
non_trainable_var = tf.Variable([1.0, 2.0], trainable=False)

# Initialize variables
initial_value = tf.random.normal((3, 3))
weight_var = tf.Variable(initial_value, name='weights')
```

### Variable Lifecycle

```
    Create Variable
           │
           ↓
    ┌──────────────┐
    │ Initialize   │
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │   Training   │◄──┐
    │   Updates    │   │
    └──────┬───────┘   │
           │           │
           ├───────────┘ (Multiple epochs)
           ↓
    ┌──────────────┐
    │    Save      │
    │  Checkpoint  │
    └──────────────┘
```

## Automatic Differentiation (Autograd)

GradientTape records operations for automatic differentiation (computing gradients).

### Gradient Computation Flow

```
    Forward Pass
         │
    ┌────▼────┐
    │ Record  │───→ GradientTape
    │  Ops    │     (stores computation graph)
    └────┬────┘
         ↓
    Loss Value
         │
         ↓
    ┌────────────┐
    │  Compute   │───→ tape.gradient()
    │ Gradients  │
    └────┬───────┘
         ↓
    Update Weights
```

### Basic Gradient Computation

```python
# Simple gradient
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2  # y = 9.0

dy_dx = tape.gradient(y, x)  # dy/dx = 2*x = 6.0
print(f"Gradient: {dy_dx.numpy()}")
```

### Multiple Gradients (Persistent Tape)

```python
x = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    y = x**2          # y = 9.0
    z = y * 2         # z = 18.0

dy_dx = tape.gradient(y, x)  # dy/dx = 2*x = 6.0
dz_dx = tape.gradient(z, x)  # dz/dx = 4*x = 12.0

print(f"dy/dx: {dy_dx.numpy()}")
print(f"dz/dx: {dz_dx.numpy()}")

del tape  # Clean up persistent tape
```

### Watching Non-Variable Tensors

```python
# Constants aren't watched by default
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x)  # Explicitly watch the constant
    y = x**2

dy_dx = tape.gradient(y, x)
print(f"Gradient: {dy_dx.numpy()}")
```

### Gradients for Multiple Variables

```python
x = tf.Variable(2.0)
y = tf.Variable(3.0)

with tf.GradientTape() as tape:
    z = x**2 + y**2  # z = 4 + 9 = 13

# Compute gradients for both variables
gradients = tape.gradient(z, [x, y])  # [dz/dx, dz/dy] = [4.0, 6.0]
print(f"dz/dx: {gradients[0].numpy()}")
print(f"dz/dy: {gradients[1].numpy()}")
```

### Nested GradientTapes (Higher-Order Derivatives)

```python
x = tf.Variable(3.0)

with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        y = x**3  # y = 27

    # First derivative: dy/dx = 3*x^2
    dy_dx = inner_tape.gradient(y, x)

# Second derivative: d²y/dx² = 6*x
d2y_dx2 = outer_tape.gradient(dy_dx, x)

print(f"First derivative: {dy_dx.numpy()}")   # 27.0
print(f"Second derivative: {d2y_dx2.numpy()}") # 18.0
```

## Keras API

### Model Building

TensorFlow provides three ways to build models, each with different levels of flexibility.

### Model Building Approaches

```
                Model Building APIs
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   Sequential    Functional API   Subclassing
   (Simplest)     (Flexible)     (Most Control)
        │              │              │
        ├─→ Linear     ├─→ Multiple   ├─→ Custom
        │   stack      │   inputs/    │   logic
        │              │   outputs    │
        └─→ Quick      ├─→ Shared     └─→ Dynamic
            prototypes │   layers         graphs
                       └─→ Complex
                           topologies
```

#### Sequential Model

Best for simple, linear stack of layers.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Method 1: Pass layers as list
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Method 2: Add layers incrementally
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# View model architecture
model.summary()
```

#### Functional API

Best for complex models with multiple inputs/outputs, shared layers, or non-linear topology.

```python
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

# Single input/output
inputs = Input(shape=(784,))
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# Multiple inputs
input1 = Input(shape=(64,), name='input1')
input2 = Input(shape=(32,), name='input2')

x1 = Dense(32, activation='relu')(input1)
x2 = Dense(32, activation='relu')(input2)

# Merge branches
merged = Concatenate()([x1, x2])
output = Dense(10, activation='softmax')(merged)

model = Model(inputs=[input1, input2], outputs=output)
```

#### Model Subclassing

Best for custom models with dynamic behavior.

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.classifier(x)

model = MyModel(num_classes=10)

# Build model by calling it
_ = model(tf.zeros((1, 784)))
model.summary()
```

### ResNet-style Skip Connection Example

```python
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model

inputs = Input(shape=(64,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

# Skip connection
outputs = Add()([inputs, x])

model = Model(inputs=inputs, outputs=outputs)
```

### Layers

#### Core Layers

```python
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten

# Dense (Fully Connected) Layer
dense = Dense(units=64, activation='relu', use_bias=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')

# Activation Layer
activation = Activation('relu')  # or use directly in Dense

# Dropout (regularization)
dropout = Dropout(rate=0.5)  # Drops 50% of inputs during training

# Flatten (convert 2D/3D to 1D)
flatten = Flatten()  # (batch, height, width, channels) → (batch, features)
```

#### Convolutional Layers

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

# 2D Convolution
conv2d = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                padding='same', activation='relu')

# Max Pooling
maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

# Average Pooling
avgpool = AveragePooling2D(pool_size=(2, 2))

# Global Pooling (reduces to 1 value per channel)
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
gap = GlobalAveragePooling2D()
gmp = GlobalMaxPooling2D()
```

#### CNN Architecture Example

```
Input Image (224×224×3)
        │
        ↓
    ┌─────────────┐
    │  Conv2D     │ filters=32, kernel=3×3
    │  + ReLU     │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ MaxPool2D   │ pool=2×2 → (112×112×32)
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Conv2D     │ filters=64, kernel=3×3
    │  + ReLU     │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Flatten   │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Dense     │ units=128
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Output    │ units=10 (classes)
    └─────────────┘
```

#### Recurrent Layers

```python
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional

# LSTM Layer
lstm = LSTM(units=128, return_sequences=True,  # Return full sequence
            return_state=False,                 # Don't return hidden states
            dropout=0.2, recurrent_dropout=0.2)

# GRU Layer
gru = GRU(units=64, return_sequences=False)

# Bidirectional wrapper
bilstm = Bidirectional(LSTM(64, return_sequences=True))

# Stacked RNNs
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    LSTM(64, return_sequences=False),
    Dense(10, activation='softmax')
])
```

#### Normalization Layers

```python
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

# Batch Normalization (normalize across batch)
batchnorm = BatchNormalization(momentum=0.99, epsilon=0.001)

# Layer Normalization (normalize across features)
layernorm = LayerNormalization()

# Usage in model
model = Sequential([
    Dense(64),
    BatchNormalization(),
    Activation('relu')
])
```

#### Embedding Layers

```python
from tensorflow.keras.layers import Embedding

# For text/categorical data
embedding = Embedding(input_dim=10000,      # Vocabulary size
                     output_dim=128,        # Embedding dimension
                     input_length=100)      # Sequence length

# Example usage
model = Sequential([
    Embedding(10000, 128, input_length=100),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```

#### Merge Layers

```python
from tensorflow.keras.layers import Add, Multiply, Concatenate, Average

# Add (element-wise sum)
add_layer = Add()([tensor1, tensor2])

# Multiply (element-wise product)
multiply_layer = Multiply()([tensor1, tensor2])

# Concatenate (along axis)
concat_layer = Concatenate(axis=-1)([tensor1, tensor2])

# Average
avg_layer = Average()([tensor1, tensor2])
```

### Activation Functions

*   `'relu'`: Rectified Linear Unit.
*   `'sigmoid'`: Sigmoid function.
*   `'tanh'`: Hyperbolic tangent function.
*   `'softmax'`: Softmax function.
*   `'elu'`: Exponential Linear Unit.
*   `'selu'`: Scaled Exponential Linear Unit.
*   `'linear'`: Linear (identity) activation.
*   `'LeakyReLU'`: Leaky Rectified Linear Unit.
*   `'PReLU'`: Parametric Rectified Linear Unit.
*   `'gelu'`: Gaussian Error Linear Unit.
*   `'swish'`: Swish activation function.

### Loss Functions

*   `tf.keras.losses.CategoricalCrossentropy`: Categorical cross-entropy.
*   `tf.keras.losses.SparseCategoricalCrossentropy`: Sparse categorical cross-entropy.
*   `tf.keras.losses.BinaryCrossentropy`: Binary cross-entropy.
*   `tf.keras.losses.MeanSquaredError`: Mean squared error.
*   `tf.keras.losses.MeanAbsoluteError`: Mean absolute error.
*   `tf.keras.losses.Hinge`: Hinge loss.
*   `tf.keras.losses.KLDivergence`: Kullback-Leibler Divergence loss.
*   `tf.keras.losses.Huber`: Huber loss.

### Optimizers

*   `tf.keras.optimizers.SGD`: Stochastic Gradient Descent.
*   `tf.keras.optimizers.Adam`: Adaptive Moment Estimation.
*   `tf.keras.optimizers.RMSprop`: Root Mean Square Propagation.
*   `tf.keras.optimizers.Adagrad`: Adaptive Gradient Algorithm.
*   `tf.keras.optimizers.Adadelta`: Adaptive Delta.
*   `tf.keras.optimizers.Adamax`: Adamax optimizer.
*   `tf.keras.optimizers.Nadam`: Nesterov Adam optimizer.
*   `tf.keras.optimizers.Ftrl`: Follow The Regularized Leader optimizer.

### Metrics

*   `tf.keras.metrics.Accuracy`: Accuracy.
*   `tf.keras.metrics.BinaryAccuracy`: Binary accuracy.
*   `tf.keras.metrics.CategoricalAccuracy`: Categorical accuracy.
*   `tf.keras.metrics.SparseCategoricalAccuracy`: Sparse categorical accuracy.
*   `tf.keras.metrics.TopKCategoricalAccuracy`: Top-K categorical accuracy.
*   `tf.keras.metrics.MeanSquaredError`: Mean squared error.
*   `tf.keras.metrics.MeanAbsoluteError`: Mean absolute error.
*   `tf.keras.metrics.Precision`: Precision.
*   `tf.keras.metrics.Recall`: Recall.
*   `tf.keras.metrics.AUC`: Area Under the Curve.
*   `tf.keras.metrics.F1Score`: F1 score.

### Model Compilation

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Training

### Training Pipeline

```
    Data Loading
         │
         ↓
    Preprocessing
         │
         ↓
    ┌────────────┐
    │  Epoch 1   │
    └─────┬──────┘
          ↓
    ┌──────────────────┐
    │  Forward Pass    │───→ Compute predictions
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Compute Loss    │───→ Loss function
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Backward Pass   │───→ Compute gradients
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │ Update Weights   │───→ Optimizer step
    └────────┬─────────┘
             │
             └──→ Repeat for all batches
```

```python
# Prepare data
data = np.random.random((1000, 784))
labels = np.random.randint(10, size=(1000,))
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=10)

# Split into train and validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    data, one_hot_labels, test_size=0.2, random_state=42
)

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1  # 0=silent, 1=progress bar, 2=one line per epoch
)

# Access training history
print(f"Training accuracy: {history.history['accuracy']}")
print(f"Validation loss: {history.history['val_loss']}")

# Train with validation split (alternative)
history = model.fit(
    data, one_hot_labels,
    batch_size=32,
    epochs=10,
    validation_split=0.2,  # Use 20% of data for validation
    shuffle=True
)
```

### Evaluation

```python
# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Evaluate with multiple metrics
test_loss, test_acc, test_precision = model.evaluate(
    X_val, y_val, verbose=0
)

# Batch-wise evaluation
for batch_data, batch_labels in test_dataset:
    loss, acc = model.evaluate(batch_data, batch_labels, verbose=0)
```

### Prediction

```python
# Make predictions
predictions = model.predict(X_val)  # Returns probabilities
print(f"Prediction shape: {predictions.shape}")  # (samples, classes)

# Get predicted classes
predicted_classes = np.argmax(predictions, axis=1)
print(f"Predicted classes: {predicted_classes[:10]}")

# Get prediction confidence
confidence = np.max(predictions, axis=1)
print(f"Confidence scores: {confidence[:10]}")

# Predict single sample
single_sample = X_val[0:1]  # Keep batch dimension
prediction = model.predict(single_sample, verbose=0)
predicted_class = np.argmax(prediction)

# Batch prediction
batch_predictions = model.predict(X_val[:100], batch_size=32)
```

### Saving and Loading Models

### Model Persistence Flow

```
    Trained Model
         │
    ┌────┴────────┐
    ↓             ↓
SavedModel     HDF5
(TF native)   (.h5)
    │             │
    ├─→ Better    ├─→ Keras only
    │   for       │   Single file
    │   serving   │   Legacy
    │             │
    └──────┬──────┘
           ↓
    Load & Use
    (Inference/
     Fine-tuning)
```

### SavedModel Format (Recommended)

```python
# Save entire model (architecture + weights + optimizer state)
model.save('saved_model/my_model')

# Load model
loaded_model = tf.keras.models.load_model('saved_model/my_model')

# Verify loaded model
predictions = loaded_model.predict(X_test[:5])

# SavedModel contains:
# - saved_model.pb (architecture + graph)
# - variables/ (weights)
# - assets/ (additional files)
```

### HDF5 Format

```python
# Save entire model to HDF5
model.save('my_model.h5')

# Load from HDF5
loaded_model = tf.keras.models.load_model('my_model.h5')

# Save only weights
model.save_weights('weights.h5')

# Load only weights (model architecture must exist)
model.load_weights('weights.h5')
```

### Weights Only (Checkpoint Format)

```python
# Save weights in TensorFlow checkpoint format
model.save_weights('./checkpoints/model_checkpoint')

# Creates:
# - model_checkpoint.index
# - model_checkpoint.data-00000-of-00001

# Load weights
model.load_weights('./checkpoints/model_checkpoint')

# Transfer learning: load specific layers
pretrained_model = tf.keras.applications.VGG16(weights='imagenet')
new_model = tf.keras.Sequential([
    pretrained_model,
    Dense(10, activation='softmax')
])
```

### Model Architecture Only

```python
# Save architecture to JSON
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)

# Load architecture
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# Then load weights separately
loaded_model.load_weights('weights.h5')
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy')
```

### Custom Objects

```python
# If model has custom layers/functions
class CustomLayer(tf.keras.layers.Layer):
    pass

# Save with custom objects
model.save('custom_model.h5')

# Load with custom objects
loaded_model = tf.keras.models.load_model(
    'custom_model.h5',
    custom_objects={'CustomLayer': CustomLayer}
)
```

### Checkpoint Management

```python
# Create checkpoint manager
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(
    optimizer=optimizer,
    model=model
)

manager = tf.train.CheckpointManager(
    checkpoint, 
    checkpoint_dir, 
    max_to_keep=3  # Keep only 3 latest checkpoints
)

# Save checkpoint during training
for epoch in range(10):
    # ... training code ...
    save_path = manager.save()
    print(f"Saved checkpoint: {save_path}")

# Restore latest checkpoint
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f"Restored from {manager.latest_checkpoint}")
```

### Callbacks

Callbacks are functions called at specific points during training for monitoring and automation.

### Callback Execution Flow

```
    Training Start
         │
         ↓
    on_train_begin()
         │
    ┌────▼────────┐
    │   Epoch     │◄──────────┐
    │   Start     │           │
    └────┬────────┘           │
         ↓                    │
    on_epoch_begin()          │
         │                    │
    ┌────▼────────┐           │
    │   Batch     │◄─────┐    │
    │   Start     │      │    │
    └────┬────────┘      │    │
         ↓               │    │
    on_batch_begin()     │    │
         ↓               │    │
    Forward + Backward   │    │
         ↓               │    │
    on_batch_end()       │    │
         │               │    │
         └───────────────┘    │
         (more batches)       │
         ↓                    │
    on_epoch_end()            │
         │                    │
         ├─→ Check metrics    │
         ├─→ Save checkpoint  │
         ├─→ Adjust LR        │
         │                    │
         └────────────────────┘
         (more epochs or early stop)
         ↓
    on_train_end()
```

### Common Callbacks

```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard,
    ReduceLROnPlateau, CSVLogger, LearningRateScheduler
)

# ModelCheckpoint: Save model during training
checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/model-{epoch:02d}-{val_loss:.2f}.h5',
    save_best_only=True,           # Only save when val_loss improves
    monitor='val_loss',             # Metric to monitor
    mode='min',                     # 'min' for loss, 'max' for accuracy
    verbose=1,
    save_weights_only=False        # Save entire model
)

# EarlyStopping: Stop training when metric stops improving
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,                     # Stop after 5 epochs without improvement
    restore_best_weights=True,      # Restore weights from best epoch
    verbose=1,
    min_delta=0.001                # Minimum change to qualify as improvement
)

# TensorBoard: Visualize training metrics
tensorboard_callback = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,               # Frequency for weight histograms
    write_graph=True,
    update_freq='epoch',            # Update frequency
    profile_batch='10,20'          # Profile batches 10-20
)

# ReduceLROnPlateau: Reduce learning rate when metric plateaus
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                     # Reduce LR by half
    patience=3,                     # Wait 3 epochs before reducing
    min_lr=1e-7,                    # Minimum learning rate
    verbose=1
)

# CSVLogger: Log training metrics to CSV
csv_logger = CSVLogger('training_log.csv', append=True)

# LearningRateScheduler: Custom learning rate schedule
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

# Use callbacks in training
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        tensorboard_callback,
        reduce_lr_callback,
        csv_logger
    ]
)
```

### Custom Callback

```python
import time

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Starting training...")
        self.train_start_time = time.time()
    
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nStarting epoch {epoch + 1}")
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
        
        # Custom logic: Save model if accuracy > threshold
        if logs['accuracy'] > 0.95:
            print("High accuracy achieved! Saving model...")
            self.model.save(f'high_acc_model_epoch_{epoch + 1}.h5')
    
    def on_train_batch_end(self, batch, logs=None):
        # Print every 100 batches
        if batch % 100 == 0:
            print(f"Batch {batch}, Loss: {logs['loss']:.4f}")
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start_time
        print(f"\nTraining completed in {total_time:.2f}s")

# Use custom callback
model.fit(
    X_train, y_train,
    epochs=10,
    callbacks=[CustomCallback()]
)
```

### Viewing TensorBoard

```bash
# Start TensorBoard server
tensorboard --logdir=./logs --port=6006

# Open in browser: http://localhost:6006
```

### Regularization

*   `tf.keras.regularizers.l1(0.01)`: L1 regularization.
*   `tf.keras.regularizers.l2(0.01)`: L2 regularization.
*   `tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)`: L1 and L2 regularization.

```python
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,),
          kernel_regularizer=regularizers.l1(0.01),  # L1 regularization
          bias_regularizer=regularizers.l2(0.01)),    # L2 regularization
    Dense(10, activation='softmax')
])
```

### Custom Layers

```python
import tensorflow as tf

class MyCustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(MyCustomLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

### Custom Loss Functions

```python
import tensorflow as tf

def my_custom_loss(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)
```

### Custom Metrics

```python
import tensorflow as tf

class MyCustomMetric(tf.keras.metrics.Metric):
    def __init__(self, name='my_custom_metric', **kwargs):
        super(MyCustomMetric, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.abs(y_true - y_pred)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            values = tf.multiply(values, sample_weight)
        self.sum.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.size(y_true), self.dtype))

    def result(self):
        return self.sum / self.count

    def reset_state(self):
        self.sum.assign(0.0)
        self.count.assign(0.0)
```

### Custom Training Loops

Custom training loops provide fine-grained control over the training process.

### Custom Training Flow

```
    Initialize
    (optimizer, loss, metrics)
         │
         ↓
    ┌────────────┐
    │  Epoch     │◄──────────┐
    └──────┬─────┘           │
           ↓                 │
    ┌────────────┐           │
    │  Get Batch │◄─────┐    │
    └──────┬─────┘      │    │
           ↓            │    │
    ┌────────────────┐  │    │
    │ Forward Pass   │  │    │
    │ (GradientTape) │  │    │
    └────────┬───────┘  │    │
             ↓          │    │
    ┌────────────────┐  │    │
    │ Compute Loss   │  │    │
    └────────┬───────┘  │    │
             ↓          │    │
    ┌────────────────┐  │    │
    │ Compute        │  │    │
    │ Gradients      │  │    │
    └────────┬───────┘  │    │
             ↓          │    │
    ┌────────────────┐  │    │
    │ Apply          │  │    │
    │ Gradients      │  │    │
    └────────┬───────┘  │    │
             ↓          │    │
    ┌────────────────┐  │    │
    │ Update Metrics │  │    │
    └────────┬───────┘  │    │
             │          │    │
             ├──────────┘    │
             │ (more batches)│
             ↓               │
    ┌────────────────┐       │
    │ Log Results    │       │
    └────────┬───────┘       │
             │               │
             └───────────────┘
             (more epochs)
```

### Basic Custom Training Loop

```python
import tensorflow as tf

# Initialize components
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_acc_metric.update_state(labels, predictions)
    
    return loss

@tf.function
def val_step(images, labels):
    predictions = model(images, training=False)
    loss = loss_fn(labels, predictions)
    val_acc_metric.update_state(labels, predictions)
    return loss

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training phase
    train_losses = []
    for batch, (images, labels) in enumerate(train_dataset):
        loss = train_step(images, labels)
        train_losses.append(loss.numpy())
        
        if batch % 100 == 0:
            print(f"Batch {batch}, Loss: {loss.numpy():.4f}")
    
    # Validation phase
    val_losses = []
    for images, labels in val_dataset:
        val_loss = val_step(images, labels)
        val_losses.append(val_loss.numpy())
    
    # Print epoch results
    train_acc = train_acc_metric.result()
    val_acc = val_acc_metric.result()
    
    print(f"Train Loss: {np.mean(train_losses):.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {np.mean(val_losses):.4f}, Val Acc: {val_acc:.4f}")
    
    # Reset metrics
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()
```

### Advanced Custom Training with Regularization

```python
@tf.function
def train_step_with_regularization(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        
        # Main loss
        loss = loss_fn(labels, predictions)
        
        # Add L2 regularization
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables 
                           if 'bias' not in v.name])
        total_loss = loss + 0.001 * l2_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    # Gradient clipping
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_acc_metric.update_state(labels, predictions)
    
    return loss, total_loss

# Training with logging
for epoch in range(epochs):
    for images, labels in train_dataset:
        loss, total_loss = train_step_with_regularization(images, labels)
```

### Custom Training with Learning Rate Scheduling

```python
# Define learning rate schedule
initial_lr = 0.001
decay_steps = 1000
decay_rate = 0.96

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=decay_steps,
    decay_rate=decay_rate
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Or manual learning rate adjustment
for epoch in range(epochs):
    # Adjust learning rate manually
    if epoch > 0 and epoch % 5 == 0:
        new_lr = optimizer.learning_rate * 0.5
        optimizer.learning_rate.assign(new_lr)
        print(f"Learning rate adjusted to: {new_lr.numpy()}")
    
    for images, labels in train_dataset:
        loss = train_step(images, labels)
```

## Data Input Pipelines (tf.data)

### tf.data Pipeline Flow

```
    Raw Data
        │
        ↓
    ┌───────────┐
    │  Create   │───→ from_tensor_slices
    │  Dataset  │     from_generator
    └─────┬─────┘     list_files
          ↓
    ┌───────────┐
    │  Shuffle  │───→ Randomize order
    └─────┬─────┘
          ↓
    ┌───────────┐
    │    Map    │───→ Preprocess
    │           │     Augment
    └─────┬─────┘
          ↓
    ┌───────────┐
    │   Cache   │───→ Store in memory/disk
    └─────┬─────┘
          ↓
    ┌───────────┐
    │   Batch   │───→ Create batches
    └─────┬─────┘
          ↓
    ┌───────────┐
    │ Prefetch  │───→ Prepare next batch
    └─────┬─────┘
          ↓
      Training
```

### Creating Datasets

```python
import tensorflow as tf
import numpy as np

# From NumPy arrays
X = np.random.random((1000, 28, 28, 1))
y = np.random.randint(10, size=(1000,))
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# From tensors
tensor_x = tf.constant([[1, 2], [3, 4]])
tensor_y = tf.constant([0, 1])
dataset = tf.data.Dataset.from_tensor_slices((tensor_x, tensor_y))

# From list of files
file_dataset = tf.data.Dataset.list_files("path/to/data/*.jpg")

# From CSV
dataset = tf.data.experimental.make_csv_dataset(
    "data.csv",
    batch_size=32,
    label_name="target",
    num_epochs=1
)

# From generator
def data_generator():
    for i in range(1000):
        yield (i, i**2)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

# Range dataset
range_dataset = tf.data.Dataset.range(100)
```

### Dataset Transformations

```python
# Create a dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Shuffle: Randomize order (buffer_size should be >= dataset size for full shuffle)
dataset = dataset.shuffle(buffer_size=1000, seed=42)

# Map: Apply preprocessing function
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    return image, label

dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Filter: Keep only certain elements
dataset = dataset.filter(lambda x, y: y < 5)  # Keep only classes 0-4

# Batch: Create batches
dataset = dataset.batch(32, drop_remainder=False)

# Repeat: Repeat dataset (for multiple epochs)
dataset = dataset.repeat(count=10)  # Repeat 10 times

# Cache: Store dataset in memory or disk
dataset = dataset.cache()  # In-memory
# dataset = dataset.cache("/path/to/cache")  # On disk

# Prefetch: Prepare next batches while training current batch
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Take: Get first N elements
small_dataset = dataset.take(100)

# Skip: Skip first N elements
remaining_dataset = dataset.skip(100)
```

### Optimized Pipeline

```python
def create_dataset(X, y, batch_size=32, shuffle=True, augment=False):
    """Create an optimized tf.data pipeline"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    # Preprocessing
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        if augment:
            # Data augmentation
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
        return image, label
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()  # Cache after preprocessing
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Usage
train_dataset = create_dataset(X_train, y_train, batch_size=64, augment=True)
val_dataset = create_dataset(X_val, y_val, batch_size=64, shuffle=False)
```

### Advanced Transformations

```python
# Zip: Combine multiple datasets
dataset1 = tf.data.Dataset.range(5)
dataset2 = tf.data.Dataset.range(5, 10)
zipped = tf.data.Dataset.zip((dataset1, dataset2))

# FlatMap: Map and flatten
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.flat_map(
    lambda x: tf.data.Dataset.from_tensor_slices([x, x*2, x*3])
)

# Interleave: Parallel processing of multiple files
files = tf.data.Dataset.list_files("data/*.csv")
dataset = files.interleave(
    lambda x: tf.data.TextLineDataset(x),
    cycle_length=4,  # Process 4 files in parallel
    num_parallel_calls=tf.data.AUTOTUNE
)

# Window: Create sliding windows
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(size=3, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda x: x.batch(3))
```

### Reading TFRecord Files

```python
raw_dataset = tf.data.TFRecordDataset("my_data.tfrecord")

# Define a feature description
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64),
    'feature1': tf.io.FixedLenFeature([], tf.string),
    'feature2': tf.io.FixedLenFeature([10], tf.float32),
}

def _parse_function(example_proto):
  # Parse the input tf.train.Example proto using the feature description.
  return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
```

## Distributed Training

### Distribution Strategies Overview

```
    Distribution Strategies
            │
    ┌───────┴───────┬─────────────┬──────────────┐
    ↓               ↓             ↓              ↓
Mirrored      MultiWorker    Parameter      TPU
Strategy      Mirrored       Server         Strategy
    │         Strategy       Strategy
    │             │             │              │
Single        Multiple      Multiple       Cloud
Machine       Machines      Machines       TPUs
Multiple      Multiple      PS + Workers
GPUs          GPUs
```

### MirroredStrategy (Single Machine, Multiple GPUs)

Synchronous training on multiple GPUs on a single machine.

```python
import tensorflow as tf

# Create strategy
strategy = tf.distribute.MirroredStrategy()

print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Build and compile model inside strategy scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Prepare dataset
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(GLOBAL_BATCH_SIZE)

# Train (distributed automatically)
model.fit(train_dataset, epochs=10)
```

### Custom Training Loop with MirroredStrategy

```python
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )

def compute_loss(labels, predictions):
    per_example_loss = loss_fn(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, 
                                     global_batch_size=GLOBAL_BATCH_SIZE)

@tf.function
def train_step(inputs):
    images, labels = inputs
    
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Training loop
for epoch in range(10):
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_dataset:
        loss = distributed_train_step(batch)
        total_loss += loss
        num_batches += 1
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / num_batches:.4f}")
```

### MultiWorkerMirroredStrategy

```python
import os, json

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    # ... build and compile model ...
```

### ParameterServerStrategy

```python
import os, json
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"],
        'ps': ["localhost:34567"]
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.experimental.ParameterServerStrategy()

with strategy.scope():
    # ... build and compile model ...
```

### TPUStrategy

```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    # ... build and compile model ...
```

## Data Augmentation

Data augmentation artificially increases training data by applying transformations.

### Image Augmentation Techniques

```
    Original Image
         │
    ┌────┴────┬────────┬────────┬─────────┐
    ↓         ↓        ↓        ↓         ↓
  Flip    Rotation  Zoom   Brightness  Contrast
    │         │        │        │         │
    └────┬────┴────────┴────────┴─────────┘
         ↓
   Augmented Images
   (More training data)
```

### Using tf.keras.layers for Augmentation

```python
# Create augmentation layers (applied during training)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),  # ±20% = ±72 degrees
    tf.keras.layers.RandomZoom(0.2),      # ±20% zoom
    tf.keras.layers.RandomTranslation(0.1, 0.1),  # 10% shift
    tf.keras.layers.RandomContrast(0.2),
])

# Build model with augmentation
model = tf.keras.Sequential([
    data_augmentation,  # Only active during training
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Using tf.image for Augmentation

```python
def augment_image(image, label):
    """Custom augmentation function"""
    # Random flip
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random saturation (for color images)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Random hue (for color images)
    image = tf.image.random_hue(image, max_delta=0.1)
    
    # Random crop
    image = tf.image.random_crop(image, size=[224, 224, 3])
    
    # Clip values to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

# Apply to dataset
train_dataset = train_dataset.map(augment_image, 
                                  num_parallel_calls=tf.data.AUTOTUNE)
```

### Advanced Augmentation (MixUp, CutMix)

```python
def mixup(image1, label1, image2, label2, alpha=0.2):
    """MixUp augmentation"""
    # Sample mixing coefficient
    lambda_val = tf.random.uniform([], 0, 1)
    lambda_val = tf.maximum(lambda_val, 1 - lambda_val)
    
    # Mix images
    mixed_image = lambda_val * image1 + (1 - lambda_val) * image2
    
    # Mix labels
    mixed_label = lambda_val * label1 + (1 - lambda_val) * label2
    
    return mixed_image, mixed_label

def cutmix(image1, label1, image2, label2, alpha=1.0):
    """CutMix augmentation"""
    # Sample cutting area
    lambda_val = tf.random.uniform([], 0, 1)
    
    # Get image size
    h, w = tf.shape(image1)[0], tf.shape(image1)[1]
    
    # Calculate cut size
    cut_h = tf.cast(tf.cast(h, tf.float32) * tf.sqrt(1 - lambda_val), tf.int32)
    cut_w = tf.cast(tf.cast(w, tf.float32) * tf.sqrt(1 - lambda_val), tf.int32)
    
    # Random position
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    
    # Calculate boundaries
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
    
    # Create mask
    mask = tf.ones([h, w, 3])
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[y1, x1], [y2, x2]],
        [tf.zeros([y2-y1, x2-x1, 3])]
    )
    
    # Apply cutmix
    mixed_image = image1 * mask + image2 * (1 - mask)
    
    # Mix labels based on area
    lambda_val = 1 - ((x2 - x1) * (y2 - y1)) / (h * w)
    mixed_label = lambda_val * label1 + (1 - lambda_val) * label2
    
    return mixed_image, mixed_label
```

### Image Preprocessing

```python
# Normalization
def normalize_image(image):
    """Normalize image to [0, 1] or [-1, 1]"""
    # To [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    # Or to [-1, 1]
    # image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    
    return image

# Standardization (ImageNet mean/std)
def standardize_image(image):
    """Standardize using ImageNet statistics"""
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - mean) / std
    
    return image

# Resize and crop
def resize_and_crop(image, size=224):
    """Resize and center crop"""
    # Resize to slightly larger
    image = tf.image.resize(image, [size + 32, size + 32])
    
    # Center crop
    image = tf.image.resize_with_crop_or_pad(image, size, size)
    
    return image

# Combined preprocessing
def preprocess_pipeline(image, label, training=False):
    """Complete preprocessing pipeline"""
    # Decode if needed
    if image.dtype == tf.string:
        image = tf.image.decode_jpeg(image, channels=3)
    
    # Resize
    image = tf.image.resize(image, [224, 224])
    
    if training:
        # Augmentation for training
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Normalize
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label
```

## TensorFlow Hub

Pre-trained models for transfer learning and feature extraction.

### Using Pre-trained Models

```python
import tensorflow_hub as hub

# Method 1: As a layer
model = tf.keras.Sequential([
    hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
        trainable=False,  # Freeze weights
        input_shape=(224, 224, 3)
    ),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Method 2: Load as a model
embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Use for embeddings
sentences = ["Hello world", "Machine learning is great"]
embeddings = embedding_model(sentences)

# Method 3: KerasLayer with trainable=True for fine-tuning
feature_extractor = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
    trainable=True,  # Fine-tune
    input_shape=(224, 224, 3)
)

model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## TensorFlow Lite

TensorFlow Lite enables on-device machine learning for mobile and IoT devices.

### TensorFlow Lite Workflow

```
    Trained Model
         │
         ↓
    ┌──────────────┐
    │   Convert    │───→ TFLite Converter
    │   to .tflite│
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │  Optimize    │───→ Quantization
    │  (Optional)  │     Pruning
    └──────┬───────┘
           ↓
    .tflite Model
    (Smaller, Faster)
           │
           ↓
    ┌──────────────┐
    │   Deploy to  │───→ Mobile (Android/iOS)
    │   Device     │     IoT (Raspberry Pi)
    └──────────────┘     Microcontrollers
```

### Converting to TensorFlow Lite

```python
# Convert from Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save to file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert from SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/my_model')
tflite_model = converter.convert()

# Convert from concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
```

### Post-Training Quantization

```python
# Dynamic range quantization (weights only)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Full integer quantization (weights + activations)
def representative_dataset():
    for i in range(100):
        # Use representative data samples
        yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

# Float16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_float16_model = converter.convert()
```

### Inference with TensorFlow Lite

```python
# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
# Or from model content
# interpreter = tf.lite.Interpreter(model_content=tflite_model)

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Input dtype: {input_details[0]['dtype']}")

# Prepare input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"Prediction: {output_data}")
```

### Batch Inference

```python
# Resize input tensor for batch processing
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.resize_tensor_input(input_details[0]['index'], [10, 224, 224, 3])
interpreter.allocate_tensors()

# Now can process batch of 10 images
batch_data = np.random.random((10, 224, 224, 3)).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], batch_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

### Model Analysis

```python
# Analyze model size
import os
model_size = os.path.getsize('model.tflite')
quant_model_size = os.path.getsize('model_quant.tflite')

print(f"Original model: {model_size / 1024:.2f} KB")
print(f"Quantized model: {quant_model_size / 1024:.2f} KB")
print(f"Size reduction: {(1 - quant_model_size/model_size) * 100:.1f}%")
```

## TensorFlow Serving

### Exporting a SavedModel

```python
tf.saved_model.save(model, "path/to/saved_model")
```

### Serving with TensorFlow Serving

1.  Install TensorFlow Serving:

    ```bash
    # See TensorFlow Serving installation guide for details
    ```

2.  Start the server:

    ```bash
    tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=my_model --model_base_path=/path/to/saved_model
    ```

3.  Send requests (using `requests` library in Python):

```python
import requests
import json

data = json.dumps({"instances": [[1.0, 2.0, ...]]}) # Example input data
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/my_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
```

## TensorFlow Extended (TFX)

TFX is a platform for building and deploying production ML pipelines.  It includes components for:

*   Data validation (`tensorflow_data_validation`)
*   Data transformation (`tensorflow_transform`)
*   Model training (`tensorflow`)
*   Model analysis (`tensorflow_model_analysis`)
*   Model serving (`tensorflow_serving`)
*   Pipeline orchestration (Apache Beam, Apache Airflow, Kubeflow Pipelines)

## TensorFlow Probability

### Installation

```bash
pip install tensorflow-probability
```

### Distributions

```python
import tensorflow_probability as tfp

tfd = tfp.distributions

# Normal distribution
normal_dist = tfd.Normal(loc=0., scale=1.)
samples = normal_dist.sample(10)
log_prob = normal_dist.log_prob(0.)

# Bernoulli distribution
bernoulli_dist = tfd.Bernoulli(probs=0.7)
samples = bernoulli_dist.sample(10)

# Categorical distribution
categorical_dist = tfd.Categorical(probs=[0.2, 0.3, 0.5])
samples = categorical_dist.sample(10)
```

### Bijectors

```python
import tensorflow_probability as tfp

tfb = tfp.bijectors

# Affine bijector
affine_bijector = tfb.Affine(shift=2., scale_diag=[3., 4.])
transformed_tensor = affine_bijector.forward(tf.constant([[1., 2.]]))

# Exp bijector
exp_bijector = tfb.Exp()
transformed_tensor = exp_bijector.forward(tf.constant([0., 1., 2.]))
```

### Markov Chain Monte Carlo (MCMC)

```python
import tensorflow_probability as tfp

tfd = tfp.distributions
tfm = tfp.mcmc

# Define a target distribution (e.g., a normal distribution)
target_log_prob_fn = tfd.Normal(loc=0., scale=1.).log_prob

# Define a kernel (e.g., Hamiltonian Monte Carlo)
kernel = tfm.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=0.1,
    num_leapfrog_steps=3)

# Run the MCMC sampler
samples, _ = tfm.sample_chain(
    num_results=1000,
    current_state=0.,
    kernel=kernel)
```

## TensorFlow Datasets (TFDS)

### Installation

```bash
pip install tensorflow-datasets
```

### Loading Datasets

```python
import tensorflow_datasets as tfds

# Load a dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Print dataset information
print(ds_info)
```

### Processing Datasets

```python
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
```

## TensorFlow Addons

### Installation

```bash
pip install tensorflow-addons
```

### Usage (Example: WeightNormalization)

```python
import tensorflow_addons as tfa

model = tf.keras.Sequential([
    tfa.layers.WeightNormalization(tf.keras.layers.Dense(64, activation="relu"), data_init=False),
    tf.keras.layers.Dense(10, activation="softmax"),
])
```

## Eager Execution

Eager execution is enabled by default in TensorFlow 2.x.  You can check if it's enabled:

```python
tf.executing_eagerly()  # Returns True
```

## tf.function

Converts Python functions into TensorFlow graphs for improved performance.

### tf.function Benefits

```
    Python Function
    (Eager Execution)
           │
           ↓
    @tf.function
           │
    ┌──────┴──────┐
    ↓             ↓
Graph Mode    Autograph
(Faster)      (Python→TF)
    │             │
    ├─→ Faster    ├─→ Handles
    │   execution │   control flow
    ├─→ GPU       ├─→ Converts
    │   optimized │   if/for/while
    └─→ Can save  └─→ Pythonic code
        as SavedModel
```

### Basic Usage

```python
# Regular function (eager execution)
def regular_function(x, y):
    return x**2 + y**2

# Decorated function (graph execution)
@tf.function
def graph_function(x, y):
    return x**2 + y**2

# Usage
x = tf.constant(3.0)
y = tf.constant(4.0)

result = graph_function(x, y)
print(result)  # 25.0

# Performance comparison
import time

# Eager execution
start = time.time()
for _ in range(1000):
    _ = regular_function(x, y)
eager_time = time.time() - start

# Graph execution
start = time.time()
for _ in range(1000):
    _ = graph_function(x, y)
graph_time = time.time() - start

print(f"Eager: {eager_time:.4f}s, Graph: {graph_time:.4f}s")
```

### Input Signatures

```python
# Specify input types and shapes
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 28, 28], dtype=tf.float32),
    tf.TensorSpec(shape=[None], dtype=tf.int32)
])
def train_step(images, labels):
    # Function body
    return images, labels

# This ensures the function is only traced once
```

### Control Flow

```python
@tf.function
def conditional_function(x):
    if x > 0:
        return x * 2
    else:
        return x * 3

@tf.function
def loop_function(n):
    result = 0
    for i in tf.range(n):
        result += i
    return result

# Using Python loops (unrolled at trace time)
@tf.function
def python_loop():
    result = 0
    for i in range(10):  # Python range
        result += i
    return result

# Using TensorFlow loops (dynamic)
@tf.function
def tf_loop(n):
    result = tf.constant(0)
    for i in tf.range(n):  # TensorFlow range
        result += i
    return result
```

### Debugging tf.function

```python
# Disable tf.function for debugging
@tf.function
def buggy_function(x):
    tf.print("Debug: x =", x)  # Use tf.print instead of print
    result = x * 2
    return result

# Run in eager mode for debugging
tf.config.run_functions_eagerly(True)
result = buggy_function(tf.constant(5.0))
tf.config.run_functions_eagerly(False)

# Or use py_function for regular Python debugging
@tf.function
def with_python_debug(x):
    tf.py_function(lambda: print(f"Python print: {x}"), [], [])
    return x * 2
```

### AutoGraph Examples

```python
@tf.function
def fibonacci(n):
    a, b = 0, 1
    for _ in tf.range(n):
        a, b = b, a + b
    return a

@tf.function
def dynamic_rnn(input_sequence):
    state = 0
    for element in input_sequence:
        state = state + element
    return state

# Conditional with variables
@tf.function
def clip_gradients(gradients, max_norm):
    clipped = []
    for grad in gradients:
        if grad is not None:
            clipped.append(tf.clip_by_norm(grad, max_norm))
        else:
            clipped.append(None)
    return clipped
```

### Polymorphic Functions

```python
@tf.function
def polymorphic_function(x):
    return x + 1

# First call traces with shape (2,)
result1 = polymorphic_function(tf.constant([1, 2]))

# Second call reuses trace with shape (2,)
result2 = polymorphic_function(tf.constant([3, 4]))

# Third call traces with new shape (3,)
result3 = polymorphic_function(tf.constant([5, 6, 7]))

# Check number of traces
print(polymorphic_function.experimental_get_tracing_count())
```

### XLA Compilation

```python
# Enable XLA compilation for additional performance
@tf.function(jit_compile=True)
def xla_optimized(x, y):
    return tf.matmul(x, y)

# Or use experimental_compile (deprecated)
@tf.function(experimental_compile=True)
def xla_function(x):
    return x**2 + tf.sin(x)
```

## Custom Training with GradientTape

```python
import tensorflow as tf

# Define the model, optimizer, and loss function
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(784,), activation='softmax')])
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Define a training step
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
epochs = 10
for epoch in range(epochs):
    for images, labels in dataset:
        loss = train_step(images, labels)
    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")
```

## Complete End-to-End Example

### Image Classification Pipeline

```python
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Data Preparation
def load_and_preprocess_data():
    """Load and preprocess image data"""
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# 2. Create Data Pipeline
def create_dataset(X, y, batch_size=64, shuffle=True, augment=False):
    """Create optimized tf.data pipeline"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    
    if augment:
        def augment_fn(image, label):
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label
        
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# 3. Build Model
def create_model(input_shape=(28, 28, 1), num_classes=10):
    """Create CNN model"""
    model = tf.keras.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                              input_shape=input_shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# 4. Setup Callbacks
def create_callbacks():
    """Create training callbacks"""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    return callbacks

# 5. Main Training Function
def train_model():
    """Complete training pipeline"""
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create datasets
    train_dataset = create_dataset(X_train, y_train, batch_size=128, augment=True)
    val_dataset = create_dataset(X_val, y_val, batch_size=128, shuffle=False)
    test_dataset = create_dataset(X_test, y_test, batch_size=128, shuffle=False)
    
    # Build model
    print("\nBuilding model...")
    model = create_model()
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=create_callbacks(),
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save final model
    model.save('final_model.h5')
    print("\nModel saved as 'final_model.h5'")
    
    return model, history

# 6. Run Training
if __name__ == '__main__':
    model, history = train_model()
    
    # Make predictions
    print("\nMaking predictions...")
    (_, _), (_, _), (X_test, y_test) = load_and_preprocess_data()
    predictions = model.predict(X_test[:10])
    predicted_classes = np.argmax(predictions, axis=1)
    print(f"Predicted: {predicted_classes}")
    print(f"Actual: {y_test[:10]}")
```

### Text Classification Example

```python
def text_classification_pipeline():
    """Complete text classification pipeline"""
    
    # 1. Load and preprocess text data
    max_words = 10000
    max_len = 200
    
    # Load IMDB dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=max_words
    )
    
    # Pad sequences
    X_train = tf.keras.preprocessing.sequence.pad_sequences(
        X_train, maxlen=max_len
    )
    X_test = tf.keras.preprocessing.sequence.pad_sequences(
        X_test, maxlen=max_len
    )
    
    # 2. Build text model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_words, 128, input_length=max_len),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 3. Compile and train
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )
    
    # 4. Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return model, history

# Run text classification
# model, history = text_classification_pipeline()
```

## Custom Callbacks

```python
import time

class CustomCallback(tf.keras.callbacks.Callback):
    """Custom callback with comprehensive monitoring"""
    
    def on_train_begin(self, logs=None):
        print("=" * 50)
        print("Training Started")
        print("=" * 50)
        self.train_start_time = time.time()
        self.epoch_times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1} Starting")
        print(f"{'='*50}")
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Loss: {logs.get('loss', 0):.4f}")
        print(f"  Accuracy: {logs.get('accuracy', 0):.4f}")
        print(f"  Val Loss: {logs.get('val_loss', 0):.4f}")
        print(f"  Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
        
        # Save model if validation accuracy > threshold
        if logs.get('val_accuracy', 0) > 0.95:
            self.model.save(f'high_acc_model_epoch_{epoch + 1}.h5')
            print(f"  ✓ Saved model (val_acc > 0.95)")
    
    def on_train_batch_end(self, batch, logs=None):
        # Print progress every 100 batches
        if batch % 100 == 0:
            print(f"  Batch {batch}: loss={logs.get('loss', 0):.4f}", end='\r')
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start_time
        avg_epoch_time = np.mean(self.epoch_times)
        
        print(f"\n{'='*50}")
        print("Training Completed")
        print(f"{'='*50}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Epoch Time: {avg_epoch_time:.2f}s")
        print(f"Total Epochs: {len(self.epoch_times)}")

# Use custom callback
model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[CustomCallback()]
)
```

## Mixed Precision Training

```python
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Build model with mixed precision
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax', dtype='float32') # Output layer should be float32
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

## Profiling

```python
import tensorflow as tf

# Profile the training steps 2 to 5
tf.profiler.experimental.start('logdir')

for step in range(10):
    # Your training step here
    with tf.profiler.experimental.Trace('train', step_num=step):
        # ... your training code ...
        pass
tf.profiler.experimental.stop()
```

Then, use TensorBoard to visualize the profiling results:

```bash
tensorboard --logdir logdir
```

## Common Deep Learning Architectures

### Convolutional Neural Network (CNN) for Image Classification

```python
def create_cnn_model(input_shape=(224, 224, 3), num_classes=10):
    """Create a CNN for image classification"""
    model = tf.keras.Sequential([
        # Convolutional Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                              input_shape=input_shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Convolutional Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Convolutional Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_cnn_model()
model.compile(optimizer='adam', 
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
```

### Recurrent Neural Network (RNN) for Sequence Processing

```python
def create_lstm_model(vocab_size=10000, embedding_dim=128, max_length=100):
    """Create an LSTM for text classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.SpatialDropout1D(0.2),
        tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, 
                            return_sequences=True),
        tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

model = create_lstm_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Transfer Learning with Pre-trained Models

```python
def create_transfer_learning_model(num_classes=10):
    """Create a model using transfer learning"""
    # Load pre-trained model without top layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom top layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

model, base_model = create_transfer_learning_model()

# Fine-tune: Unfreeze and train with lower learning rate
# After initial training:
# base_model.trainable = True
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), ...)
```

### Autoencoder for Dimensionality Reduction

```python
def create_autoencoder(input_dim=784, encoding_dim=32):
    """Create an autoencoder"""
    # Encoder
    encoder_input = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(128, activation='relu')(encoder_input)
    encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
    decoder_output = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    # Autoencoder model
    autoencoder = tf.keras.Model(encoder_input, decoder_output)
    
    # Encoder model (for encoding only)
    encoder = tf.keras.Model(encoder_input, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder
```

## Best Practices

### Development Workflow

```
    Problem Definition
           │
           ↓
    ┌──────────────────┐
    │  Data Collection │
    │  & Exploration   │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Preprocessing   │───→ Normalization
    │  & Augmentation  │    Data augmentation
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Model Selection │───→ Start simple
    │                  │    Use transfer learning
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Training        │───→ Monitor metrics
    │                  │    Use callbacks
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Evaluation      │───→ Validation set
    │                  │    Cross-validation
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Hyperparameter  │───→ Grid/Random search
    │  Tuning          │    Bayesian optimization
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Deployment      │───→ SavedModel/TFLite
    └──────────────────┘
```

### Performance Optimization

**Data Pipeline:**
- Use `tf.data` API for efficient data loading
- Apply `prefetch()` to overlap data preprocessing and training
- Use `cache()` to cache preprocessed data in memory
- Set `num_parallel_calls=tf.data.AUTOTUNE` for parallel processing

**Model Training:**
- Use `@tf.function` to convert Python functions to graphs
- Enable mixed precision training with `tf.keras.mixed_precision`
- Use distributed training strategies for multiple GPUs
- Apply XLA compilation with `jit_compile=True`

**Memory Management:**
- Use appropriate batch sizes
- Enable GPU memory growth: `tf.config.experimental.set_memory_growth()`
- Use gradient accumulation for large models
- Apply gradient checkpointing to trade computation for memory

### Model Design

**Architecture Choices:**
- Start with simple models and increase complexity
- Use pre-trained models and transfer learning when possible
- Apply batch normalization after convolutional/dense layers
- Use dropout for regularization (0.2-0.5 for most cases)

**Regularization Techniques:**
- L1/L2 regularization: `kernel_regularizer=tf.keras.regularizers.l2(0.01)`
- Dropout: `tf.keras.layers.Dropout(0.5)`
- Early stopping: `EarlyStopping(patience=5, restore_best_weights=True)`
- Data augmentation for computer vision tasks

**Hyperparameters:**
- Learning rate: Start with 0.001, use learning rate schedulers
- Batch size: 32-256 (larger for GPUs, depends on memory)
- Optimizer: Adam for most cases, SGD with momentum for fine-tuning
- Epochs: Use early stopping instead of fixed number

### Debugging & Monitoring

**Training Monitoring:**
- Use TensorBoard for visualization
- Monitor both training and validation metrics
- Watch for overfitting (val_loss increases while train_loss decreases)
- Check gradient norms to detect vanishing/exploding gradients

**Common Issues:**
- **NaN losses:** Reduce learning rate, check for log(0) or division by zero
- **Slow convergence:** Increase learning rate, check data preprocessing
- **Overfitting:** Add regularization, increase training data
- **Underfitting:** Increase model capacity, train longer

### Code Quality

**Organization:**
- Separate data preprocessing, model definition, and training code
- Use configuration files for hyperparameters
- Version control your code and model checkpoints
- Document model architecture and training procedures

**Reproducibility:**
- Set random seeds: `tf.random.set_seed(42)`
- Save model configurations and hyperparameters
- Track experiments with MLflow or Weights & Biases
- Keep detailed training logs

**Testing:**
- Validate model with unit tests
- Test data pipeline with small batches
- Verify model outputs on known examples
- Benchmark inference speed and memory usage

## Common Issues and Debugging

### Debugging Decision Tree

```
    Training Issue?
           │
    ┌──────┴──────┬───────────┬──────────┐
    ↓             ↓           ↓          ↓
  Loss is      Loss is    Training    Shape/Type
   NaN       Not Going    is Slow     Errors
    │          Down         │           │
    ↓             ↓          ↓           ↓
Check LR    Check Model  Profile    Check Tensor
Gradient    Capacity     Code       Dimensions
Clipping    Learning     Use GPU    Use tf.shape
Data        Rate         tf.data    Cast types
            Data         Mixed
                        Precision
```

### Out of Memory (OOM) Errors

```python
# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Reduce batch size
BATCH_SIZE = 16  # Instead of 64

# Use mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Gradient accumulation for large effective batch size
accumulation_steps = 4
for step, (images, labels) in enumerate(dataset):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions) / accumulation_steps
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    if (step + 1) % accumulation_steps == 0:
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Clear session to free memory
tf.keras.backend.clear_session()
```

### NaN (Not a Number) Losses

```python
# Check for NaN in data
def check_for_nan(data, labels):
    assert not np.isnan(data).any(), "Found NaN in data"
    assert not np.isnan(labels).any(), "Found NaN in labels"

# Use gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

# Or clip gradients manually
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    # Clip gradients
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Add numerical stability to loss functions
# Instead of: tf.math.log(y_pred)
# Use: tf.math.log(y_pred + 1e-7)

# Use stable loss implementations
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Add assertions
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
        tf.debugging.check_numerics(loss, "Loss is NaN or Inf")
    return loss
```

### Slow Training

```python
# Profile training
tf.profiler.experimental.start('logdir')
for step in range(100):
    with tf.profiler.experimental.Trace('train', step_num=step):
        train_step(images, labels)
tf.profiler.experimental.stop()

# Optimize data pipeline
dataset = dataset.cache()  # Cache in memory
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Use XLA compilation
@tf.function(jit_compile=True)
def train_step(images, labels):
    # ... training code ...
    pass

# Mixed precision training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### Shape Mismatches

```python
# Inspect tensor shapes
x = tf.constant([[1, 2], [3, 4]])
print(f"Shape: {x.shape}")           # TensorShape([2, 2])
print(f"Dynamic shape: {tf.shape(x)}")  # Tensor([2, 2])

# Debug shape issues
@tf.function
def debug_shapes(x, y):
    tf.print("x shape:", tf.shape(x))
    tf.print("y shape:", tf.shape(y))
    return x + y

# Assert shapes
def model_forward(x):
    tf.debugging.assert_rank(x, 4, message="Input must be 4D (NHWC)")
    tf.debugging.assert_equal(tf.shape(x)[1:3], [224, 224], 
                             message="Image size must be 224x224")
    return model(x)

# Reshape tensors
x = tf.reshape(x, [-1, 28, 28, 1])  # -1 infers batch dimension
```

### Gradient Issues

```python
# Check gradient norms
@tf.function
def train_step_with_gradient_monitoring(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Check gradient norms
    gradient_norm = tf.sqrt(sum([tf.reduce_sum(g**2) for g in gradients 
                                 if g is not None]))
    tf.print("Gradient norm:", gradient_norm)
    
    # Clip gradients if too large
    if gradient_norm > 10.0:
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Use batch normalization
model = tf.keras.Sequential([
    Conv2D(32, 3, activation='relu'),
    BatchNormalization(),
    # ...
])

# Use skip connections (ResNet-style)
def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x
```

### Overfitting vs Underfitting

```python
# Monitor overfitting
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

# Combat overfitting
model = tf.keras.Sequential([
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

### Debugging Tools

```python
# Use tf.print for debugging inside @tf.function
@tf.function
def debug_function(x):
    tf.print("Input:", x, output_stream=sys.stdout)
    result = x * 2
    tf.print("Output:", result)
    return result

# Use assertions
@tf.function
def safe_divide(x, y):
    tf.debugging.assert_positive(y, message="Divisor must be positive")
    tf.debugging.assert_all_finite(x, message="x contains NaN or Inf")
    return x / y

# Enable eager execution for debugging
tf.config.run_functions_eagerly(True)
# ... debug code ...
tf.config.run_functions_eagerly(False)

# Use tf.py_function to call Python code
def python_debug_function(x):
    print(f"Python print: {x.numpy()}")
    return x

@tf.function
def tf_function_with_python(x):
    tf.py_function(python_debug_function, [x], [])
    return x * 2

# Check model summary
model.summary()

# Visualize model architecture
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
```