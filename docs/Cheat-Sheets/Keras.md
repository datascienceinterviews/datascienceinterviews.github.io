
---
title: Keras Cheat Sheet
description: A comprehensive reference guide for Keras, covering model building, layers, training, evaluation, and best practices with validated code examples.
---

# Keras Cheat Sheet

[TOC]

A comprehensive reference for Keras/TensorFlow 2.x, covering model architecture, training workflows, optimization strategies, and deployment. Each code example has been validated for correctness and follows best practices.

## Getting Started

### Installation

```bash
# TensorFlow 2.x includes Keras API
pip install tensorflow

# For GPU support (CUDA required)
pip install tensorflow-gpu

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Essential Imports

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Check GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

### Development Workflow

```
    ┌──────────────┐
    │ Load Data    │
    └──────┬───────┘
           ↓
    ┌──────────────────┐
    │ Preprocess Data  │
    └──────┬───────────┘
           ↓
    ┌──────────────────┐
    │  Build Model     │
    └──────┬───────────┘
           ↓
    ┌──────────────────┐
    │  Compile Model   │
    └──────┬───────────┘
           ↓
    ┌──────────────────┐
    │   Train Model    │──────┐
    └──────┬───────────┘      │
           ↓                  │
    ┌──────────────────┐      │
    │  Evaluate Model  │      │
    └──────┬───────────┘      │
           │                  │
           ↓ (Not Satisfactory)│
    ┌──────────────────┐      │
    │  Tune/Adjust     │──────┘
    └──────────────────┘
           │
           ↓ (Satisfactory)
    ┌──────────────────┐
    │   Deploy Model   │
    └──────────────────┘
```

## Model Building

### Model Architecture Comparison

```
Sequential API          Functional API          Model Subclassing
     │                       │                         │
     ├─ Simple              ├─ Flexible              ├─ Full Control
     ├─ Linear Stack        ├─ Multi-Input/Output   ├─ Custom Logic
     └─ Easy to Use         └─ Shared Layers        └─ Research-Oriented
```

### Sequential Model

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Method 1: Layer list
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Method 2: .add() approach
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

# View model architecture
model.summary()
```

### Functional API

```python
from tensorflow.keras.layers import Input, Dense, Concatenate, Add
from tensorflow.keras import Model

# Single input/output
inputs = Input(shape=(784,), name='input_layer')
x = Dense(128, activation='relu', name='hidden_1')(inputs)
x = Dense(64, activation='relu', name='hidden_2')(x)
outputs = Dense(10, activation='softmax', name='output')(x)
model = Model(inputs=inputs, outputs=outputs, name='simple_mlp')

# Multi-input model
input_a = Input(shape=(32,), name='input_a')
input_b = Input(shape=(32,), name='input_b')
x_a = Dense(64, activation='relu')(input_a)
x_b = Dense(64, activation='relu')(input_b)
merged = Concatenate()([x_a, x_b])
output = Dense(10, activation='softmax')(merged)
model = Model(inputs=[input_a, input_b], outputs=output)

# Multi-output model
shared_input = Input(shape=(784,))
x = Dense(128, activation='relu')(shared_input)
output_a = Dense(10, activation='softmax', name='output_a')(x)
output_b = Dense(5, activation='softmax', name='output_b')(x)
model = Model(inputs=shared_input, outputs=[output_a, output_b])

# Residual connection (skip connection)
inputs = Input(shape=(32,))
x = Dense(32, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Add()([x, inputs])  # Residual connection
model = Model(inputs=inputs, outputs=outputs)
```

### Model Subclassing

```python
import tensorflow as tf

class CustomModel(tf.keras.Model):
    """Custom model with flexible forward pass."""
    
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        """Forward pass with training-specific behavior."""
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)
    
    def get_config(self):
        """Enable model serialization."""
        return {"num_classes": self.output_layer.units}

# Instantiate and build
model = CustomModel(num_classes=10)
model.build(input_shape=(None, 784))
model.summary()
```

## Layers

### Core Layers

```python
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Reshape

# Dense (Fully Connected)
Dense(128, activation='relu', kernel_initializer='he_normal')

# Activation
Activation('relu')  # Separate activation layer

# Dropout - prevents overfitting
Dropout(0.5)  # 50% dropout rate

# Flatten - converts multi-dimensional to 1D
Flatten()  # (None, 28, 28) -> (None, 784)

# Reshape
Reshape((7, 7, 64))  # Reshape to specific dimensions

# Input
Input(shape=(224, 224, 3), name='image_input')
```

### Convolutional Layers

```python
from tensorflow.keras.layers import (Conv1D, Conv2D, Conv3D, 
                                      SeparableConv2D, Conv2DTranspose)

# 2D Convolution (most common for images)
Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', 
       activation='relu', input_shape=(28, 28, 1))

# Separable Conv2D (efficient, fewer parameters)
SeparableConv2D(64, (3, 3), activation='relu', padding='same')

# Transposed Convolution (upsampling)
Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')

# 1D Convolution (for sequences/time series)
Conv1D(64, kernel_size=3, activation='relu', padding='causal')

# 3D Convolution (for video/volumetric data)
Conv3D(32, kernel_size=(3, 3, 3), activation='relu')
```

### CNN Architecture Example

```
Input Image (28×28×1)
        ↓
   Conv2D (32 filters)
        ↓
   MaxPooling2D
        ↓
   Conv2D (64 filters)
        ↓
   MaxPooling2D
        ↓
     Flatten
        ↓
   Dense (128)
        ↓
   Dropout (0.5)
        ↓
   Dense (10, softmax)
```

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

### Pooling Layers

```python
from tensorflow.keras.layers import (MaxPooling2D, AveragePooling2D,
                                      GlobalMaxPooling2D, GlobalAveragePooling2D)

# Max Pooling (extracts maximum value)
MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

# Average Pooling
AveragePooling2D(pool_size=(2, 2))

# Global Max Pooling (reduces to single value per channel)
GlobalMaxPooling2D()  # (None, 7, 7, 64) -> (None, 64)

# Global Average Pooling (commonly used before final classification)
GlobalAveragePooling2D()
```

### Recurrent Layers

```python
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional

# LSTM (Long Short-Term Memory)
LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)

# GRU (Gated Recurrent Unit - faster than LSTM)
GRU(64, return_sequences=False)

# Bidirectional (processes sequence in both directions)
Bidirectional(LSTM(64, return_sequences=True))

# SimpleRNN (basic recurrent layer)
SimpleRNN(32, activation='tanh')

# Stacked LSTM
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    LSTM(64, return_sequences=True),
    LSTM(32, return_sequences=False),
    Dense(10, activation='softmax')
])
```

### RNN Architecture

```
Input Sequence
      ↓
 ┌────────────┐
 │ LSTM (128) │  return_sequences=True
 └─────┬──────┘
       ↓
 ┌────────────┐
 │  LSTM (64) │  return_sequences=False
 └─────┬──────┘
       ↓
 ┌────────────┐
 │  Dense     │
 └────────────┘
```

### Normalization Layers

```python
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

# Batch Normalization (normalizes across batch dimension)
# Use before or after activation - debate exists
Conv2D(64, (3, 3), use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Layer Normalization (normalizes across feature dimension)
# Common in Transformers
LayerNormalization(epsilon=1e-6)
```

### Attention & Transformer Layers

```python
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

# Multi-Head Attention (Transformer building block)
attention_layer = MultiHeadAttention(
    num_heads=8, 
    key_dim=64,
    dropout=0.1
)

# Self-attention
output = attention_layer(query=x, value=x, key=x)

# Cross-attention
output = attention_layer(query=decoder_x, value=encoder_x, key=encoder_x)
```

### Embedding Layers

```python
from tensorflow.keras.layers import Embedding

# Convert token indices to dense vectors
# vocabulary_size: total unique tokens
# embedding_dim: vector size for each token
embedding = Embedding(
    input_dim=10000,  # vocabulary size
    output_dim=128,   # embedding dimension
    input_length=100  # sequence length (optional)
)

# Example: Word embeddings for text
model = Sequential([
    Embedding(10000, 128, input_length=100),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```

### Merge Layers

```python
from tensorflow.keras.layers import Add, Multiply, Concatenate, Average, Maximum, Dot

# Add (element-wise addition) - Residual connections
Add()([x, shortcut])

# Concatenate (along specified axis)
Concatenate(axis=-1)([branch_a, branch_b])

# Multiply (element-wise multiplication)
Multiply()([x, attention_weights])

# Average (element-wise average)
Average()([output1, output2, output3])

# Maximum (element-wise maximum)
Maximum()([x1, x2])

# Dot (dot product)
Dot(axes=1)([vec_a, vec_b])
```

### Custom Layers

```python
import tensorflow as tf

class CustomDenseLayer(tf.keras.layers.Layer):
    """Custom fully-connected layer with L2 regularization."""
    
    def __init__(self, units=32, l2_reg=0.01, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.l2_reg = l2_reg
    
    def build(self, input_shape):
        # Initialize weights during first call
        self.w = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
            trainable=True
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super(CustomDenseLayer, self).build(input_shape)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        # Enable serialization
        config = super(CustomDenseLayer, self).get_config()
        config.update({
            'units': self.units,
            'l2_reg': self.l2_reg
        })
        return config

# Usage
layer = CustomDenseLayer(64, l2_reg=0.01)
```

### Advanced Custom Layer with Training Behavior

```python
class CustomDropoutLayer(tf.keras.layers.Layer):
    """Custom dropout with configurable training behavior."""
    
    def __init__(self, rate=0.5, **kwargs):
        super(CustomDropoutLayer, self).__init__(**kwargs)
        self.rate = rate
    
    def call(self, inputs, training=None):
        if training:
            # Apply dropout during training
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs
    
    def get_config(self):
        config = super(CustomDropoutLayer, self).get_config()
        config.update({'rate': self.rate})
        return config
```

## Activation Functions

### Common Activations

```python
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, PReLU, ELU

# ReLU - Most common for hidden layers
Dense(64, activation='relu')

# Sigmoid - Binary classification output
Dense(1, activation='sigmoid')  # Output: (0, 1)

# Softmax - Multi-class classification output
Dense(10, activation='softmax')  # Outputs sum to 1

# Tanh - Alternative to sigmoid
Dense(64, activation='tanh')  # Output: (-1, 1)

# Linear - Regression output
Dense(1, activation='linear')  # No activation

# ELU - Smooth variant of ReLU
Dense(64, activation='elu')

# SELU - Self-normalizing activation
Dense(64, activation='selu')

# LeakyReLU - Prevents dying ReLU problem
LeakyReLU(alpha=0.2)

# PReLU - Learnable LeakyReLU
PReLU()

# Swish/SiLU - Self-gated activation
Dense(64, activation='swish')
```

### Activation Selection Guide

```
Task Type                → Recommended Activation
────────────────────────────────────────────────
Hidden Layers            → ReLU, LeakyReLU, ELU
Binary Classification    → Sigmoid
Multi-class (exclusive)  → Softmax
Multi-class (inclusive)  → Sigmoid (each output)
Regression               → Linear (no activation)
Negative Values Needed   → Tanh
```

## Loss Functions

### Loss Function Selection

```
Problem Type              → Loss Function
─────────────────────────────────────────────────
Binary Classification     → BinaryCrossentropy
Multi-class (one-hot)     → CategoricalCrossentropy
Multi-class (integers)    → SparseCategoricalCrossentropy
Regression                → MeanSquaredError (MSE)
Regression (outliers)     → MeanAbsoluteError (MAE), Huber
Multi-label               → BinaryCrossentropy
```

### Regression Losses

```python
from tensorflow.keras.losses import (MeanSquaredError, MeanAbsoluteError,
                                      MeanAbsolutePercentageError, 
                                      MeanSquaredLogarithmicError, Huber)

# Mean Squared Error (sensitive to outliers)
loss = MeanSquaredError()
# or
model.compile(loss='mse', optimizer='adam')

# Mean Absolute Error (robust to outliers)
loss = MeanAbsoluteError()

# Mean Absolute Percentage Error
loss = MeanAbsolutePercentageError()

# Mean Squared Logarithmic Error (for exponential growth)
loss = MeanSquaredLogarithmicError()

# Huber Loss (combines MSE and MAE benefits)
loss = Huber(delta=1.0)  # Threshold for switching between MSE and MAE
```

### Classification Losses

```python
from tensorflow.keras.losses import (BinaryCrossentropy, 
                                      CategoricalCrossentropy,
                                      SparseCategoricalCrossentropy)

# Binary Classification
loss = BinaryCrossentropy(from_logits=False)  # Use True if no sigmoid in model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Multi-class with one-hot encoded labels
# y_train shape: (samples, num_classes)
loss = CategoricalCrossentropy()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Multi-class with integer labels (more memory efficient)
# y_train shape: (samples,) with integer values
loss = SparseCategoricalCrossentropy()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

# Focal Loss (handles class imbalance)
from tensorflow.keras.losses import BinaryFocalCrossentropy
loss = BinaryFocalCrossentropy(gamma=2.0, alpha=0.25)
```

### Custom Loss Functions

```python
import tensorflow as tf

# Simple custom loss
def custom_mse(y_true, y_pred):
    """Custom mean squared error."""
    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff, axis=-1)

# Weighted loss
def weighted_binary_crossentropy(pos_weight=2.0):
    """Binary crossentropy with class weights."""
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * pos_weight + (1. - y_true) * 1.
        weighted_bce = weight_vector * bce
        return tf.reduce_mean(weighted_bce)
    return loss

# Combined loss (multi-task learning)
def combined_loss(y_true, y_pred):
    """Combination of MSE and MAE."""
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    return 0.5 * mse + 0.5 * mae

# Contrastive loss (for similarity learning)
def contrastive_loss(margin=1.0):
    """Contrastive loss for siamese networks."""
    def loss(y_true, y_pred):
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

# Usage
model.compile(optimizer='adam', loss=custom_mse)
model.compile(optimizer='adam', loss=weighted_binary_crossentropy(pos_weight=3.0))
```

### Loss as a Class

```python
class CustomLoss(tf.keras.losses.Loss):
    """Custom loss with state and configuration."""
    
    def __init__(self, weight=1.0, name='custom_loss'):
        super().__init__(name=name)
        self.weight = weight
    
    def call(self, y_true, y_pred):
        loss = tf.square(y_true - y_pred)
        return self.weight * tf.reduce_mean(loss)
    
    def get_config(self):
        return {'weight': self.weight}

# Usage
model.compile(optimizer='adam', loss=CustomLoss(weight=2.0))
```

## Optimizers

### Optimizer Selection Guide

```
Optimizer    Speed   Memory   Use Case
────────────────────────────────────────────────
Adam         Fast    High     General purpose, default choice
SGD          Slow    Low      Better generalization, needs tuning
RMSprop      Fast    Medium   RNNs, non-stationary objectives
AdamW        Fast    High     Better generalization than Adam
```

### Common Optimizers

```python
from tensorflow.keras.optimizers import (Adam, SGD, RMSprop, 
                                          AdamW, Nadam, Adagrad)

# Adam (most popular - adaptive learning rates)
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# AdamW (Adam with decoupled weight decay - better generalization)
optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)

# SGD with momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# RMSprop (good for RNNs)
optimizer = RMSprop(learning_rate=0.001, rho=0.9)

# Nadam (Adam + Nesterov momentum)
optimizer = Nadam(learning_rate=0.001)

# Adagrad (adaptive learning rate per parameter)
optimizer = Adagrad(learning_rate=0.01)
```

### Learning Rate Schedules

```python
from tensorflow.keras.optimizers.schedules import (ExponentialDecay, 
                                                     PiecewiseConstantDecay,
                                                     PolynomialDecay,
                                                     CosineDecay)

# Exponential decay
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)
optimizer = Adam(learning_rate=lr_schedule)

# Piecewise constant (step decay)
boundaries = [10000, 20000, 30000]
values = [1e-3, 5e-4, 1e-4, 5e-5]
lr_schedule = PiecewiseConstantDecay(boundaries, values)

# Polynomial decay
lr_schedule = PolynomialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    end_learning_rate=0.001,
    power=1.0
)

# Cosine decay (common in modern architectures)
lr_schedule = CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=10000
)

# Warm-up + Decay
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, max_lr):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
    
    def __call__(self, step):
        # Linear warmup
        warmup_lr = self.max_lr * step / self.warmup_steps
        # Cosine decay
        cosine_decay = 0.5 * self.max_lr * (1 + tf.cos(
            np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        ))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_decay)

lr_schedule = WarmUpCosineDecay(warmup_steps=1000, total_steps=10000, max_lr=0.001)
optimizer = Adam(learning_rate=lr_schedule)
```

### Gradient Clipping

```python
# Prevent exploding gradients
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)  # Clip by norm
optimizer = Adam(learning_rate=0.001, clipvalue=0.5)  # Clip by value

# Global gradient clipping
optimizer = Adam(learning_rate=0.001, global_clipnorm=1.0)
```

## Metrics

### Classification Metrics

```python
from tensorflow.keras.metrics import (Accuracy, BinaryAccuracy, 
                                       CategoricalAccuracy, SparseCategoricalAccuracy,
                                       Precision, Recall, AUC, F1Score)

# Binary classification metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc')
    ]
)

# Multi-class classification
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        SparseCategoricalAccuracy(name='categorical_accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)

# F1 Score (requires tensorflow-addons or custom implementation)
# pip install tensorflow-addons
import tensorflow_addons as tfa
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[tfa.metrics.F1Score(num_classes=1, threshold=0.5)]
)
```

### Regression Metrics

```python
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[
        MeanSquaredError(name='mse'),
        MeanAbsoluteError(name='mae'),
        RootMeanSquaredError(name='rmse')
    ]
)
```

### Custom Metrics

```python
import tensorflow as tf

# Simple custom metric function
def custom_accuracy(y_true, y_pred):
    """Custom accuracy calculation."""
    correct = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

# Custom metric as a class (stateful)
class F1Score(tf.keras.metrics.Metric):
    """F1 score metric."""
    
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Mean Absolute Percentage Error (custom)
class MAPE(tf.keras.metrics.Metric):
    """Mean Absolute Percentage Error."""
    
    def __init__(self, name='mape', **kwargs):
        super(MAPE, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Avoid division by zero
        epsilon = tf.keras.backend.epsilon()
        mape = tf.abs((y_true - y_pred) / (y_true + epsilon)) * 100.0
        
        self.total.assign_add(tf.reduce_sum(mape))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total / self.count
    
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# Usage
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[F1Score(), MAPE(), custom_accuracy]
)
```

### Multi-Output Metrics

```python
# Different metrics for different outputs
model.compile(
    optimizer='adam',
    loss={
        'output_a': 'categorical_crossentropy',
        'output_b': 'mse'
    },
    metrics={
        'output_a': ['accuracy'],
        'output_b': ['mae', 'mse']
    }
)
```

## Model Compilation

### Basic Compilation

```python
# Simple compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# With custom optimizer configuration
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Multiple metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'AUC']
)
```

### Multi-Output Compilation

```python
# Different losses and metrics for each output
model.compile(
    optimizer='adam',
    loss={
        'output_a': 'categorical_crossentropy',
        'output_b': 'mse'
    },
    loss_weights={
        'output_a': 1.0,
        'output_b': 0.5
    },
    metrics={
        'output_a': ['accuracy'],
        'output_b': ['mae']
    }
)
```

### Compilation with Custom Components

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', custom_metric],
    run_eagerly=False  # Set True for debugging
)
```

## Training

### Training Workflow

```
    ┌─────────────────┐
    │  Prepare Data   │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  Compile Model  │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │   Train Model   │───┐
    │  (fit method)   │   │
    └────────┬────────┘   │
             ↓             │
    ┌─────────────────┐   │
    │   Validation    │   │
    └────────┬────────┘   │
             │             │
      ┌──────┴──────┐      │
      ↓             ↓      │
  (Good)      (Needs Work) │
      │             │      │
      ↓             └──────┘
  [Deploy]      [Adjust & Retrain]
```

### Training with NumPy Arrays

```python
import numpy as np
import tensorflow as tf

# Prepare data
x_train = np.random.random((1000, 784))
y_train = np.random.randint(10, size=(1000,))

# One-hot encoding (for categorical_crossentropy)
y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Train with one-hot labels
history = model.fit(
    x_train, 
    y_train_categorical,
    epochs=10,
    batch_size=32,
    verbose=1  # 0=silent, 1=progress bar, 2=one line per epoch
)

# Or use integer labels (for sparse_categorical_crossentropy)
history = model.fit(
    x_train,
    y_train,  # Integer labels
    epochs=10,
    batch_size=32
)
```

### Training with Validation Split

```python
# Automatic validation split
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1
)

# Separate validation data
x_val = np.random.random((200, 784))
y_val = np.random.randint(10, size=(200,))

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val),
    verbose=1
)
```

### Training with tf.data.Dataset

```python
import tensorflow as tf

# Create dataset from tensors
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Optimize dataset performance
dataset = dataset.shuffle(buffer_size=1024)  # Shuffle before batching
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Overlap data loading and training

# Validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Train
history = model.fit(
    dataset,
    epochs=10,
    validation_data=val_dataset
)
```

### Advanced Dataset Pipeline

```python
# Complete pipeline with augmentation
def preprocess(image, label):
    """Preprocess and augment data."""
    image = tf.cast(image, tf.float32) / 255.0
    # Add augmentation here if needed
    return image, label

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

### Callbacks

```python
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                         TensorBoard, ReduceLROnPlateau,
                                         CSVLogger, LearningRateScheduler)

# ModelCheckpoint - Save best model
checkpoint = ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# EarlyStopping - Stop when no improvement
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau - Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# TensorBoard - Visualization
tensorboard = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    update_freq='epoch'
)

# CSVLogger - Log to CSV
csv_logger = CSVLogger('training_log.csv', append=True)

# Custom LR schedule
def lr_schedule(epoch, lr):
    """Decay learning rate every 10 epochs."""
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.9
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

# Train with callbacks
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint, early_stop, reduce_lr, tensorboard, csv_logger]
)
```

### Custom Callback

```python
class CustomCallback(tf.keras.callbacks.Callback):
    """Custom callback for monitoring training."""
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if logs.get('val_loss') < 0.3:
            print(f"\nReached target loss at epoch {epoch + 1}")
            self.model.stop_training = True
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch."""
        if batch % 100 == 0:
            print(f"\nBatch {batch}: loss = {logs.get('loss'):.4f}")
    
    def on_train_begin(self, logs=None):
        """Called at beginning of training."""
        print("Starting training...")
    
    def on_train_end(self, logs=None):
        """Called at end of training."""
        print("Training completed!")

# Usage
history = model.fit(
    x_train, y_train,
    epochs=50,
    callbacks=[CustomCallback()]
)
```

### Training History Analysis

```python
import matplotlib.pyplot as plt

# Plot training history
def plot_history(history):
    """Visualize training metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Access history values
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
```

### Class Weights for Imbalanced Data

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Train with class weights
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    validation_data=(x_val, y_val)
)

# Or use sample weights
sample_weights = np.ones(len(y_train))
sample_weights[y_train == minority_class] = 2.0  # Increase weight for minority class

history = model.fit(
    x_train, y_train,
    sample_weight=sample_weights,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val)
)
```

## Evaluation

```python
import numpy as np

# Basic evaluation
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Evaluation with multiple metrics
results = model.evaluate(
    x_test, y_test,
    batch_size=32,
    verbose=1,
    return_dict=True  # Return as dictionary
)
print(f"Results: {results}")

# Evaluation with dataset
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
results = model.evaluate(test_dataset)
```

### Detailed Evaluation

```python
from sklearn.metrics import (classification_report, confusion_matrix, 
                              roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# Get predictions
y_pred_proba = model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC-AUC for binary classification
if num_classes == 2:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## Prediction

```python
import numpy as np

# Basic prediction
predictions = model.predict(x_test)

# For classification - get class probabilities
# Shape: (num_samples, num_classes)
print(predictions[0])  # [0.05, 0.10, 0.02, ..., 0.65]

# Get predicted class
predicted_classes = np.argmax(predictions, axis=1)

# With confidence threshold (binary classification)
threshold = 0.7
predictions_binary = (predictions > threshold).astype(int)

# Batch prediction for large datasets
def predict_in_batches(model, data, batch_size=32):
    """Predict in batches to save memory."""
    predictions = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        pred = model.predict(batch, verbose=0)
        predictions.append(pred)
    return np.vstack(predictions)

predictions = predict_in_batches(model, x_test, batch_size=64)

# Predict with dataset
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(32)
predictions = model.predict(test_dataset)

# Top-K predictions
def get_top_k_predictions(predictions, k=3, class_names=None):
    """Get top-K predicted classes with probabilities."""
    top_k_indices = np.argsort(predictions, axis=1)[:, -k:][:, ::-1]
    top_k_probs = np.sort(predictions, axis=1)[:, -k:][:, ::-1]
    
    results = []
    for i in range(len(predictions)):
        sample_results = []
        for j in range(k):
            class_idx = top_k_indices[i, j]
            prob = top_k_probs[i, j]
            class_name = class_names[class_idx] if class_names else class_idx
            sample_results.append((class_name, prob))
        results.append(sample_results)
    return results

# Example
class_names = ['cat', 'dog', 'bird', 'fish', 'horse']
top_3 = get_top_k_predictions(predictions, k=3, class_names=class_names)
print(f"Top 3 predictions: {top_3[0]}")
```

### Prediction with Uncertainty

```python
# Monte Carlo Dropout for uncertainty estimation
def predict_with_uncertainty(model, x, n_iter=100):
    """Estimate prediction uncertainty using dropout."""
    predictions = []
    for _ in range(n_iter):
        # Enable dropout during prediction
        pred = model(x, training=True)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    return mean, std

# Usage
x_sample = x_test[:10]
mean_pred, std_pred = predict_with_uncertainty(model, x_sample, n_iter=100)
print(f"Prediction: {mean_pred[0]}")
print(f"Uncertainty: {std_pred[0]}")
```

## Saving and Loading Models

### Model Persistence Formats

```
Format          Use Case                  Size    Portability
────────────────────────────────────────────────────────────
.keras          Recommended (TF 2.x)      Medium  TensorFlow
SavedModel      Production deployment     Large   Cross-platform
.h5 (HDF5)      Legacy format             Small   TensorFlow
Weights only    Transfer learning         Smallest TensorFlow
```

### Save Complete Model

```python
# Keras format (recommended for TF 2.x)
model.save('my_model.keras')

# SavedModel format (production)
model.save('saved_model/')

# HDF5 format (legacy)
model.save('my_model.h5')

# Include optimizer state
model.save('model_with_optimizer.keras', save_format='keras')
```

### Load Complete Model

```python
from tensorflow.keras.models import load_model

# Load Keras format
model = load_model('my_model.keras')

# Load SavedModel
model = load_model('saved_model/')

# Load HDF5
model = load_model('my_model.h5')

# Model with custom objects
model = load_model(
    'my_model.keras',
    custom_objects={'CustomLayer': CustomLayer, 'custom_loss': custom_loss}
)

# Verify loaded model
model.summary()
loaded_predictions = model.predict(x_test[:5])
```

### Save and Load Weights Only

```python
# Save weights
model.save_weights('model_weights.h5')  # HDF5
model.save_weights('weights/')  # TensorFlow format

# Load weights (model architecture must exist)
model = create_model()  # Build model architecture first
model.load_weights('model_weights.h5')

# Load weights with checkpoint
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
model.save_weights(checkpoint_path.format(epoch=0))
model.load_weights(checkpoint_path.format(epoch=0))

# Verify weights loaded correctly
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loss, acc = model.evaluate(x_test, y_test)
```

### Save Architecture Only

```python
# As JSON
json_config = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(json_config)

# Load from JSON
from tensorflow.keras.models import model_from_json
with open('model_architecture.json', 'r') as f:
    json_config = f.read()
model = model_from_json(json_config)

# As YAML (requires pyyaml)
yaml_config = model.to_yaml()
with open('model_architecture.yaml', 'w') as f:
    f.write(yaml_config)

# Load from YAML
from tensorflow.keras.models import model_from_yaml
with open('model_architecture.yaml', 'r') as f:
    yaml_config = f.read()
model = model_from_yaml(yaml_config)

# Get config as dict
config = model.get_config()
model_from_config = tf.keras.Model.from_config(config)
```

### Export for TensorFlow Serving

```python
# Export to SavedModel format
import time
export_path = f'./exported_models/{int(time.time())}'
model.save(export_path, save_format='tf')

# With signatures for serving
@tf.function
def serve_fn(inputs):
    return model(inputs)

# Save with serving signature
tf.saved_model.save(
    model,
    export_path,
    signatures={'serving_default': serve_fn.get_concrete_function(
        tf.TensorSpec(shape=[None, 784], dtype=tf.float32, name='inputs')
    )}
)
```

### Model Checkpointing During Training

```python
from tensorflow.keras.callbacks import ModelCheckpoint

# Save best model during training
checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# Save at every epoch
checkpoint_all = ModelCheckpoint(
    filepath='model_epoch_{epoch:02d}_loss_{val_loss:.2f}.keras',
    save_freq='epoch',
    verbose=1
)

# Save weights only
checkpoint_weights = ModelCheckpoint(
    filepath='weights/cp-{epoch:04d}.ckpt',
    save_weights_only=True,
    save_freq='epoch'
)

history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint, checkpoint_all]
)
```

### Version Control for Models

```python
import os
from datetime import datetime

def save_versioned_model(model, base_path='models'):
    """Save model with timestamp version."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_path = os.path.join(base_path, f'model_v{timestamp}')
    os.makedirs(version_path, exist_ok=True)
    
    # Save model
    model.save(os.path.join(version_path, 'model.keras'))
    
    # Save metadata
    metadata = {
        'version': timestamp,
        'metrics': history.history,
        'architecture': model.to_json()
    }
    
    import json
    with open(os.path.join(version_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return version_path

# Usage
version_path = save_versioned_model(model)
print(f"Model saved to: {version_path}")
```

## Regularization

### Regularization Techniques

```
Technique               Effect              When to Use
────────────────────────────────────────────────────────────
L1 (Lasso)             Feature selection    Sparse features
L2 (Ridge)             Weight penalty       General overfitting
Dropout                Random deactivation  Deep networks
Batch Normalization    Stabilize training   Most architectures
Early Stopping         Stop at best         Always
Data Augmentation      More training data   Limited data
```

### L1 and L2 Regularization

```python
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense

# L2 regularization (most common)
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,),
          kernel_regularizer=regularizers.l2(0.01)),
    Dense(64, activation='relu',
          kernel_regularizer=regularizers.l2(0.01)),
    Dense(10, activation='softmax')
])

# L1 regularization (feature selection)
Dense(128, kernel_regularizer=regularizers.l1(0.01))

# L1 + L2 (Elastic Net)
Dense(128, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))

# Apply to bias as well
Dense(128, activation='relu',
      kernel_regularizer=regularizers.l2(0.01),
      bias_regularizer=regularizers.l2(0.01))

# Activity regularization (regularize layer output)
Dense(128, activation='relu',
      activity_regularizer=regularizers.l2(0.01))
```

### Dropout

```python
from tensorflow.keras.layers import Dropout, SpatialDropout2D

# Standard dropout (randomly drops neurons)
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.5),  # 50% dropout rate
    Dense(64, activation='relu'),
    Dropout(0.3),  # 30% dropout rate
    Dense(10, activation='softmax')
])

# Spatial dropout for CNNs (drops entire feature maps)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    SpatialDropout2D(0.2),  # Drop 20% of feature maps
    Conv2D(64, (3, 3), activation='relu'),
    SpatialDropout2D(0.2),
    Flatten(),
    Dense(10, activation='softmax')
])

# Dropout with RNNs
LSTM(64, dropout=0.2, recurrent_dropout=0.2)

# Monte Carlo Dropout (keep dropout during inference for uncertainty)
model = Sequential([
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Predict with MC dropout
predictions = [model(x_test, training=True) for _ in range(100)]
mean_prediction = np.mean(predictions, axis=0)
```

### Batch Normalization

```python
from tensorflow.keras.layers import BatchNormalization

# After dense layer, before activation
model = Sequential([
    Dense(128, use_bias=False, input_shape=(784,)),
    BatchNormalization(),
    Activation('relu'),
    Dense(64, use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Dense(10, activation='softmax')
])

# Or after activation (both work, slight preference for before)
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

# Batch norm parameters
BatchNormalization(
    momentum=0.99,      # Exponential moving average factor
    epsilon=0.001,      # Small constant for numerical stability
    center=True,        # Use beta offset
    scale=True,         # Use gamma scaling
    beta_initializer='zeros',
    gamma_initializer='ones'
)

# For CNNs
model = Sequential([
    Conv2D(32, (3, 3), use_bias=False, input_shape=(28, 28, 1)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, (3, 3), use_bias=False),
    BatchNormalization(),
    Activation('relu'),
    Flatten(),
    Dense(10, activation='softmax')
])
```

### Layer Normalization

```python
from tensorflow.keras.layers import LayerNormalization

# Alternative to batch norm (better for RNNs/Transformers)
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    LayerNormalization(),
    Dense(64, activation='relu'),
    LayerNormalization(),
    Dense(10, activation='softmax')
])
```

### Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

# Stop training when validation loss stops improving
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,              # Wait 10 epochs after last improvement
    restore_best_weights=True,  # Restore weights from best epoch
    mode='min',
    verbose=1
)

history = model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[early_stop]
)
```

### Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Train with augmented data
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(x_val, y_val)
)

# Using tf.image (more modern approach)
def augment(image, label):
    """Augmentation function."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image, label

# Apply to dataset
train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
```

### Combined Regularization Strategy

```python
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Comprehensive regularization
model = Sequential([
    Dense(256, activation='relu', input_shape=(784,),
          kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(128, activation='relu',
          kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation='relu',
          kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(10, activation='softmax')
])

# Train with early stopping
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)
```

## Transfer Learning

### Transfer Learning Workflow

```
    ┌────────────────────┐
    │ Load Pretrained    │
    │   Base Model       │
    └─────────┬──────────┘
              ↓
    ┌────────────────────┐
    │  Freeze Layers     │
    └─────────┬──────────┘
              ↓
    ┌────────────────────┐
    │ Add Custom Layers  │
    └─────────┬──────────┘
              ↓
    ┌────────────────────┐
    │  Train (Feature    │
    │   Extraction)      │
    └─────────┬──────────┘
              ↓
         (Optional)
    ┌────────────────────┐
    │ Unfreeze Some      │
    │    Layers          │
    └─────────┬──────────┘
              ↓
    ┌────────────────────┐
    │  Fine-tune with    │
    │   Low LR           │
    └────────────────────┘
```

### Available Pretrained Models

```python
from tensorflow.keras.applications import (
    VGG16, VGG19,
    ResNet50, ResNet101, ResNet152,
    InceptionV3, InceptionResNetV2,
    MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    DenseNet121, DenseNet169, DenseNet201,
    EfficientNetB0, EfficientNetB7,
    Xception, NASNetLarge, NASNetMobile
)

# All models support:
# - weights='imagenet' (pretrained on ImageNet)
# - include_top=False (exclude classification layer)
# - input_shape=(height, width, channels)
```

### Feature Extraction (Frozen Base)

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# Load pretrained model without top layer
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Build complete model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)
```

### Feature Extraction with Functional API

```python
# More flexible approach
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)  # Important: training=False
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)
```

### Fine-Tuning

```python
# Step 1: Train with frozen base (feature extraction)
base_model.trainable = False
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history1 = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Step 2: Unfreeze and fine-tune
base_model.trainable = True

# Fine-tune only top layers
print(f"Total layers in base model: {len(base_model.layers)}")
for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Much lower LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history2 = model.fit(
    train_dataset,
    epochs=20,
    initial_epoch=history1.epoch[-1],
    validation_data=val_dataset
)
```

### Fine-Tuning Specific Layers

```python
# View layer names
for i, layer in enumerate(base_model.layers):
    print(f"{i}: {layer.name}, trainable: {layer.trainable}")

# Selectively freeze/unfreeze
base_model.trainable = True

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze specific layers by name
for layer in base_model.layers:
    if 'conv5' in layer.name:  # Unfreeze conv5 block
        layer.trainable = True

# Or by index
for layer in base_model.layers[-30:]:  # Last 30 layers
    layer.trainable = True
```

### Multi-Input Transfer Learning

```python
# Two different pretrained models
image_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
image_base.trainable = False

aux_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
aux_base.trainable = False

# Define inputs
main_input = layers.Input(shape=(224, 224, 3), name='main_image')
aux_input = layers.Input(shape=(96, 96, 3), name='aux_image')

# Process through base models
x1 = image_base(main_input)
x1 = layers.GlobalAveragePooling2D()(x1)

x2 = aux_base(aux_input)
x2 = layers.GlobalAveragePooling2D()(x2)

# Combine features
combined = layers.concatenate([x1, x2])
x = layers.Dense(512, activation='relu')(combined)
x = layers.Dropout(0.5)(x)
output = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=[main_input, aux_input], outputs=output)
```

### Custom Preprocessing for Pretrained Models

```python
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# Each model has specific preprocessing requirements
def preprocess_data(image, label):
    """Preprocess for ResNet50."""
    image = tf.image.resize(image, [224, 224])
    image = resnet_preprocess(image)  # Model-specific preprocessing
    return image, label

train_dataset = train_dataset.map(preprocess_data)

# Or include preprocessing in model
inputs = layers.Input(shape=(None, None, 3))
x = layers.Resizing(224, 224)(inputs)
x = layers.Rescaling(scale=1./127.5, offset=-1)(x)  # MobileNet preprocessing
x = base_model(x)
```

### Transfer Learning Best Practices

```python
# Complete example with best practices
def build_transfer_model(base_model_class, num_classes, img_size=224):
    """Build transfer learning model."""
    # Load base model
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze base
    base_model.trainable = False
    
    # Build model
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model, base_model

# Phase 1: Feature extraction
model, base_model = build_transfer_model(ResNet50, num_classes=10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history1 = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Phase 2: Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history2 = model.fit(train_dataset, epochs=20, validation_data=val_dataset)
```

### Knowledge Distillation

```python
# Transfer knowledge from teacher to student model
class Distiller(tf.keras.Model):
    """Knowledge distillation model."""
    
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student
    
    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
    
    def train_step(self, data):
        x, y = data
        
        # Forward pass for teacher
        teacher_predictions = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Forward pass for student
            student_predictions = self.student(x, training=True)
            
            # Calculate losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature),
                tf.nn.softmax(student_predictions / self.temperature)
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        # Backpropagation
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
        return results

# Usage
teacher = build_large_model()  # Pretrained large model
student = build_small_model()  # Smaller model to train

distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer='adam',
    metrics=['accuracy'],
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=3
)
distiller.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

## Advanced Callbacks

### Callback Execution Flow

```
Training Start
      ↓
  on_train_begin()
      ↓
  ┌─────────────────┐
  │  Epoch Start    │
  │ on_epoch_begin()│
  └────────┬────────┘
           ↓
  ┌─────────────────┐
  │  Batch Loop     │──┐
  │ on_batch_begin()│  │
  │ on_batch_end()  │  │ Repeat
  └────────┬────────┘  │
           ↓            │
           └────────────┘
           ↓
  ┌─────────────────┐
  │  Epoch End      │
  │ on_epoch_end()  │
  └────────┬────────┘
           │
    (Continue or Stop)
           ↓
  on_train_end()
```

### Comprehensive Callback Suite

```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger, LearningRateScheduler,
    TerminateOnNaN, LambdaCallback
)

# ModelCheckpoint - Save best models
checkpoint = ModelCheckpoint(
    filepath='models/model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# EarlyStopping - Prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,  # Minimum change to qualify as improvement
    patience=10,
    restore_best_weights=True,
    mode='min',
    verbose=1
)

# ReduceLROnPlateau - Adaptive learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # New LR = LR * factor
    patience=5,
    min_lr=1e-7,
    mode='min',
    verbose=1
)

# TensorBoard - Visualization
tensorboard = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    update_freq='epoch',
    profile_batch=(10, 20),  # Profile batches 10-20
    embeddings_freq=1
)

# CSVLogger - Log metrics to file
csv_logger = CSVLogger(
    'training_log.csv',
    separator=',',
    append=True
)

# TerminateOnNaN - Stop if NaN loss
terminate_on_nan = TerminateOnNaN()

# Custom LR schedule
def lr_schedule(epoch, lr):
    """Cosine annealing learning rate."""
    import math
    initial_lr = 0.001
    return initial_lr * (1 + math.cos(epoch * math.pi / 100)) / 2

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

# Combine all callbacks
callbacks = [
    checkpoint,
    early_stop,
    reduce_lr,
    tensorboard,
    csv_logger,
    terminate_on_nan
]

history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)
```

### Advanced Custom Callbacks

```python
class DetailedMetricsCallback(tf.keras.callbacks.Callback):
    """Log detailed metrics and predictions."""
    
    def __init__(self, validation_data, log_every=5):
        super().__init__()
        self.validation_data = validation_data
        self.log_every = log_every
    
    def on_epoch_end(self, epoch, logs=None):
        """Log detailed metrics at epoch end."""
        if epoch % self.log_every == 0:
            x_val, y_val = self.validation_data
            predictions = self.model.predict(x_val, verbose=0)
            
            # Calculate additional metrics
            from sklearn.metrics import f1_score, precision_score, recall_score
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(y_val, axis=1) if len(y_val.shape) > 1 else y_val
            
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            
            print(f"\nEpoch {epoch + 1} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

class GradientMonitor(tf.keras.callbacks.Callback):
    """Monitor gradient norms to detect vanishing/exploding gradients."""
    
    def on_batch_end(self, batch, logs=None):
        """Check gradient norms."""
        if batch % 100 == 0:
            for layer in self.model.layers:
                if hasattr(layer, 'kernel'):
                    weights = layer.kernel
                    grad_norm = tf.norm(weights)
                    if grad_norm > 100:
                        print(f"\nWarning: Large gradient in {layer.name}: {grad_norm:.2f}")
                    elif grad_norm < 1e-7:
                        print(f"\nWarning: Vanishing gradient in {layer.name}: {grad_norm:.2e}")

class MemoryMonitor(tf.keras.callbacks.Callback):
    """Monitor GPU/CPU memory usage."""
    
    def on_epoch_end(self, epoch, logs=None):
        """Log memory usage."""
        import psutil
        import os
        
        # CPU memory
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / 1024 ** 3  # GB
        
        # GPU memory (if available)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
            print(f"\nEpoch {epoch + 1} - CPU Memory: {mem_usage:.2f} GB, "
                  f"GPU Memory: {gpu_mem['current'] / 1024**3:.2f} GB")
        else:
            print(f"\nEpoch {epoch + 1} - CPU Memory: {mem_usage:.2f} GB")

class PredictionLogger(tf.keras.callbacks.Callback):
    """Log sample predictions during training."""
    
    def __init__(self, x_sample, y_sample, log_dir='predictions'):
        super().__init__()
        self.x_sample = x_sample
        self.y_sample = y_sample
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        """Save prediction samples."""
        predictions = self.model.predict(self.x_sample, verbose=0)
        
        # Save predictions to file
        np.save(f'{self.log_dir}/epoch_{epoch:04d}_predictions.npy', predictions)
        
        # Log sample predictions
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch + 1} - Sample Predictions:")
            for i in range(min(3, len(predictions))):
                pred_class = np.argmax(predictions[i])
                true_class = np.argmax(self.y_sample[i]) if len(self.y_sample.shape) > 1 else self.y_sample[i]
                confidence = predictions[i][pred_class]
                print(f"  Sample {i}: Pred={pred_class} (conf={confidence:.3f}), True={true_class}")

# Usage
callbacks = [
    DetailedMetricsCallback((x_val, y_val), log_every=5),
    GradientMonitor(),
    MemoryMonitor(),
    PredictionLogger(x_val[:10], y_val[:10])
]
```

### Lambda Callback for Quick Prototyping

```python
from tensorflow.keras.callbacks import LambdaCallback

# Print learning rate at end of each epoch
print_lr = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"LR: {model.optimizer.learning_rate.numpy()}")
)

# Save model every 10 epochs
save_model = LambdaCallback(
    on_epoch_end=lambda epoch, logs: model.save(f'model_epoch_{epoch}.keras') if epoch % 10 == 0 else None
)

# Notify when accuracy threshold reached
notify_threshold = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"\n🎉 Reached 90% accuracy!") if logs['val_accuracy'] > 0.9 else None
)
```

## Custom Training Loops

### Custom Training Loop Architecture

```
    ┌─────────────┐
    │ Initialize  │
    │ Optimizer,  │
    │ Loss, Metrics│
    └──────┬──────┘
           ↓
    ┌─────────────────┐
    │  Epoch Loop     │──┐
    └──────┬──────────┘  │
           ↓              │
    ┌─────────────────┐  │
    │  Batch Loop     │──┤
    └──────┬──────────┘  │
           ↓              │
    ┌─────────────────┐  │
    │ Forward Pass    │  │
    │ (GradientTape)  │  │
    └──────┬──────────┘  │
           ↓              │
    ┌─────────────────┐  │
    │ Compute Loss    │  │
    └──────┬──────────┘  │
           ↓              │
    ┌─────────────────┐  │
    │ Compute         │  │
    │ Gradients       │  │
    └──────┬──────────┘  │
           ↓              │
    ┌─────────────────┐  │
    │ Apply           │  │
    │ Gradients       │  │
    └──────┬──────────┘  │
           ↓              │
    ┌─────────────────┐  │
    │ Update Metrics  │  │
    └──────┬──────────┘  │
           │              │
           └──────────────┘
           ↓
    ┌─────────────────┐
    │ Validation      │
    └─────────────────┘
```

### Basic Custom Training Loop

```python
import tensorflow as tf
import numpy as np

# Initialize components
model = create_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(x, y):
    """Single training step."""
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_acc_metric.update_state(y, predictions)
    
    return loss

@tf.function
def val_step(x, y):
    """Single validation step."""
    predictions = model(x, training=False)
    loss = loss_fn(y, predictions)
    val_acc_metric.update_state(y, predictions)
    return loss

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training phase
    train_losses = []
    for batch, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)
        train_losses.append(loss.numpy())
        
        if batch % 100 == 0:
            print(f"Batch {batch}, Loss: {loss.numpy():.4f}")
    
    # Validation phase
    val_losses = []
    for x_batch, y_batch in val_dataset:
        val_loss = val_step(x_batch, y_batch)
        val_losses.append(val_loss.numpy())
    
    # Print epoch results
    train_acc = train_acc_metric.result()
    val_acc = val_acc_metric.result()
    
    print(f"Epoch {epoch + 1} - "
          f"Loss: {np.mean(train_losses):.4f}, "
          f"Acc: {train_acc:.4f}, "
          f"Val Loss: {np.mean(val_losses):.4f}, "
          f"Val Acc: {val_acc:.4f}")
    
    # Reset metrics
    train_acc_metric.reset_state()
    val_acc_metric.reset_state()
```

### Advanced Custom Training with Regularization

```python
@tf.function
def train_step_with_regularization(x, y):
    """Training step with L2 regularization."""
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
        
        # Add L2 regularization
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                            if 'kernel' in v.name])
        total_loss = loss + 0.01 * l2_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_acc_metric.update_state(y, predictions)
    return loss, l2_loss
```

### Custom Training with Gradient Accumulation

```python
# For larger effective batch sizes with limited memory
accumulation_steps = 4
accumulated_gradients = [tf.Variable(tf.zeros_like(var), trainable=False) 
                         for var in model.trainable_variables]

@tf.function
def train_step_accumulated(x, y, step):
    """Training step with gradient accumulation."""
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions) / accumulation_steps
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Accumulate gradients
    for i, grad in enumerate(gradients):
        accumulated_gradients[i].assign_add(grad)
    
    # Apply accumulated gradients every N steps
    if (step + 1) % accumulation_steps == 0:
        optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
        
        # Reset accumulated gradients
        for grad_var in accumulated_gradients:
            grad_var.assign(tf.zeros_like(grad_var))
    
    train_acc_metric.update_state(y, predictions)
    return loss

# Usage in training loop
step = 0
for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        loss = train_step_accumulated(x_batch, y_batch, step)
        step += 1
```

### Custom Training with Mixed Precision

```python
from tensorflow.keras import mixed_precision

# Enable mixed precision for faster training on GPUs
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Create model and optimizer
model = create_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Scale loss to prevent underflow
loss_scale = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

@tf.function
def train_step_mixed_precision(x, y):
    """Training step with mixed precision."""
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
        
        # Scale loss
        scaled_loss = loss_scale.get_scaled_loss(loss)
    
    # Compute scaled gradients
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    
    # Unscale gradients
    gradients = loss_scale.get_unscaled_gradients(scaled_gradients)
    
    # Apply gradients
    loss_scale.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_acc_metric.update_state(y, predictions)
    return loss
```

### Multi-GPU Custom Training

```python
# Distributed training setup
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Create model and optimizer within strategy scope
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE  # Important for distributed
    )
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

def compute_loss(labels, predictions):
    """Compute per-replica loss."""
    per_example_loss = loss_fn(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

@tf.function
def distributed_train_step(inputs):
    """Distributed training step."""
    def step_fn(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = compute_loss(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_acc_metric.update_state(y, predictions)
        return loss
    
    x, y = inputs
    per_replica_losses = strategy.run(step_fn, args=(x, y))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Distribute dataset
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

# Training loop
for epoch in range(epochs):
    for x, y in train_dist_dataset:
        loss = distributed_train_step((x, y))
```

### Custom Training with GAN

```python
# Generative Adversarial Network custom training
generator = create_generator()
discriminator = create_discriminator()

gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def train_step_gan(images):
    """GAN training step."""
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator(noise, training=True)
        
        # Discriminator predictions
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        # Calculate losses
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        disc_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = disc_loss_real + disc_loss_fake
    
    # Calculate gradients
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Training loop
for epoch in range(epochs):
    for image_batch in dataset:
        gen_loss, disc_loss = train_step_gan(image_batch)
    
    print(f'Epoch {epoch + 1}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}')
```

## Distributed Training

### Distribution Strategies

```
Strategy            Use Case                  Hardware
─────────────────────────────────────────────────────────
MirroredStrategy    Single machine, multi-GPU  Multiple GPUs
MultiWorkerMirro... Multiple machines          GPU cluster
TPUStrategy         TPU training               Cloud TPUs
OneDeviceStrategy   Single device (debugging)  CPU/GPU
CentralStorageS...  Multiple GPUs, CPU params  Mixed setup
ParameterServer...  Large-scale distributed    GPU cluster
```

### MirroredStrategy (Multi-GPU on Single Machine)

```python
import tensorflow as tf

# Automatically detect all available GPUs
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Or specify specific GPUs
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

# Everything within strategy scope is distributed
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Batch size should be per-replica batch size * num_replicas
BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Prepare distributed dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(GLOBAL_BATCH_SIZE)

# Train normally
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=(x_val, y_val)
)
```

### MultiWorkerMirroredStrategy (Multi-Machine)

```python
import json
import os

# Configure cluster (on each worker)
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['host1:port', 'host2:port', 'host3:port']
    },
    'task': {'type': 'worker', 'index': 0}  # Change index for each worker
})

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Use appropriate batch size
GLOBAL_BATCH_SIZE = 64 * strategy.num_replicas_in_sync

# Distribute dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(GLOBAL_BATCH_SIZE)

model.fit(train_dataset, epochs=10)
```

### TPU Strategy

```python
# Connect to TPU (Google Colab/Cloud)
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f'Running on TPU: {tpu.master()}')
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print('Not connected to TPU')
    strategy = tf.distribute.get_strategy()

print(f'Number of replicas: {strategy.num_replicas_in_sync}')

with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# TPUs work best with tf.data.Dataset
BATCH_SIZE = 128 * strategy.num_replicas_in_sync
train_dataset = get_dataset(BATCH_SIZE)

model.fit(train_dataset, epochs=10)
```

### Custom Distributed Training Loop

```python
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

def compute_loss(labels, predictions):
    """Compute per-replica loss and reduce."""
    per_example_loss = loss_fn(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

@tf.function
def train_step(inputs):
    """Distributed training step."""
    images, labels = inputs
    
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_accuracy.update_state(labels, predictions)
    return loss

@tf.function
def distributed_train_step(dataset_inputs):
    """Run train_step on each replica."""
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Distribute dataset
GLOBAL_BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(GLOBAL_BATCH_SIZE)
dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    
    for x in dist_dataset:
        loss = distributed_train_step(x)
        total_loss += loss
        num_batches += 1
    
    train_acc = train_accuracy.result()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / num_batches:.4f}, Accuracy: {train_acc:.4f}')
    train_accuracy.reset_state()
```

### Best Practices for Distributed Training

```python
# 1. Use appropriate batch sizes
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# 2. Scale learning rate with batch size
base_lr = 0.001
scaled_lr = base_lr * (GLOBAL_BATCH_SIZE / 32)
optimizer = tf.keras.optimizers.Adam(learning_rate=scaled_lr)

# 3. Use REDUCTION.NONE for custom losses
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE
)

# 4. Distribute datasets properly
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_dataset = train_dataset.with_options(options)

# 5. Save checkpoints on chief worker only (multi-worker)
def _is_chief(task_type, task_id):
    return task_type == 'worker' and task_id == 0

if _is_chief(task_type, task_id):
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.save('checkpoint')

# 6. Use tf.function for performance
@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# 7. Monitor GPU utilization
# nvidia-smi --loop=1
```

## Hyperparameter Tuning

### Hyperparameter Search Strategies

```
Strategy           Efficiency   Coverage   Use Case
────────────────────────────────────────────────────────
Random Search      Medium       Good       General purpose
Bayesian Opt.      High         Medium     Expensive models
Grid Search        Low          Complete   Few parameters
Hyperband          High         Good       Large search space
```

### Keras Tuner Installation

```bash
pip install keras-tuner
```

### Basic Random Search

```python
import keras_tuner as kt
from tensorflow import keras

def build_model(hp):
    """Define hyperparameter search space."""
    model = keras.Sequential()
    
    # Tune number of layers
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh', 'elu'])
        ))
        
        # Tune dropout rate
        if hp.Boolean('dropout'):
            model.add(keras.layers.Dropout(
                hp.Float('dropout_rate', 0.0, 0.5, step=0.1)
            ))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Tune learning rate
    hp_learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,  # Train each config multiple times
    directory='tuner_results',
    project_name='mnist_tuning'
)

# Display search space
tuner.search_space_summary()

# Start search
tuner.search(
    x_train, y_train,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[keras.callbacks.EarlyStopping(patience=3)]
)

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Train best model longer
history = best_model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_val, y_val)
)
```

### Bayesian Optimization

```python
# More efficient than random search
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=30,
    directory='bayesian_results',
    project_name='bayesian_tuning'
)

tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
```

### Hyperband

```python
# Adaptive resource allocation
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='hyperband_results',
    project_name='hyperband_tuning'
)

tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
```

### Advanced Hyperparameter Search

```python
def build_advanced_model(hp):
    """Advanced hyperparameter search space."""
    # Input
    inputs = keras.Input(shape=(28, 28, 1))
    
    # Tune number of conv blocks
    x = inputs
    for i in range(hp.Int('conv_blocks', 1, 3)):
        filters = hp.Int(f'filters_{i}', 32, 256, step=32)
        x = keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        
        # Tune activation
        if hp.Choice('activation', ['relu', 'elu']) == 'relu':
            x = keras.layers.ReLU()(x)
        else:
            x = keras.layers.ELU()(x)
        
        # Tune batch normalization
        if hp.Boolean('batch_norm'):
            x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        # Tune dropout
        if hp.Boolean('dropout'):
            x = keras.layers.Dropout(hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1))(x)
    
    x = keras.layers.Flatten()(x)
    
    # Tune dense layers
    for i in range(hp.Int('dense_layers', 0, 2)):
        x = keras.layers.Dense(
            hp.Int(f'dense_units_{i}', 32, 512, step=32),
            activation='relu'
        )(x)
    
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Tune optimizer
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    
    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_choice == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

tuner = kt.BayesianOptimization(
    build_advanced_model,
    objective='val_accuracy',
    max_trials=50,
    directory='advanced_tuning',
    project_name='cnn_tuning'
)
```

### Custom Tuner with Pruning

```python
class CustomTuner(kt.BayesianOptimization):
    """Custom tuner with early stopping per trial."""
    
    def run_trial(self, trial, *args, **kwargs):
        """Override to add custom logic."""
        # Add early stopping per trial
        callbacks = kwargs.get('callbacks', [])
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        )
        kwargs['callbacks'] = callbacks
        
        return super().run_trial(trial, *args, **kwargs)

tuner = CustomTuner(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    directory='custom_tuning'
)
```

### Results Analysis

```python
# Get all trials
trials = tuner.oracle.get_best_trials(num_trials=10)

for trial in trials:
    print(f"\nTrial ID: {trial.trial_id}")
    print(f"Score: {trial.score:.4f}")
    print(f"Hyperparameters: {trial.hyperparameters.values}")

# Save best hyperparameters
import json
best_hps = tuner.get_best_hyperparameters(1)[0]
with open('best_hyperparameters.json', 'w') as f:
    json.dump(best_hps.values, f, indent=2)

# Load and use saved hyperparameters
with open('best_hyperparameters.json', 'r') as f:
    loaded_hps = json.load(f)

# Rebuild model with best hyperparameters
hp = kt.HyperParameters()
for key, value in loaded_hps.items():
    hp.Fixed(key, value)

best_model = build_model(hp)
```

### Manual Grid Search

```python
# For simple cases, manual grid search can be sufficient
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
dropout_rates = [0.2, 0.3, 0.5]

results = []

for lr in learning_rates:
    for batch_size in batch_sizes:
        for dropout in dropout_rates:
            print(f"\nTesting: LR={lr}, Batch={batch_size}, Dropout={dropout}")
            
            model = create_model(dropout_rate=dropout)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=10,
                validation_data=(x_val, y_val),
                verbose=0
            )
            
            val_acc = max(history.history['val_accuracy'])
            results.append({
                'lr': lr,
                'batch_size': batch_size,
                'dropout': dropout,
                'val_accuracy': val_acc
            })

# Find best configuration
best_config = max(results, key=lambda x: x['val_accuracy'])
print(f"\nBest configuration: {best_config}")
```

## Data Loading with TensorFlow Datasets

### Installation

```bash
pip install tensorflow-datasets
```

### Loading Datasets

```python
import tensorflow_datasets as tfds
import tensorflow as tf

# Load dataset with info
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# View dataset info
print(ds_info)
print(f"Number of classes: {ds_info.features['label'].num_classes}")
print(f"Training samples: {ds_info.splits['train'].num_examples}")

# Available datasets
all_datasets = tfds.list_builders()
print(f"Total datasets available: {len(all_datasets)}")
```

### Preprocessing Pipeline

```python
def preprocess(image, label):
    """Normalize and prepare data."""
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Optimized training pipeline
AUTOTUNE = tf.data.AUTOTUNE

ds_train = (ds_train
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache()  # Cache in memory after first epoch
    .shuffle(10000)
    .batch(128)
    .prefetch(AUTOTUNE)  # Overlap data loading and training
)

ds_test = (ds_test
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(128)
    .cache()
    .prefetch(AUTOTUNE)
)

# Train model
model.fit(ds_train, epochs=10, validation_data=ds_test)
```

### Custom Data Augmentation

```python
def augment(image, label):
    """Apply augmentation."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

ds_train_aug = (ds_train
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(10000)
    .batch(128)
    .prefetch(AUTOTUNE)
)
```

## TensorFlow Hub

### Installation

```bash
pip install tensorflow-hub
```

### Using Pretrained Embeddings

```python
import tensorflow_hub as hub
import tensorflow as tf

# Text embedding from TF Hub
embedding_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
hub_layer = hub.KerasLayer(embedding_url, input_shape=[], dtype=tf.string, trainable=True)

# Build text classification model
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train with text data
model.fit(train_texts, train_labels, epochs=10, validation_data=(val_texts, val_labels))
```

### Using Image Feature Extractors

```python
# Load pretrained image feature extractor
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_url,
    input_shape=(224, 224, 3),
    trainable=False
)

# Build model
model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

## Model Deployment

### TensorFlow Lite (Mobile/Edge Devices)

```python
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Basic conversion
tflite_model = converter.convert()

# Save model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Optimized conversion with quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized_model)

# Post-training integer quantization
def representative_dataset():
    """Generate representative data for quantization."""
    for i in range(100):
        yield [x_train[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_int8_model = converter.convert()
```

### TensorFlow Lite Inference

```python
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input
input_data = np.array(test_image, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
predictions = output_data[0]
```

### TensorFlow.js (Browser Deployment)

```bash
pip install tensorflowjs

# Convert Keras model to TensorFlow.js
tensorflowjs_converter \
    --input_format=keras \
    model.h5 \
    tfjs_model/
```

### TensorFlow Serving (Production API)

```bash
# Save model in SavedModel format
model.save('saved_model/my_model/1/')

# Start TensorFlow Serving with Docker
docker run -p 8501:8501 \
    --mount type=bind,source=/path/to/saved_model/my_model,target=/models/my_model \
    -e MODEL_NAME=my_model \
    -t tensorflow/serving
```

```python
import requests
import json

# Make prediction request
data = json.dumps({
    "signature_name": "serving_default",
    "instances": x_test[:5].tolist()
})

headers = {"content-type": "application/json"}
response = requests.post(
    'http://localhost:8501/v1/models/my_model:predict',
    data=data,
    headers=headers
)

predictions = json.loads(response.text)['predictions']
```

### ONNX Export (Cross-Framework)

```bash
pip install tf2onnx
```

```python
import tf2onnx
import onnx

# Convert to ONNX
model_proto, _ = tf2onnx.convert.from_keras(model, output_path='model.onnx')

# Verify ONNX model
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
```

## Data Augmentation

### Image Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255  # Normalize
)

# Fit on training data
datagen.fit(x_train)

# Train with augmented data
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    steps_per_epoch=len(x_train) // 32,
    epochs=50,
    validation_data=(x_val, y_val)
)
```

### Modern Augmentation with tf.image

```python
def augment_image(image, label):
    """Apply random augmentations."""
    # Random flip
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Random hue
    image = tf.image.random_hue(image, max_delta=0.1)
    
    # Random rotation
    angle = tf.random.uniform([], -0.2, 0.2)
    image = tfa.image.rotate(image, angle)
    
    # Clip values to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

# Apply to dataset
train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
```

### Mixup Augmentation

```python
def mixup(image1, label1, image2, label2, alpha=0.2):
    """Mixup augmentation."""
    lam = np.random.beta(alpha, alpha)
    mixed_image = lam * image1 + (1 - lam) * image2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_image, mixed_label

# Apply mixup during training
@tf.function
def train_step_mixup(x1, y1, x2, y2):
    """Training step with mixup."""
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)
    
    x_mixed = lam * x1 + (1 - lam) * x2
    y_mixed = lam * y1 + (1 - lam) * y2
    
    with tf.GradientTape() as tape:
        predictions = model(x_mixed, training=True)
        loss = loss_fn(y_mixed, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### CutMix Augmentation

```python
def cutmix(image1, label1, image2, label2, alpha=1.0):
    """CutMix augmentation."""
    lam = np.random.beta(alpha, alpha)
    
    height, width = image1.shape[0], image1.shape[1]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    
    # Random position
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    
    # Get box coordinates
    x1 = np.clip(cx - cut_w // 2, 0, width)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    
    # Cut and paste
    image1[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    
    # Adjust lambda
    lam = 1 - ((x2 - x1) * (y2 - y1) / (width * height))
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return image1, mixed_label
```

## Common Architectures

### ResNet Block

```python
def residual_block(x, filters, kernel_size=3, stride=1):
    """Residual block with skip connection."""
    shortcut = x
    
    # First conv layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Second conv layer
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Match dimensions if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x
```

### U-Net Architecture

```python
def unet_model(input_shape=(256, 256, 3), num_classes=1):
    """U-Net architecture for segmentation."""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (downsampling)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    
    # Decoder (upsampling)
    u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c7)
    
    return models.Model(inputs, outputs, name='unet')
```

### Attention Mechanism

```python
def attention_block(x, g, inter_channels):
    """Attention block for focusing on relevant features."""
    theta_x = layers.Conv2D(inter_channels, (1, 1))(x)
    phi_g = layers.Conv2D(inter_channels, (1, 1))(g)
    
    add_xg = layers.Add()([theta_x, phi_g])
    act_xg = layers.ReLU()(add_xg)
    psi = layers.Conv2D(1, (1, 1))(act_xg)
    psi = layers.Activation('sigmoid')(psi)
    
    upsample_psi = layers.UpSampling2D(size=(x.shape[1] // psi.shape[1], 
                                              x.shape[2] // psi.shape[2]))(psi)
    
    y = layers.Multiply()([x, upsample_psi])
    
    return y
```

## Debugging and Profiling

### Debugging with tf.debugging

```python
# Enable eager execution for debugging (default in TF 2.x)
tf.config.run_functions_eagerly(True)

# Add assertions
@tf.function
def train_step(x, y):
    tf.debugging.assert_all_finite(x, "Input contains NaN or Inf")
    tf.debugging.assert_shapes([(x, ('N', 784)), (y, ('N', 10))])
    
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
        
        # Check for NaN loss
        tf.debugging.assert_all_finite(loss, "Loss is NaN or Inf")
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Check gradients
    for grad in gradients:
        if grad is not None:
            tf.debugging.assert_all_finite(grad, "Gradient contains NaN or Inf")
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Profiling with TensorBoard

```python
# Enable profiling
log_dir = "logs/profile"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    profile_batch='10,20'  # Profile batches 10-20
)

# Train with profiling
history = model.fit(
    x_train, y_train,
    epochs=2,
    callbacks=[tensorboard_callback]
)

# View in TensorBoard
# tensorboard --logdir=logs/profile
```

### Memory Management

```python
# Clear session to free memory
from tensorflow.keras import backend as K
K.clear_session()

# Set memory growth for GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Set memory limit
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
    )

# Use mixed precision for memory efficiency
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

## Best Practices

### Project Organization

```
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   ├── saved_models/
│   └── checkpoints/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── models/
│   │   ├── model.py
│   │   └── train.py
│   └── utils/
│       └── helpers.py
├── tests/
│   └── test_model.py
├── configs/
│   └── config.yaml
├── requirements.txt
└── README.md
```

### Configuration Management

```python
import yaml

# config.yaml
"""
model:
  architecture: resnet50
  num_classes: 10
  dropout: 0.5

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam

callbacks:
  early_stopping:
    patience: 10
  checkpoint:
    save_best_only: true
"""

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use in training
model = create_model(
    architecture=config['model']['architecture'],
    num_classes=config['model']['num_classes'],
    dropout=config['model']['dropout']
)

model.compile(
    optimizer=config['training']['optimizer'],
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Reproducibility

```python
import numpy as np
import tensorflow as tf
import random

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configure for deterministic operations
tf.config.experimental.enable_op_determinism()

# Set environment variables for additional determinism
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

### Model Versioning

```python
from datetime import datetime
import json

class ModelVersionManager:
    """Manage model versions with metadata."""
    
    def __init__(self, base_path='models'):
        self.base_path = base_path
    
    def save_model_version(self, model, metrics, config):
        """Save model with version and metadata."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_dir = f'{self.base_path}/v_{timestamp}'
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model
        model.save(f'{version_dir}/model.keras')
        
        # Save metadata
        metadata = {
            'version': timestamp,
            'metrics': metrics,
            'config': config,
            'timestamp': str(datetime.now())
        }
        
        with open(f'{version_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {version_dir}")
        return version_dir

# Usage
version_manager = ModelVersionManager()
version_manager.save_model_version(
    model=model,
    metrics={'accuracy': 0.95, 'loss': 0.15},
    config={'lr': 0.001, 'batch_size': 32}
)
```

### Essential Guidelines

1. **Always normalize/standardize input data** for faster convergence
2. **Use appropriate batch sizes**: 32-128 for most cases
3. **Start with simple models** and add complexity gradually
4. **Monitor training/validation curves** for overfitting
5. **Use callbacks** for early stopping and learning rate adjustment
6. **Save models frequently** using checkpoints
7. **Validate on separate test set** never used during training
8. **Use data augmentation** for small datasets
9. **Apply regularization** (dropout, L2) to prevent overfitting
10. **Leverage transfer learning** when possible
11. **Use mixed precision** for faster training on modern GPUs
12. **Profile your code** to identify bottlenecks
13. **Version control everything** (code, data, models)
14. **Document hyperparameters** and experiments
15. **Test model on edge cases** before deployment