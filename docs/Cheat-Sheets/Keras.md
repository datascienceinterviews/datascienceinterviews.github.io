
---
title: Keras Cheat Sheet
description: A comprehensive reference guide for Keras, covering model building, layers, training, evaluation, and more.
---

# Keras Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of the Keras deep learning library, covering essential concepts, code snippets, and best practices for efficient model building, training, and evaluation. It aims to be a one-stop reference for common tasks.

## Getting Started

### Installation

```bash
pip install tensorflow  # Installs TensorFlow with Keras
# or
pip install keras # Installs Keras with a backend (TensorFlow, Theano, or CNTK)
```

### Importing Keras

```python
import tensorflow as tf  # If using TensorFlow backend
from tensorflow import keras
# or
import keras  # If using standalone Keras
```

## Model Building

### Sequential Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
```

### Functional API

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(784,))
x = Dense(128, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
```

### Model Subclassing

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()
```

## Layers

### Core Layers

*   `Dense`: Fully connected layer.
*   `Activation`: Applies an activation function.
*   `Dropout`: Applies dropout regularization.
*   `Flatten`: Flattens the input.
*   `Input`: Creates an input tensor.
*   `Reshape`: Reshapes the input.
*   `Embedding`: Turns positive integers (indexes) into dense vectors of fixed size.

### Convolutional Layers

*   `Conv1D`: 1D convolution layer.
*   `Conv2D`: 2D convolution layer.
*   `Conv3D`: 3D convolution layer.
*   `SeparableConv2D`: Depthwise separable 2D convolution layer.
*   `DepthwiseConv2D`: Depthwise 2D convolution layer.
*   `Conv2DTranspose`: Transposed convolution layer (deconvolution).

### Pooling Layers

*   `MaxPooling1D`, `MaxPooling2D`, `MaxPooling3D`: Max pooling layers.
*   `AveragePooling1D`, `AveragePooling2D`, `AveragePooling3D`: Average pooling layers.
*   `GlobalMaxPooling1D`, `GlobalMaxPooling2D`, `GlobalMaxPooling3D`: Global max pooling layers.
*   `GlobalAveragePooling1D`, `GlobalAveragePooling2D`, `GlobalAveragePooling3D`: Global average pooling layers.

### Recurrent Layers

*   `LSTM`: Long Short-Term Memory layer.
*   `GRU`: Gated Recurrent Unit layer.
*   `SimpleRNN`: Fully-connected RNN where the output is to be fed back to input.
*   `Bidirectional`: Wraps another recurrent layer to run it in both directions.
*   `ConvLSTM2D`: ConvLSTM2D layer.

### Normalization Layers

*   `BatchNormalization`: Applies batch normalization.
*   `LayerNormalization`: Applies layer normalization.

### Advanced Activation Layers

*   `LeakyReLU`: Leaky version of a Rectified Linear Unit.
*   `PReLU`: Parametric Rectified Linear Unit.
*   `ELU`: Exponential Linear Unit.

### Embedding Layers

*   `Embedding`: Turns positive integers (indexes) into dense vectors of fixed size.

### Merge Layers

*   `Add`: Adds inputs.
*   `Multiply`: Multiplies inputs.
*   `Average`: Averages inputs.
*   `Maximum`: Takes the maximum of inputs.
*   `Concatenate`: Concatenates inputs.
*   `Dot`: Performs a dot product between inputs.

### Writing Custom Layers

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

## Activation Functions

*   `relu`: Rectified Linear Unit.
*   `sigmoid`: Sigmoid function.
*   `tanh`: Hyperbolic tangent function.
*   `softmax`: Softmax function (for multi-class classification).
*   `elu`: Exponential Linear Unit.
*   `selu`: Scaled Exponential Linear Unit.
*   `linear`: Linear (identity) activation.
*   `LeakyReLU`: Leaky Rectified Linear Unit.

## Loss Functions

### Regression Losses

*   `MeanSquaredError`: Mean squared error.
*   `MeanAbsoluteError`: Mean absolute error.
*   `MeanAbsolutePercentageError`: Mean absolute percentage error.
*   `MeanSquaredLogarithmicError`: Mean squared logarithmic error.
*   `Huber`: Huber loss.

### Classification Losses

*   `BinaryCrossentropy`: Binary cross-entropy (for binary classification).
*   `CategoricalCrossentropy`: Categorical cross-entropy (for multi-class classification with one-hot encoded labels).
*   `SparseCategoricalCrossentropy`: Sparse categorical cross-entropy (for multi-class classification with integer labels).
*   `Hinge`: Hinge loss (for "maximum-margin" classification).
*   `KLDivergence`: Kullback-Leibler Divergence loss.
*   `Poisson`: Poisson loss.

### Custom Loss Functions

```python
import tensorflow as tf

def my_custom_loss(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`
```

## Optimizers

*   `SGD`: Stochastic Gradient Descent.
*   `Adam`: Adaptive Moment Estimation.
*   `RMSprop`: Root Mean Square Propagation.
*   `Adagrad`: Adaptive Gradient Algorithm.
*   `Adadelta`: Adaptive Delta.
*   `Adamax`: Adamax optimizer from Adam and max operators.
*   `Nadam`: Nesterov Adam optimizer.
*   `Ftrl`: Follow The Regularized Leader optimizer.

### Optimizer Configuration

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
```

## Metrics

*   `Accuracy`: Accuracy.
*   `BinaryAccuracy`: Binary accuracy.
*   `CategoricalAccuracy`: Categorical accuracy.
*   `SparseCategoricalAccuracy`: Sparse categorical accuracy.
*   `TopKCategoricalAccuracy`: Computes how often targets are in the top K predictions.
*   `MeanAbsoluteError`: Mean absolute error.
*   `MeanSquaredError`: Mean squared error.
*   `Precision`: Precision.
*   `Recall`: Recall.
*   `AUC`: Area Under the Curve.
*   `F1Score`: F1 Score.

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

## Model Compilation

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## Training

### Training with NumPy Arrays

```python
import numpy as np

data = np.random.random((1000, 784))
labels = np.random.randint(10, size=(1000,))
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=10)

model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

### Training with tf.data.Dataset

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((data, one_hot_labels))
dataset = dataset.batch(32)

model.fit(dataset, epochs=10)
```

### Validation

```python
val_data = np.random.random((100, 784))
val_labels = np.random.randint(10, size=(100,))
one_hot_val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=10)

model.fit(data, one_hot_labels, epochs=10, batch_size=32,
          validation_data=(val_data, one_hot_val_labels))
```

### Callbacks

*   `ModelCheckpoint`: Saves the model at certain intervals.
*   `EarlyStopping`: Stops training when a monitored metric has stopped improving.
*   `TensorBoard`: Enables visualization of metrics and more.
*   `ReduceLROnPlateau`: Reduces the learning rate when a metric has stopped improving.
*   `CSVLogger`: Streams epoch results to a CSV file.

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5',
                                     save_best_only=True,
                                     monitor='val_loss',
                                     verbose=1)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

model.fit(data, one_hot_labels, epochs=10, batch_size=32,
          validation_data=(val_data, one_hot_val_labels),
          callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback])
```

## Evaluation

```python
loss, accuracy = model.evaluate(val_data, one_hot_val_labels)
print('Loss:', loss)
print('Accuracy:', accuracy)
```

## Prediction

```python
predictions = model.predict(val_data)
predicted_classes = np.argmax(predictions, axis=1)
```

## Saving and Loading Models

### Save the Entire Model

```python
model.save('my_model.h5')  # Saves the model architecture, weights, and optimizer state
```

### Load the Entire Model

```python
from tensorflow.keras.models import load_model

loaded_model = load_model('my_model.h5')
```

### Save Model Architecture as JSON

```python
json_string = model.to_json()
# Save the JSON string to a file
with open('model_architecture.json', 'w') as f:
    f.write(json_string)
```

### Load Model Architecture from JSON

```python
from tensorflow.keras.models import model_from_json

# Load the JSON string from a file
with open('model_architecture.json', 'r') as f:
    json_string = f.read()

model = model_from_json(json_string)
```

### Save Model Weights

```python
model.save_weights('model_weights.h5')
```

### Load Model Weights

```python
model.load_weights('model_weights.h5')
```

## Regularization

### L1 and L2 Regularization

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

### Dropout

```python
from tensorflow.keras.layers import Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.5),  # Dropout layer with 50% dropout rate
    Dense(10, activation='softmax')
])
```

### Batch Normalization

```python
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    BatchNormalization(),  # Batch normalization layer
    Dense(10, activation='softmax')
])
```

## Transfer Learning

### Feature Extraction

```python
from tensorflow.keras.applications import VGG16

# Load pre-trained VGG16 model without the top (classification) layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the base model
base_model.trainable = False

# Add custom classification layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

### Fine-Tuning

```python
# Unfreeze some of the layers in the base model
base_model.trainable = True
for layer in base_model.layers[:-4]:  # Unfreeze the last 4 layers
    layer.trainable = False

# Recompile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Continue training
model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
```

## Callbacks

### ModelCheckpoint

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
```

### EarlyStopping

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
```

### ReduceLROnPlateau

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1
)
```

### TensorBoard

```python
from tensorflow.keras.callbacks import TensorBoard

tensorboard_callback = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)
```

## Custom Training Loops

```python
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
metric_fn = tf.keras.metrics.CategoricalAccuracy()

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    metric_fn.update_state(labels, predictions)
    return loss

epochs = 10
for epoch in range(epochs):
    for images, labels in dataset:
        loss = train_step(images, labels)
    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}, Accuracy: {metric_fn.result().numpy():.4f}")
    metric_fn.reset_state()
```

## Distributed Training

### MirroredStrategy

```python
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

## Hyperparameter Tuning

### Using Keras Tuner

Installation:

```bash
pip install keras-tuner
```

Define a Hypermodel:

```python
from tensorflow import keras
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(
        hp.Choice('units', [32, 64, 128]),
        activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
```

Run the Tuner:

```python
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='my_project')

tuner.search_space_summary()

tuner.search(x_train, y_train,
             epochs=10,
             validation_data=(x_val, y_val))

best_model = tuner.get_best_models(num_models=1)[0]
```

## TensorFlow Datasets

### Installation

```bash
pip install tensorflow-datasets
```

### Usage

```python
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

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

model.fit(ds_train, epochs=12, validation_data=ds_test)
```

## TensorFlow Hub

### Installation

```bash
pip install tensorflow-hub
```

### Usage

```python
import tensorflow_hub as hub

embedding = "https://tfhub.dev/google/nnlm-en-dim128/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

## TensorFlow Lite

### Convert to TensorFlow Lite

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```

## Tips and Best Practices

*   Use virtual environments to isolate project dependencies.
*   Use meaningful names for layers, models, and variables.
*   Follow the DRY (Don't Repeat Yourself) principle.
*   Write unit tests to ensure code quality.
*   Use a consistent coding style.
*   Document your code.
*   Use a version control system (e.g., Git).
*   Use a GPU for training if possible.
*   Monitor your training progress with TensorBoard.
*   Use callbacks to save the best model and stop training early.
*   Use regularization techniques to prevent overfitting.
*   Experiment with different optimizers and learning rates.
*   Use data augmentation to improve model performance.
*   Use transfer learning to leverage pre-trained models.
*   Use a TPU for faster training.
*   Use a distributed training strategy for large datasets.
*   Use a profiler to identify performance bottlenecks.
*   Use a model quantization technique to reduce model size.
*   Use a model pruning technique to reduce model complexity.
*   Use a model distillation technique to create a smaller model.
*   Use a model compression technique to reduce model size.
*   Use a model deployment tool to deploy your model to production.
*   Use a model monitoring tool to monitor your model's performance in production.