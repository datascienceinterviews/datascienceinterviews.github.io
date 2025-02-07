
---
title: TensorFlow Cheat Sheet
description: A comprehensive reference guide for TensorFlow 2.x, covering tensors, operations, models, layers, training, and more.
---

# TensorFlow Cheat Sheet

[TOC]

This cheat sheet provides an exhaustive overview of TensorFlow 2.x, covering essential concepts, code snippets, and best practices for efficient deep learning model building, training, evaluation, and deployment. It aims to be a one-stop reference for common tasks.

## Getting Started

### Installation

```bash
pip install tensorflow
```

For GPU support:

```bash
pip install tensorflow-gpu  # (Deprecated in TF 2.10)
# For newer versions, TensorFlow automatically uses GPU if available
```

### Importing TensorFlow

```python
import tensorflow as tf
```

### Checking Version

```python
print(tf.__version__)
```

### Checking GPU Availability

```python
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

## Tensors

### Creating Tensors

```python
# Constant tensors
a = tf.constant([[1, 2], [3, 4]])
b = tf.zeros((2, 3))
c = tf.ones((3, 2))
d = tf.eye(3)  # Identity matrix
e = tf.random.normal((2, 2))  # Normal distribution
f = tf.random.uniform((2, 2))  # Uniform distribution

# From NumPy arrays
import numpy as np
arr = np.array([1, 2, 3])
tensor_from_np = tf.convert_to_tensor(arr)

# Sequences
range_tensor = tf.range(start=0, limit=10, delta=2) # 0, 2, 4, 6, 8
linspace_tensor = tf.linspace(start=0.0, stop=1.0, num=5) # 0.0, 0.25, 0.5, 0.75, 1.0
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
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Element-wise operations
c = a + b       # Addition
d = a - b       # Subtraction
e = a * b       # Multiplication
f = a / b       # Division
g = tf.add(a, b) # Functional form
h = tf.multiply(a, b) # Functional form

# Matrix multiplication
i = tf.matmul(a, b)

# Transpose
j = tf.transpose(a)

# Reshape
k = tf.reshape(a, (1, 4))

# Squeeze and Expand
l = tf.squeeze(tf.constant([[[1], [2], [3]]])) # Removes dimensions of size 1
m = tf.expand_dims(tf.constant([1, 2, 3]), axis=0) # Adds a dimension of size 1

# Concatenation
n = tf.concat([a, b], axis=0)  # Concatenate along rows
o = tf.stack([a, b], axis=0)   # Stack along a new dimension

# Reduce operations
p = tf.reduce_sum(a)        # Sum of all elements
q = tf.reduce_mean(a)       # Mean of all elements
r = tf.reduce_max(a)        # Maximum element
s = tf.reduce_min(a)        # Minimum element
t = tf.reduce_prod(a)       # Product of all elements

# Argmax and Argmin
u = tf.argmax(a, axis=1)    # Index of the maximum element along axis 1
v = tf.argmin(a, axis=0)    # Index of the minimum element along axis 0

# Casting
w = tf.cast(a, tf.float32)  # Cast to float32
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

```python
var = tf.Variable([1.0, 2.0])
var.assign([3.0, 4.0])
var.assign_add([1.0, 1.0])
```

## Automatic Differentiation (Autograd)

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2

dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())  # Output: 6.0
```

### Persistent Gradient Tape

```python
x = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    y = x**2
    z = y * 2

dy_dx = tape.gradient(y, x)  # 6.0
dz_dx = tape.gradient(z, x)  # 12.0
print(dy_dx.numpy())
print(dz_dx.numpy())

del tape  # Drop the reference to the tape
```

### Watching Non-Variable Tensors

```python
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * x
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())
```

## Keras API

### Model Building

#### Sequential Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
```

#### Functional API

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(784,))
x = Dense(128, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
```

#### Model Subclassing

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None): # Add training argument
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()
```

### Layers

*   `tf.keras.layers.Dense`: Fully connected layer.
*   `tf.keras.layers.Conv2D`: 2D convolution layer.
*   `tf.keras.layers.MaxPooling2D`: Max pooling layer.
*   `tf.keras.layers.ReLU`: ReLU activation function.
*   `tf.keras.layers.Activation`: Applies an activation function.
*   `tf.keras.layers.Softmax`: Softmax activation function.
*   `tf.keras.layers.BatchNormalization`: Batch normalization layer.
*   `tf.keras.layers.Dropout`: Dropout layer.
*   `tf.keras.layers.Flatten`: Flattens the input.
*   `tf.keras.layers.Reshape`: Reshapes the input.
*   `tf.keras.layers.Embedding`: Embedding layer.
*   `tf.keras.layers.LSTM`: LSTM layer.
*   `tf.keras.layers.GRU`: GRU layer.
*   `tf.keras.layers.Bidirectional`: Bidirectional wrapper for RNNs.
*   `tf.keras.layers.Input`: Creates an input tensor.
*   `tf.keras.layers.Add`, `tf.keras.layers.Multiply`, `tf.keras.layers.Concatenate`: Merge layers.
*   `tf.keras.layers.GlobalAveragePooling2D`, `tf.keras.layers.GlobalMaxPooling2D`: Global pooling layers.

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

```python
import numpy as np

data = np.random.random((1000, 784))
labels = np.random.randint(10, size=(1000,))
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=10)

model.fit(data, one_hot_labels, epochs=10, batch_size=32, validation_split=0.2)
```

### Evaluation

```python
loss, accuracy = model.evaluate(data, one_hot_labels)
print('Loss:', loss)
print('Accuracy:', accuracy)
```

### Prediction

```python
predictions = model.predict(data)
predicted_classes = np.argmax(predictions, axis=1)
```

### Saving and Loading Models

```python
# Save the entire model
model.save('my_model.h5')

# Load the entire model
loaded_model = tf.keras.models.load_model('my_model.h5')

# Save model weights
model.save_weights('my_model_weights.h5')

# Load model weights
model.load_weights('my_model_weights.h5')
```

### Callbacks

*   `tf.keras.callbacks.ModelCheckpoint`: Saves the model at certain intervals.
*   `tf.keras.callbacks.EarlyStopping`: Stops training when a monitored metric has stopped improving.
*   `tf.keras.callbacks.TensorBoard`: Enables visualization of metrics and more.
*   `tf.keras.callbacks.ReduceLROnPlateau`: Reduces the learning rate when a metric has stopped improving.
*   `tf.keras.callbacks.CSVLogger`: Streams epoch results to a CSV file.
*   `tf.keras.callbacks.LearningRateScheduler`: Schedules the learning rate.
*   `tf.keras.callbacks.TerminateOnNaN`: Terminates training when a NaN loss is encountered.

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

## Data Input Pipelines (tf.data)

### Creating Datasets

```python
import tensorflow as tf

# From NumPy arrays
dataset = tf.data.Dataset.from_tensor_slices((data, one_hot_labels))

# From a list of files
dataset = tf.data.Dataset.list_files("path/to/data/*.tfrecord")

# From a generator
def my_generator():
    for i in range(1000):
        yield i, i**2

dataset = tf.data.Dataset.from_generator(my_generator, output_types=(tf.int32, tf.int32))
```

### Dataset Transformations

*   `dataset.batch(batch_size)`: Combines consecutive elements into batches.
*   `dataset.shuffle(buffer_size)`: Randomly shuffles the elements of the dataset.
*   `dataset.repeat(count=None)`: Repeats the dataset (indefinitely if `count` is None).
*   `dataset.map(map_func)`: Applies a function to each element.
*   `dataset.prefetch(buffer_size)`: Prefetches elements for performance.
*   `dataset.cache()`: Caches the elements of the dataset.
*   `dataset.filter(predicate)`: Filters elements based on a predicate.
*   `dataset.interleave(map_func, cycle_length=None, block_length=None)`: Maps `map_func` across the dataset and interleaves the results.
*   `dataset.flat_map(map_func)`: Maps `map_func` across the dataset and flattens the result.
*   `dataset.take(count)`: Creates a dataset with at most `count` elements.
*   `dataset.skip(count)`: Skips the first `count` elements.
*   `dataset.zip(datasets)`: Zips together multiple datasets.

```python
dataset = tf.data.Dataset.from_tensor_slices((data, one_hot_labels))
dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
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

### MirroredStrategy

```python
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model.fit(data, one_hot_labels, epochs=10, batch_size=32)
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

## TensorFlow Hub

### Using Pre-trained Models

```python
import tensorflow_hub as hub

model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
                   trainable=False),  # Feature extraction
    tf.keras.layers.Dense(10, activation='softmax')
])

model.build([None, 224, 224, 3])  # Build the model
```

## TensorFlow Lite

### Converting to TensorFlow Lite

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Quantization

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

### Inference with TensorFlow Lite

```python
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input tensor
input_data = np.array(np.random.random_sample(input_details[0]['shape']), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
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

```python
@tf.function
def my_function(x, y):
    return x + y

# Call the function
result = my_function(tf.constant(1), tf.constant(2))
print(result)
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

## Custom Callbacks

```python
class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch}, loss: {logs['loss']:.4f}")

    def on_train_batch_begin(self, batch, logs=None):
        print(f"Training: Starting batch {batch}")

    def on_train_batch_end(self, batch, logs=None):
        print(f"Training: Finished batch {batch}, loss: {logs['loss']:.4f}")

model.fit(data, one_hot_labels, epochs=10, callbacks=[MyCustomCallback()])
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

## Best Practices

*   **Use `tf.data` for efficient input pipelines:**  `tf.data` provides optimized data loading and preprocessing.
*   **Use `tf.function` to compile your functions into graphs:** This can significantly improve performance.
*   **Use mixed precision training on compatible GPUs:** This can speed up training and reduce memory usage.
*   **Use distributed training strategies for large models and datasets:** Distribute the workload across multiple GPUs or machines.
*   **Use TensorBoard to monitor training progress:** Visualize metrics, graphs, and more.
*   **Save and restore your models regularly:** Use checkpoints to save your model's progress.
*   **Use Keras whenever possible:** The Keras API is generally easier to use and more intuitive than the lower-level TensorFlow APIs.
*   **Use pre-trained models and transfer learning:** Leverage existing models to speed up development and improve performance.
*   **Regularize your models to prevent overfitting:** Use techniques like dropout, L1/L2 regularization, and batch normalization.
*   **Tune your hyperparameters:** Use techniques like grid search, random search, or Bayesian optimization to find the best hyperparameters for your model.
*   **Validate your models carefully:** Use a separate validation set to evaluate your model's performance and prevent overfitting.
*   **Use appropriate data types:** Use `tf.float32` for most computations, but consider `tf.float16` for mixed precision training.
*   **Vectorize your operations:** Avoid using Python loops when possible; use TensorFlow's vectorized operations instead.
*   **Use XLA (Accelerated Linear Algebra) for further performance improvements:** Add `@tf.function(experimental_compile=True)` to your functions.
*   **Profile your code:** Use the TensorFlow Profiler to identify performance bottlenecks.
*   **Keep your TensorFlow version up-to-date:** Newer versions often include performance improvements and bug fixes.
*   **Read the TensorFlow documentation:** The TensorFlow documentation is comprehensive and well-written.

## Common Issues and Debugging

*   **Out of Memory (OOM) Errors:**
    *   Reduce batch size.
    *   Use mixed precision training (`tf.float16`).
    *   Use gradient accumulation.
    *   Use a smaller model.
    *   Use gradient checkpointing.
    *   Free up memory by deleting unused tensors and variables.
    *   Use `tf.config.experimental.set_memory_growth(gpu, True)` to allow GPU memory to grow as needed (instead of allocating all at once).

*   **NaN (Not a Number) Losses:**
    *   Reduce the learning rate.
    *   Use gradient clipping.
    *   Check for numerical instability (e.g., division by zero, taking the logarithm of a non-positive number).
    *   Use a different optimizer.
    *   Initialize weights appropriately.
    *   Use batch normalization.
    *   Check your data for errors (e.g., NaN values).

*   **Slow Training:**
    *   Use a GPU.
    *   Use `tf.data` for efficient input pipelines.
    *   Use mixed precision training.
    *   Use distributed training.
    *   Use XLA compilation.
    *   Profile your code to identify bottlenecks.
    *   Increase batch size (if memory allows).
    *   Use asynchronous data loading.
    *   Use prefetching.

*   **Shape Mismatches:**
    *   Carefully check the shapes of your tensors and ensure they are compatible with the operations you are performing.
    *   Use `tf.shape` and `tf.reshape` to inspect and modify tensor shapes.

*   **Data Type Errors:**
    *   Ensure that your tensors have the correct data types (e.g., `tf.float32` for floating-point operations, `tf.int64` for indices).
    *   Use `tf.cast` to convert between data types.

*   **Device Placement Errors:**
    *   Ensure that all tensors and operations are placed on the same device (CPU or GPU).
    *   Use `tf.device` to explicitly specify the device.
    *   Use `tf.distribute.Strategy` for distributed training.

*   **Gradient Issues (Vanishing/Exploding Gradients):**
    *   Use gradient clipping.
    *   Use batch normalization.
    *   Use skip connections (e.g., ResNet).
    *   Use a different activation function (e.g., ReLU, LeakyReLU).
    *   Use a smaller learning rate.
    *   Use a different optimizer.

*   **Overfitting:**
    *   Use regularization techniques (L1/L2 regularization, dropout).
    *   Use data augmentation.
    *   Use early stopping.
    *   Reduce model complexity.
    *   Increase the amount of training data.

*   **Underfitting:**
    *   Increase model capacity.
    *   Train for longer.
    *   Use a more complex optimizer.
    *   Add more features.
    *   Reduce regularization.

*   **Debugging with `tf.print`:**

    ```python
    @tf.function
    def my_function(x):
        tf.print("x:", x)  # Print the value of x
        return x * 2
    ```

*   **Debugging with `tf.debugging.assert_*`:**

    ```python
    @tf.function
    def my_function(x):
        tf.debugging.assert_positive(x, message="x must be positive")
        return x * 2
    ```

*   **Using the TensorFlow Debugger (tfdbg):** (Less common with TF 2.x eager execution, but still useful for graph mode)

*   **Using Python's `pdb` debugger:**  You can use `pdb.set_trace()` inside your `@tf.function` decorated functions, but you'll need to run your code with eager execution disabled (`tf.config.run_functions_eagerly(False)`) or use `tf.py_function`.