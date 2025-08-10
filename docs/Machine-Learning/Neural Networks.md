---
title: Neural Networks
description: Comprehensive guide to Neural Networks with mathematical intuition, architectures, implementations, and interview questions.
comments: true
---

# 🧠 Neural Networks

Neural Networks are computing systems inspired by biological neural networks, consisting of interconnected nodes (neurons) that learn complex patterns through iterative weight adjustments using backpropagation.

**Resources:** [Deep Learning Book](https://www.deeplearningbook.org/) | [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) | [TensorFlow Tutorial](https://www.tensorflow.org/tutorials)

## 
 Summary

Neural Networks (also known as Artificial Neural Networks or ANNs) are computational models inspired by the human brain's structure and function. They consist of interconnected processing units called neurons or nodes, organized in layers that transform input data through weighted connections and activation functions.

**Key Components:**
- **Neurons/Nodes**: Basic processing units that receive inputs, apply weights, and produce outputs
- **Layers**: Collections of neurons (input layer, hidden layers, output layer)
- **Weights**: Parameters that determine the strength of connections between neurons
- **Biases**: Additional parameters that help shift the activation function
- **Activation Functions**: Non-linear functions that introduce complexity to the model

**Types of Neural Networks:**
- **Feedforward Networks**: Information flows in one direction from input to output
- **Convolutional Neural Networks (CNNs)**: Specialized for image processing
- **Recurrent Neural Networks (RNNs)**: Handle sequential data with memory
- **Long Short-Term Memory (LSTM)**: Advanced RNNs for long sequences
- **Autoencoders**: Learn compressed representations of data
- **Generative Adversarial Networks (GANs)**: Generate new data samples

**Applications:**
- Image recognition and computer vision
- Natural language processing
- Speech recognition
- Recommendation systems
- Time series prediction
- Game playing (AlphaGo, chess)
- Medical diagnosis
- Autonomous vehicles

**Advantages:**
- Can learn complex non-linear relationships
- Universal function approximators
- Automatic feature learning
- Scalable to large datasets
- Versatile across domains

## 🧠 Intuition

### Biological Inspiration

Neural networks are inspired by how biological neurons work:
- **Biological neuron**: Receives signals through dendrites, processes them in the cell body, and sends output through axons
- **Artificial neuron**: Receives inputs, applies weights and bias, passes through activation function, and produces output

### Mathematical Foundation

#### Single Neuron (Perceptron)

A single neuron computes:
$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(w^T x + b)$$

Where:
- $x_i$ are input features
- $w_i$ are weights
- $b$ is bias
- $f$ is the activation function
- $y$ is the output

#### Multi-layer Neural Network

For a network with $L$ layers:

**Forward Propagation:**
$$a^{(l)} = f^{(l)}\left(W^{(l)} a^{(l-1)} + b^{(l)}\right)$$

Where:
- $a^{(l)}$ is the activation of layer $l$
- $W^{(l)}$ is the weight matrix for layer $l$
- $b^{(l)}$ is the bias vector for layer $l$
- $f^{(l)}$ is the activation function for layer $l$

#### Activation Functions

**Sigmoid:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Hyperbolic Tangent (tanh):**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**ReLU (Rectified Linear Unit):**
$$\text{ReLU}(x) = \max(0, x)$$

**Leaky ReLU:**
$$\text{LeakyReLU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}$$

**Softmax (for multi-class output):**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

#### Loss Functions

**Mean Squared Error (Regression):**
$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Cross-entropy (Classification):**
$$L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

#### Backpropagation Algorithm

**Chain Rule Application:**
$$\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial a_j^{(l)}} \cdot \frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ij}^{(l)}}$$

**Weight Update Rule:**
$$w_{ij}^{(l)} = w_{ij}^{(l)} - \alpha \frac{\partial L}{\partial w_{ij}^{(l)}}$$

Where $\alpha$ is the learning rate.

### Universal Approximation Theorem

Neural networks with at least one hidden layer containing sufficient neurons can approximate any continuous function to arbitrary accuracy, making them powerful universal function approximators.

## =" Implementation using Libraries

### Using TensorFlow/Keras

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                         n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a simple feedforward neural network
def create_model(input_dim, hidden_layers=[64, 32], output_dim=1, activation='relu'):
    """
    Create a feedforward neural network
    
    Args:
        input_dim: Number of input features
        hidden_layers: List of neurons in each hidden layer
        output_dim: Number of output neurons
        activation: Activation function for hidden layers
    """
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Dense(hidden_layers[0], 
                               activation=activation, 
                               input_dim=input_dim))
    model.add(keras.layers.Dropout(0.3))
    
    # Hidden layers
    for neurons in hidden_layers[1:]:
        model.add(keras.layers.Dense(neurons, activation=activation))
        model.add(keras.layers.Dropout(0.3))
    
    # Output layer
    if output_dim == 1:
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        model.add(keras.layers.Dense(output_dim, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    # Compile model
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    return model

# Create and train model
model = create_model(input_dim=X_train_scaled.shape[1])
print("Model Architecture:")
model.summary()

# Train the model
history = model.fit(X_train_scaled, y_train,
                   batch_size=32,
                   epochs=50,
                   validation_split=0.2,
                   verbose=0)

# Evaluate the model
train_loss, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

### Multi-class Classification with Iris Dataset

```python
# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create multi-class model
multiclass_model = create_model(input_dim=4, 
                              hidden_layers=[10, 8], 
                              output_dim=3)

# Train model
history = multiclass_model.fit(X_train_scaled, y_train,
                              epochs=100,
                              batch_size=16,
                              validation_split=0.2,
                              verbose=0)

# Predictions
predictions = multiclass_model.predict(X_test_scaled)
predicted_classes = np.argmax(predictions, axis=1)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix
print("\nMulti-class Classification Results:")
print("Classification Report:")
print(classification_report(y_test, predicted_classes, 
                          target_names=iris.target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted_classes))
```

### Using PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    """
    Simple feedforward neural network in PyTorch
    """
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.3):
        super(SimpleNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        if output_size == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
pytorch_model = SimpleNN(input_size=4, hidden_sizes=[10, 8], output_size=3)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = pytorch_model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

# Evaluate PyTorch model
with torch.no_grad():
    test_outputs = pytorch_model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'PyTorch Model Test Accuracy: {accuracy:.4f}')
```

##  From Scratch Implementation

### Complete Neural Network from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    """
    Neural Network implementation from scratch using NumPy
    """
    
    def __init__(self, layers, learning_rate=0.01, random_seed=42):
        """
        Initialize neural network
        
        Args:
            layers: List of integers representing number of neurons in each layer
            learning_rate: Learning rate for gradient descent
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        self.layers = layers
        self.learning_rate = learning_rate
        self.num_layers = len(layers)
        
        # Initialize weights and biases using He initialization
        self.weights = {}
        self.biases = {}
        
        for i in range(1, self.num_layers):
            # He initialization for ReLU activation
            self.weights[f'W{i}'] = np.random.randn(layers[i-1], layers[i]) * np.sqrt(2/layers[i-1])
            self.biases[f'b{i}'] = np.zeros((1, layers[i]))
        
        # Store activations and gradients
        self.activations = {}
        self.gradients = {}
        
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def softmax(self, z):
        """Softmax activation function"""
        # Numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Forward propagation through the network
        
        Args:
            X: Input data of shape (m, n_features)
            
        Returns:
            Final output of the network
        """
        self.activations['A0'] = X
        
        for i in range(1, self.num_layers):
            # Linear transformation
            Z = np.dot(self.activations[f'A{i-1}'], self.weights[f'W{i}']) + self.biases[f'b{i}']
            self.activations[f'Z{i}'] = Z
            
            # Apply activation function
            if i == self.num_layers - 1:  # Output layer
                if self.layers[-1] == 1:  # Binary classification
                    A = self.sigmoid(Z)
                else:  # Multi-class classification
                    A = self.softmax(Z)
            else:  # Hidden layers
                A = self.relu(Z)
            
            self.activations[f'A{i}'] = A
        
        return self.activations[f'A{self.num_layers-1}']
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute loss function
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Loss value
        """
        m = y_true.shape[0]
        
        if self.layers[-1] == 1:  # Binary classification
            # Binary cross-entropy
            epsilon = 1e-15  # Small value to prevent log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:  # Multi-class classification
            # Categorical cross-entropy
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return loss
    
    def backward_propagation(self, X, y):
        """
        Backward propagation to compute gradients
        
        Args:
            X: Input data
            y: True labels
        """
        m = X.shape[0]
        
        # Output layer gradient
        if self.layers[-1] == 1:  # Binary classification
            dZ = self.activations[f'A{self.num_layers-1}'] - y.reshape(-1, 1)
        else:  # Multi-class classification
            dZ = self.activations[f'A{self.num_layers-1}'] - y
        
        # Backpropagate through layers
        for i in range(self.num_layers - 1, 0, -1):
            # Compute gradients
            self.gradients[f'dW{i}'] = (1/m) * np.dot(self.activations[f'A{i-1}'].T, dZ)
            self.gradients[f'db{i}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            if i > 1:  # Not the first layer
                # Compute dA for previous layer
                dA_prev = np.dot(dZ, self.weights[f'W{i}'].T)
                # Compute dZ for previous layer (ReLU derivative)
                dZ = dA_prev * self.relu_derivative(self.activations[f'Z{i-1}'])
    
    def update_parameters(self):
        """Update weights and biases using gradients"""
        for i in range(1, self.num_layers):
            self.weights[f'W{i}'] -= self.learning_rate * self.gradients[f'dW{i}']
            self.biases[f'b{i}'] -= self.learning_rate * self.gradients[f'db{i}']
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network
        
        Args:
            X: Training data
            y: Training labels
            epochs: Number of training epochs
            verbose: Whether to print training progress
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward_propagation(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            
            # Backward propagation
            self.backward_propagation(X, y)
            
            # Update parameters
            self.update_parameters()
            
            # Print progress
            if verbose and epoch % 100 == 0:
                accuracy = self.accuracy(y, y_pred)
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
        
        return losses
    
    def predict(self, X):
        """Make predictions on new data"""
        y_pred = self.forward_propagation(X)
        
        if self.layers[-1] == 1:  # Binary classification
            return (y_pred > 0.5).astype(int)
        else:  # Multi-class classification
            return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.forward_propagation(X)
    
    def accuracy(self, y_true, y_pred):
        """Compute accuracy"""
        if self.layers[-1] == 1:  # Binary classification
            predictions = (y_pred > 0.5).astype(int)
            return np.mean(predictions == y_true.reshape(-1, 1))
        else:  # Multi-class classification
            predictions = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
            return np.mean(predictions == y_true_labels)

# Demonstration with Moon dataset
def demo_neural_network():
    """Demonstrate neural network on moon dataset"""
    # Generate moon dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train neural network
    nn = NeuralNetwork(layers=[2, 10, 8, 1], learning_rate=0.1)
    
    print("Training Neural Network...")
    losses = nn.fit(X_train, y_train, epochs=1000, verbose=True)
    
    # Make predictions
    train_pred = nn.predict(X_train)
    test_pred = nn.predict(X_test)
    
    train_accuracy = np.mean(train_pred == y_train.reshape(-1, 1))
    test_accuracy = np.mean(test_pred == y_test.reshape(-1, 1))
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot loss curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot original data
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('Original Data')
    plt.colorbar(scatter)
    
    # Plot decision boundary
    plt.subplot(1, 3, 3)
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.title('Decision Boundary')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    return nn

# Run demonstration
neural_network = demo_neural_network()
```

### Advanced Features Implementation

```python
class AdvancedNeuralNetwork(NeuralNetwork):
    """
    Extended neural network with advanced features
    """
    
    def __init__(self, layers, learning_rate=0.01, momentum=0.9, 
                 regularization=0.01, dropout_rate=0.5, random_seed=42):
        super().__init__(layers, learning_rate, random_seed)
        
        self.momentum = momentum
        self.regularization = regularization
        self.dropout_rate = dropout_rate
        
        # Initialize momentum terms
        self.velocity_w = {}
        self.velocity_b = {}
        
        for i in range(1, self.num_layers):
            self.velocity_w[f'W{i}'] = np.zeros_like(self.weights[f'W{i}'])
            self.velocity_b[f'b{i}'] = np.zeros_like(self.biases[f'b{i}'])
    
    def dropout(self, A, training=True):
        """Apply dropout regularization"""
        if training and self.dropout_rate > 0:
            mask = np.random.rand(*A.shape) > self.dropout_rate
            return A * mask / (1 - self.dropout_rate)
        return A
    
    def forward_propagation(self, X, training=True):
        """Forward propagation with dropout"""
        self.activations['A0'] = X
        
        for i in range(1, self.num_layers):
            Z = np.dot(self.activations[f'A{i-1}'], self.weights[f'W{i}']) + self.biases[f'b{i}']
            self.activations[f'Z{i}'] = Z
            
            if i == self.num_layers - 1:  # Output layer
                if self.layers[-1] == 1:
                    A = self.sigmoid(Z)
                else:
                    A = self.softmax(Z)
            else:  # Hidden layers
                A = self.relu(Z)
                A = self.dropout(A, training)  # Apply dropout
            
            self.activations[f'A{i}'] = A
        
        return self.activations[f'A{self.num_layers-1}']
    
    def compute_loss_with_regularization(self, y_true, y_pred):
        """Compute loss with L2 regularization"""
        base_loss = self.compute_loss(y_true, y_pred)
        
        # Add L2 regularization
        l2_penalty = 0
        for i in range(1, self.num_layers):
            l2_penalty += np.sum(self.weights[f'W{i}'] ** 2)
        
        regularized_loss = base_loss + (self.regularization / 2) * l2_penalty
        return regularized_loss
    
    def update_parameters_with_momentum(self):
        """Update parameters using momentum"""
        for i in range(1, self.num_layers):
            # Add L2 regularization to gradients
            reg_dW = self.gradients[f'dW{i}'] + self.regularization * self.weights[f'W{i}']
            
            # Update velocity
            self.velocity_w[f'W{i}'] = (self.momentum * self.velocity_w[f'W{i}'] - 
                                      self.learning_rate * reg_dW)
            self.velocity_b[f'b{i}'] = (self.momentum * self.velocity_b[f'b{i}'] - 
                                      self.learning_rate * self.gradients[f'db{i}'])
            
            # Update parameters
            self.weights[f'W{i}'] += self.velocity_w[f'W{i}']
            self.biases[f'b{i}'] += self.velocity_b[f'b{i}']
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """Train with advanced features"""
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation (with dropout)
            y_pred = self.forward_propagation(X, training=True)
            
            # Compute loss with regularization
            loss = self.compute_loss_with_regularization(y, y_pred)
            losses.append(loss)
            
            # Backward propagation
            self.backward_propagation(X, y)
            
            # Update parameters with momentum
            self.update_parameters_with_momentum()
            
            # Print progress
            if verbose and epoch % 100 == 0:
                # Use forward propagation without dropout for accuracy calculation
                y_pred_eval = self.forward_propagation(X, training=False)
                accuracy = self.accuracy(y, y_pred_eval)
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
        
        return losses
    
    def predict(self, X):
        """Make predictions without dropout"""
        y_pred = self.forward_propagation(X, training=False)
        
        if self.layers[-1] == 1:
            return (y_pred > 0.5).astype(int)
        else:
            return np.argmax(y_pred, axis=1)
```

##   Assumptions and Limitations

### Assumptions

**Data Assumptions:**
- **Independent and identically distributed (IID) data**: Training and test data come from the same distribution
- **Sufficient training data**: Need enough data to learn complex patterns without overfitting
- **Feature relevance**: Input features contain useful information for the target variable
- **Stationarity**: Data distribution doesn't change significantly over time

**Model Assumptions:**
- **Universal approximation**: Any continuous function can be approximated with sufficient neurons
- **Differentiability**: Loss function and activations should be differentiable for backpropagation
- **Local minima acceptability**: Finding global minimum is not required for good performance
- **Feature scaling**: Input features should be normalized for optimal performance

### Limitations

**Computational Limitations:**
- **High computational cost**: Training can be expensive, especially for large networks
- **Memory requirements**: Need to store activations, gradients, and parameters
- **Training time**: Can take hours or days for complex problems
- **Hardware dependency**: Performance varies significantly across different hardware

**Theoretical Limitations:**
- **Black box nature**: Difficult to interpret decisions and understand learned features
- **Overfitting tendency**: Can memorize training data instead of learning generalizable patterns
- **Hyperparameter sensitivity**: Performance highly dependent on architecture and parameter choices
- **Local minima**: Gradient descent may get stuck in suboptimal solutions

**Practical Limitations:**
- **Data hunger**: Require large amounts of labeled data
- **Vanishing/exploding gradients**: Deep networks suffer from gradient flow problems
- **Catastrophic forgetting**: Forget previously learned tasks when learning new ones
- **Adversarial vulnerability**: Small input perturbations can cause misclassification

### Common Problems and Solutions

| Problem | Cause | Solutions |
|---------|-------|-----------|
| **Overfitting** | Too complex model, insufficient data | Dropout, regularization, early stopping, data augmentation |
| **Underfitting** | Too simple model, insufficient training | More layers/neurons, longer training, reduce regularization |
| **Vanishing Gradients** | Deep networks, saturating activations | ReLU, ResNet, LSTM, batch normalization |
| **Exploding Gradients** | Poor weight initialization, high learning rate | Gradient clipping, proper initialization, lower learning rate |
| **Slow Convergence** | Poor optimization settings | Adam optimizer, learning rate scheduling, batch normalization |

### When to Use Neural Networks

**Best suited for:**
- Large datasets with complex patterns
- Image, text, and speech recognition
- Non-linear relationships
- Automatic feature learning
- High-dimensional data

**Not ideal for:**
- Small datasets (< 1000 samples)
- Linear relationships
- Interpretability is crucial
- Limited computational resources
- Simple problems with clear patterns

## ❓ Interview Questions

??? question "**Q1: Explain the backpropagation algorithm and its mathematical foundation.**"

    **Answer:**
    
    Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to network parameters.
    
    **Mathematical Foundation:**
    Uses the chain rule of calculus to compute partial derivatives:
    
    $$\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial a_j^{(l)}} \cdot \frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ij}^{(l)}}$$
    
    **Steps:**
    1. **Forward pass**: Compute activations for all layers
    2. **Loss computation**: Calculate loss at output layer
    3. **Backward pass**: Compute gradients layer by layer from output to input
    4. **Parameter update**: Update weights and biases using computed gradients
    
    **Key insight**: Error signals propagate backward through the network, with each layer's gradients depending on the subsequent layer's gradients.

??? question "**Q2: What is the vanishing gradient problem and how can it be addressed?**"

    **Answer:**
    
    **Vanishing Gradient Problem:**
    In deep networks, gradients become exponentially smaller as they propagate backward through layers, making early layers learn very slowly or not at all.
    
    **Causes:**
    - Sigmoid/tanh activation functions (derivatives d 0.25)
    - Weight initialization issues
    - Deep network architectures
    
    **Solutions:**
    
    1. **ReLU Activation**: `ReLU(x) = max(0, x)` has gradient 1 for positive inputs
    2. **Proper Weight Initialization**: He/Xavier initialization
    3. **Batch Normalization**: Normalizes inputs to each layer
    4. **Residual Connections**: Skip connections in ResNets
    5. **LSTM/GRU**: For sequential data
    6. **Gradient Clipping**: Prevent exploding gradients
    
    ```python
    # Example: ReLU vs Sigmoid gradient
    def sigmoid_derivative(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)  # Max value: 0.25
    
    def relu_derivative(x):
        return (x > 0).astype(float)  # Value: 0 or 1
    ```

??? question "**Q3: Compare different activation functions and their use cases.**"

    **Answer:**
    
    | Activation | Formula | Range | Derivative | Use Case | Pros | Cons |
    |------------|---------|-------|------------|----------|------|------|
    | **Sigmoid** | $\frac{1}{1+e^{-x}}$ | (0,1) | $\sigma(x)(1-\sigma(x))$ | Binary classification output | Smooth, interpretable probabilities | Vanishing gradients, not zero-centered |
    | **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | (-1,1) | $1-\tanh^2(x)$ | Hidden layers (legacy) | Zero-centered, smooth | Vanishing gradients |
    | **ReLU** | $\max(0,x)$ | [0,) | $\begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | Hidden layers | Simple, no vanishing gradients | Dead neurons, not zero-centered |
    | **Leaky ReLU** | $\begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$ | (-,) | $\begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$ | Hidden layers | Fixes dead ReLU problem | Hyperparameter ± |
    | **Softmax** | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | (0,1), $\sum=1$ | Complex | Multi-class output | Probability distribution | Only for output layer |
    
    **Recommendations:**
    - **Hidden layers**: ReLU or Leaky ReLU
    - **Binary output**: Sigmoid
    - **Multi-class output**: Softmax
    - **Regression output**: Linear (no activation)

??? question "**Q4: How do you prevent overfitting in neural networks?**"

    **Answer:**
    
    **Regularization Techniques:**
    
    1. **Dropout**: Randomly set neurons to zero during training
       ```python
       def dropout(x, keep_prob=0.5, training=True):
           if training:
               mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
               return x * mask
           return x
       ```
    
    2. **L1/L2 Regularization**: Add penalty to loss function
       $$L_{total} = L_{original} + \lambda \sum_{i} |w_i|$$ (L1)
       $$L_{total} = L_{original} + \lambda \sum_{i} w_i^2$$ (L2)
    
    3. **Early Stopping**: Stop training when validation loss stops improving
    
    4. **Data Augmentation**: Artificially increase training data
    
    5. **Batch Normalization**: Normalize inputs to each layer
    
    6. **Reduce Model Complexity**: Fewer layers/neurons
    
    7. **Cross-validation**: Use k-fold validation for model selection
    
    **Implementation:**
    ```python
    model.add(keras.layers.Dropout(0.5))
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy',
                  regularizers=keras.regularizers.l2(0.01))
    ```

??? question "**Q5: Explain the differences between batch, mini-batch, and stochastic gradient descent.**"

    **Answer:**
    
    **Gradient Descent Variants:**
    
    1. **Batch Gradient Descent:**
       - Uses entire dataset for each update
       - Formula: $w = w - \alpha \nabla_w J(w)$
       - **Pros**: Stable convergence, guaranteed global minimum for convex functions
       - **Cons**: Slow for large datasets, memory intensive
    
    2. **Stochastic Gradient Descent (SGD):**
       - Uses one sample at a time
       - Formula: $w = w - \alpha \nabla_w J(w; x^{(i)}, y^{(i)})$
       - **Pros**: Fast updates, can escape local minima
       - **Cons**: Noisy updates, may oscillate around minimum
    
    3. **Mini-batch Gradient Descent:**
       - Uses small batches (typically 32-256 samples)
       - Combines benefits of both approaches
       - **Pros**: Balanced speed and stability, vectorization benefits
       - **Cons**: Additional hyperparameter (batch size)
    
    **Comparison:**
    ```python
    # Batch size effects
    batch_sizes = [1, 32, 128, len(X_train)]  # SGD, mini-batch, mini-batch, batch
    names = ['SGD', 'Mini-batch (32)', 'Mini-batch (128)', 'Batch GD']
    ```
    
    **Modern Practice**: Mini-batch GD with adaptive optimizers (Adam, RMSprop) is most common.

??? question "**Q6: What is the Universal Approximation Theorem and what does it mean for neural networks?**"

    **Answer:**
    
    **Universal Approximation Theorem:**
    A feedforward neural network with:
    - At least one hidden layer
    - Sufficient number of neurons
    - Non-linear activation functions
    
    Can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary accuracy.
    
    **Mathematical Statement:**
    For any continuous function $f: [0,1]^n \to \mathbb{R}$ and $\epsilon > 0$, there exists a neural network $F$ such that:
    $$|F(x) - f(x)| < \epsilon \text{ for all } x \in [0,1]^n$$
    
    **Implications:**
    - **Theoretical**: Neural networks are universal function approximators
    - **Practical**: Width vs depth trade-offs exist
    - **Limitation**: Says nothing about learnability or generalization
    - **Reality**: Need appropriate architecture, optimization, and data
    
    **Important Notes:**
    - Theorem guarantees approximation exists, not that SGD will find it
    - Doesn't specify required network size
    - Doesn't guarantee good generalization

??? question "**Q7: How do you initialize weights in neural networks and why is it important?**"

    **Answer:**
    
    **Why Initialization Matters:**
    - Breaks symmetry between neurons
    - Prevents vanishing/exploding gradients
    - Affects convergence speed and final performance
    
    **Common Initialization Methods:**
    
    1. **Zero Initialization**: 
       - All weights = 0
       - **Problem**: All neurons learn the same features (symmetry)
    
    2. **Random Initialization**:
       ```python
       W = np.random.randn(n_in, n_out) * 0.01
       ```
       - **Problem**: May cause vanishing gradients
    
    3. **Xavier/Glorot Initialization**:
       ```python
       W = np.random.randn(n_in, n_out) * np.sqrt(1/n_in)
       # or
       W = np.random.randn(n_in, n_out) * np.sqrt(2/(n_in + n_out))
       ```
       - **Best for**: Sigmoid, tanh activations
    
    4. **He Initialization**:
       ```python
       W = np.random.randn(n_in, n_out) * np.sqrt(2/n_in)
       ```
       - **Best for**: ReLU activations
    
    **Rule of thumb**: Use He initialization with ReLU, Xavier with sigmoid/tanh.

??? question "**Q8: Explain the concept of batch normalization and its benefits.**"

    **Answer:**
    
    **Batch Normalization:**
    Normalizes inputs to each layer by adjusting and scaling activations.
    
    **Mathematical Formula:**
    For a layer with inputs $x_1, x_2, ..., x_m$ (mini-batch):
    
    $$\mu = \frac{1}{m}\sum_{i=1}^{m} x_i$$
    $$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu)^2$$
    $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
    $$y_i = \gamma \hat{x}_i + \beta$$
    
    Where $\gamma$ and $\beta$ are learnable parameters.
    
    **Benefits:**
    1. **Faster training**: Higher learning rates possible
    2. **Reduced sensitivity**: Less dependent on initialization
    3. **Regularization effect**: Slight noise helps prevent overfitting
    4. **Gradient flow**: Helps with vanishing gradient problem
    5. **Internal covariate shift**: Reduces change in input distributions
    
    **Implementation:**
    ```python
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    ```

??? question "**Q9: What are the differences between feed-forward, convolutional, and recurrent neural networks?**"

    **Answer:**
    
    | Aspect | Feedforward | Convolutional (CNN) | Recurrent (RNN) |
    |--------|-------------|---------------------|-----------------|
    | **Architecture** | Layers connected sequentially | Convolution + pooling layers | Feedback connections |
    | **Information Flow** | Input  Hidden  Output | Local receptive fields | Sequential processing |
    | **Parameter Sharing** | No | Yes (shared kernels) | Yes (across time) |
    | **Best For** | Tabular data, classification | Images, spatial data | Sequences, time series |
    | **Key Advantage** | Simplicity, universal approximation | Translation invariance | Memory of past inputs |
    | **Main Challenge** | Limited to fixed input sizes | Large parameter count | Vanishing gradients |
    
    **Feedforward:**
    ```python
    # Simple MLP
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    ```
    
    **CNN:**
    ```python
    # For image classification
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    ```
    
    **RNN:**
    ```python
    # For sequence data
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(timesteps, features)),
        LSTM(50),
        Dense(1)
    ])
    ```

??? question "**Q10: How do you handle class imbalance in neural network classification?**"

    **Answer:**
    
    **Class Imbalance Strategies:**
    
    1. **Class Weights**: Penalize minority class errors more heavily
       ```python
       from sklearn.utils.class_weight import compute_class_weight
       
       class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(y_train), 
                                         y=y_train)
       class_weight_dict = dict(enumerate(class_weights))
       
       model.fit(X_train, y_train, class_weight=class_weight_dict)
       ```
    
    2. **Resampling Techniques**:
       - **Oversampling**: SMOTE, ADASYN
       - **Undersampling**: Random undersampling
       - **Combined**: SMOTETomek
    
    3. **Custom Loss Functions**:
       ```python
       def weighted_binary_crossentropy(pos_weight):
           def loss(y_true, y_pred):
               return K.mean(-pos_weight * y_true * K.log(y_pred) - 
                           (1 - y_true) * K.log(1 - y_pred))
           return loss
       ```
    
    4. **Focal Loss**: Focuses on hard examples
       ```python
       def focal_loss(alpha=0.25, gamma=2.0):
           def loss(y_true, y_pred):
               pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
               return -alpha * (1 - pt) ** gamma * tf.log(pt)
           return loss
       ```
    
    5. **Evaluation Metrics**: Use precision, recall, F1-score, AUC-ROC instead of accuracy
    
    6. **Threshold Tuning**: Adjust classification threshold based on validation set

## 💡 Examples

### Real-world Example: Image Classification with CIFAR-10

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of classes: {len(class_names)}")

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels to categorical
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Create CNN model
def create_cnn_model():
    model = keras.Sequential([
        # First Convolutional Block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Second Convolutional Block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Third Convolutional Block
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        
        # Dense Layers
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

# Create and compile model
model = create_cnn_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("CNN Model Architecture:")
model.summary()

# Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(X_train)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)

# Train model
print("Training CNN model...")
history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=32),
                    epochs=50,
                    validation_data=(X_test, y_test_cat),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_cat, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, 
                          target_names=class_names))

# Visualizations
plt.figure(figsize=(18, 6))

# Training history
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Confusion matrix
plt.subplot(1, 3, 3)
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

# Sample predictions visualization
def plot_predictions(images, true_labels, predicted_labels, class_names, num_samples=12):
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        
        true_class = class_names[true_labels[i]]
        pred_class = class_names[predicted_labels[i]]
        confidence = np.max(y_pred[i]) * 100
        
        color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
        plt.title(f'True: {true_class}\nPred: {pred_class} ({confidence:.1f}%)', 
                 color=color, fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Show sample predictions
plot_predictions(X_test[:12], y_true_classes[:12], y_pred_classes[:12], class_names)
```

### Time Series Prediction with RNN/LSTM

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Generate synthetic time series data
def generate_time_series(n_samples=1000):
    """Generate synthetic time series with trend, seasonality, and noise"""
    time = np.arange(n_samples)
    
    # Trend component
    trend = 0.02 * time
    
    # Seasonal components
    yearly = 10 * np.sin(2 * np.pi * time / 365.25)
    monthly = 5 * np.sin(2 * np.pi * time / 30.4)
    weekly = 3 * np.sin(2 * np.pi * time / 7)
    
    # Noise
    noise = np.random.normal(0, 2, n_samples)
    
    # Combine components
    series = 100 + trend + yearly + monthly + weekly + noise
    
    return pd.Series(series, index=pd.date_range('2020-01-01', periods=n_samples, freq='D'))

# Generate data
ts_data = generate_time_series(1000)

print(f"Time series length: {len(ts_data)}")
print(f"Date range: {ts_data.index[0]} to {ts_data.index[-1]}")

# Prepare data for LSTM
def prepare_lstm_data(data, lookback_window=60, forecast_horizon=1):
    """
    Prepare time series data for LSTM training
    
    Args:
        data: Time series data
        lookback_window: Number of previous time steps to use as input
        forecast_horizon: Number of time steps to predict
    
    Returns:
        X, y arrays for training
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    
    for i in range(lookback_window, len(scaled_data) - forecast_horizon + 1):
        X.append(scaled_data[i-lookback_window:i, 0])
        y.append(scaled_data[i:i+forecast_horizon, 0])
    
    return np.array(X), np.array(y), scaler

# Prepare data
lookback = 60
forecast_horizon = 10

X, y, scaler = prepare_lstm_data(ts_data, lookback, forecast_horizon)

# Reshape for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create LSTM model
def create_lstm_model(input_shape, forecast_horizon):
    """Create LSTM model for time series prediction"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(forecast_horizon)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build and train model
lstm_model = create_lstm_model((lookback, 1), forecast_horizon)

print("LSTM Model Architecture:")
lstm_model.summary()

# Train model
history = lstm_model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=50,
                        validation_data=(X_test, y_test),
                        verbose=1)

# Make predictions
train_predictions = lstm_model.predict(X_train)
test_predictions = lstm_model.predict(X_test)

# Inverse transform predictions
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_orig = scaler.inverse_transform(y_train)
y_test_orig = scaler.inverse_transform(y_test)

# Calculate metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_mae = mean_absolute_error(y_train_orig.flatten(), train_predictions.flatten())
test_mae = mean_absolute_error(y_test_orig.flatten(), test_predictions.flatten())
train_rmse = np.sqrt(mean_squared_error(y_train_orig.flatten(), train_predictions.flatten()))
test_rmse = np.sqrt(mean_squared_error(y_test_orig.flatten(), test_predictions.flatten()))

print(f"\nModel Performance:")
print(f"Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")

# Visualizations
plt.figure(figsize=(18, 12))

# Original time series
plt.subplot(3, 2, 1)
plt.plot(ts_data.index, ts_data.values)
plt.title('Original Time Series')
plt.xlabel('Date')
plt.ylabel('Value')

# Training history
plt.subplot(3, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Training predictions vs actual
plt.subplot(3, 2, 3)
plt.plot(y_train_orig[:, 0], label='Actual', alpha=0.7)
plt.plot(train_predictions[:, 0], label='Predicted', alpha=0.7)
plt.title('Training: Actual vs Predicted (First Step)')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()

# Test predictions vs actual
plt.subplot(3, 2, 4)
plt.plot(y_test_orig[:, 0], label='Actual', alpha=0.7)
plt.plot(test_predictions[:, 0], label='Predicted', alpha=0.7)
plt.title('Test: Actual vs Predicted (First Step)')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()

# Residuals plot
plt.subplot(3, 2, 5)
test_residuals = y_test_orig[:, 0] - test_predictions[:, 0]
plt.scatter(test_predictions[:, 0], test_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Plot (Test Set)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# Multi-step ahead predictions
plt.subplot(3, 2, 6)
sample_idx = 50
actual_sequence = y_test_orig[sample_idx]
predicted_sequence = test_predictions[sample_idx]

plt.plot(range(len(actual_sequence)), actual_sequence, 'o-', label='Actual')
plt.plot(range(len(predicted_sequence)), predicted_sequence, 's-', label='Predicted')
plt.title(f'Multi-step Prediction (Sample {sample_idx})')
plt.xlabel('Future Time Step')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# Feature importance analysis for time series
def analyze_lstm_importance(model, X_sample, scaler, n_steps=10):
    """Analyze which time steps are most important for prediction"""
    baseline_pred = model.predict(X_sample.reshape(1, -1, 1))
    importances = []
    
    for i in range(len(X_sample)):
        # Perturb each time step
        X_perturbed = X_sample.copy()
        X_perturbed[i] = np.mean(X_sample)  # Replace with mean
        
        perturbed_pred = model.predict(X_perturbed.reshape(1, -1, 1))
        importance = np.abs(baseline_pred - perturbed_pred).mean()
        importances.append(importance)
    
    return np.array(importances)

# Analyze importance for a sample
sample_importance = analyze_lstm_importance(lstm_model, X_test[0], scaler)

plt.figure(figsize=(12, 4))
plt.plot(range(len(sample_importance)), sample_importance)
plt.title('Time Step Importance for Prediction')
plt.xlabel('Time Step (from past)')
plt.ylabel('Importance Score')
plt.show()

print(f"Most important time steps: {np.argsort(sample_importance)[-5:]}")
```

## 📚 References

**Foundational Books:**
- [Deep Learning](https://www.deeplearningbook.org/) - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) - Christopher Bishop
- [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) - Hastie, Tibshirani, Friedman

**Classic Papers:**
- [Backpropagation](https://www.nature.com/articles/323533a0) - Rumelhart, Hinton, Williams (1986)
- [Universal Approximation Theorem](https://link.springer.com/article/10.1007/BF02551274) - Hornik, Stinchcombe, White (1989)
- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber (1997)
- [Dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) - Srivastava et al. (2014)
- [Batch Normalization](https://arxiv.org/abs/1502.03167) - Ioffe & Szegedy (2015)

**Modern Architectures:**
- [ResNet](https://arxiv.org/abs/1512.03385) - He et al. (2016)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)
- [BERT](https://arxiv.org/abs/1810.04805) - Devlin et al. (2018)
- [GPT](https://openai.com/blog/language-unsupervised/) - Radford et al. (2018)

**Online Resources:**
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Keras Documentation](https://keras.io/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [CS224n: Natural Language Processing](http://cs224n.stanford.edu/)

**Practical Guides:**
- [Neural Networks and Deep Learning Course](https://www.coursera.org/learn/neural-networks-deep-learning) - Andrew Ng
- [FastAI Practical Deep Learning](https://course.fast.ai/)
- [MIT 6.034 Artificial Intelligence](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)

**Specialized Topics:**
- [Convolutional Neural Networks for Visual Recognition](https://arxiv.org/abs/1511.08458)
- [Recurrent Neural Networks for Sequence Learning](https://arxiv.org/abs/1506.00019)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Neural Architecture Search](https://arxiv.org/abs/1707.07012)