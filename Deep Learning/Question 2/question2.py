import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, momentum=0.9):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.weight_velocities = []
        self.bias_velocities = []
        
        # Input layer to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * 0.01)
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        self.weight_velocities.append(np.zeros((input_size, hidden_sizes[0])))
        self.bias_velocities.append(np.zeros((1, hidden_sizes[0])))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, hidden_sizes[i+1])))
            self.weight_velocities.append(np.zeros((hidden_sizes[i], hidden_sizes[i+1])))
            self.bias_velocities.append(np.zeros((1, hidden_sizes[i+1])))
        
        # Last hidden layer to output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.01)
        self.biases.append(np.zeros((1, output_size)))
        self.weight_velocities.append(np.zeros((hidden_sizes[-1], output_size)))
        self.bias_velocities.append(np.zeros((1, output_size)))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.layer_inputs = []
        self.layer_outputs = [X]
        
        for i in range(len(self.weights)):
            layer_input = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(layer_input)
            layer_output = self.sigmoid(layer_input)
            self.layer_outputs.append(layer_output)
        
        return self.layer_outputs[-1]
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Calculate gradients
        weight_gradients = [0] * len(self.weights)
        bias_gradients = [0] * len(self.biases)
        
        # Output layer error
        error = output - y
        delta = error * self.sigmoid_derivative(output)
        
        # Last layer gradients
        weight_gradients[-1] = np.dot(self.layer_outputs[-2].T, delta) / m
        bias_gradients[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * self.sigmoid_derivative(self.layer_outputs[l+1])
            weight_gradients[l] = np.dot(self.layer_outputs[l].T, delta) / m
            bias_gradients[l] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Update weights and biases using momentum
        for i in range(len(self.weights)):
            self.weight_velocities[i] = self.momentum * self.weight_velocities[i] - self.learning_rate * weight_gradients[i]
            self.bias_velocities[i] = self.momentum * self.bias_velocities[i] - self.learning_rate * bias_gradients[i]
            
            self.weights[i] += self.weight_velocities[i]
            self.biases[i] += self.bias_velocities[i]
        
        return weight_gradients, bias_gradients
    
    def train(self, X, y, epochs=10, batch_size=10):
        m = X.shape[0]
        history = {'loss': [], 'accuracy': []}
        weight_updates = []
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_weight_updates = []
            
            # Mini-batch training
            for i in range(0, m, batch_size):
                end = min(i + batch_size, m)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Compute loss
                batch_loss = np.mean(np.square(output - y_batch))
                epoch_loss += batch_loss * (end - i) / m
                
                # Compute accuracy (assuming binary classification threshold = 0.5)
                predictions = (output >= 0.5).astype(int)
                batch_accuracy = np.mean(np.all(predictions == y_batch, axis=1))
                epoch_accuracy += batch_accuracy * (end - i) / m
                
                # Backward pass
                weight_gradients, _ = self.backward(X_batch, y_batch, output)
                epoch_weight_updates.append([np.abs(grad).mean() for grad in weight_gradients])
            
            # Average weight updates for the epoch
            avg_weight_updates = np.mean(epoch_weight_updates, axis=0)
            weight_updates.append(avg_weight_updates)
            
            # Store metrics
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)
            
            # Display progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            print(f"Weight Updates: {[f'{w:.6f}' for w in avg_weight_updates]}")
            print("-" * 50)
        
        return history, weight_updates
    
    def predict(self, X):
        output = self.forward(X)
        return (output >= 0.5).astype(int)

# Generate random data (100 samples, 3 features, 2 output classes)
np.random.seed(42)
X = np.random.rand(100, 3)
y = np.random.randint(0, 2, size=(100, 2))

# Create and train the MLP
mlp = MLP(
    input_size=3,
    hidden_sizes=[4, 4],
    output_size=2,
    learning_rate=0.01,  # Changed from 0.1 to 0.01
    momentum=0.9
)

# Train the model
history, weight_updates = mlp.train(X, y, epochs=10, batch_size=10)

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), history['loss'], marker='o')
plt.title('Loss vs Epochs (Learning Rate = 0.01)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

# Print final metrics
print("\nFinal Results:")
print(f"Final Loss: {history['loss'][-1]:.4f}")
print(f"Final Accuracy: {history['accuracy'][-1]:.4f}")
print("\nWeight updates per layer and epoch:")
for epoch, updates in enumerate(weight_updates):
    print(f"Epoch {epoch+1}: {[f'{w:.6f}' for w in updates]}")
    
    
# Epoch 1/10
# Loss: 0.2501, Accuracy: 0.3000
# Weight Updates: ['0.000000', '0.000033', '0.012584']
# --------------------------------------------------
# Epoch 2/10
# Loss: 0.2495, Accuracy: 0.3200
# Weight Updates: ['0.000000', '0.000041', '0.013641']
# --------------------------------------------------
# Epoch 3/10
# Loss: 0.2491, Accuracy: 0.3200
# Weight Updates: ['0.000000', '0.000047', '0.011450']
# --------------------------------------------------
# Epoch 4/10
# Loss: 0.2488, Accuracy: 0.3200
# Weight Updates: ['0.000000', '0.000094', '0.019962']
# --------------------------------------------------
# Epoch 5/10
# Loss: 0.2485, Accuracy: 0.3200
# Weight Updates: ['0.000000', '0.000090', '0.016009']       
# --------------------------------------------------
# Epoch 6/10
# Loss: 0.2484, Accuracy: 0.3200
# Weight Updates: ['0.000001', '0.000164', '0.017260']
# --------------------------------------------------
# Epoch 7/10
# Loss: 0.2480, Accuracy: 0.3200
# Weight Updates: ['0.000001', '0.000113', '0.012423']       
# --------------------------------------------------
# Epoch 8/10
# Loss: 0.2479, Accuracy: 0.3200
# Weight Updates: ['0.000001', '0.000213', '0.016104']       
# --------------------------------------------------
# Epoch 9/10
# Loss: 0.2478, Accuracy: 0.3200
# Weight Updates: ['0.000001', '0.000159', '0.012639']       
# --------------------------------------------------
# Epoch 10/10
# Loss: 0.2477, Accuracy: 0.3200
# Weight Updates: ['0.000001', '0.000181', '0.015682']       
# --------------------------------------------------
