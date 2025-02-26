# Deep-Learning-Assignment-1

Multi-Layer Perceptron (MLP) and Regression Models Assignment

Overview

This repository contains implementations for various machine learning tasks, including training Multi-Layer Perceptrons (MLP) and regression models with different techniques.

Assignment Tasks

1️⃣ Implement an MLP with the following specifications:

Inputs: 3

Hidden Layers: 2 (each with 4 neurons)

Output Layer: 2 neurons

Activation Function: Sigmoid in all layers

Optimizer: SGD with momentum = 0.9

Loss Function: Mean Squared Error (MSE)

Training Data: 100 random samples

Epochs: 10

Batch Size: 10

Output: Weight updates, loss, and accuracy after each epoch, plus a loss curve plot

2️⃣ Modify the above implementation:

Use a learning rate of 0.01 instead of 0.1.

Print the loss values, accuracy, and weight updates.

Plot the loss curve.

Explain observations regarding loss, accuracy, and weight updates.

3️⃣ Compare MSE and Binary Cross-Entropy (BCE) Loss:

Implement the same MLP but use BCE loss instead of MSE.

Display and compare loss and accuracy values for MSE and BCE.

Provide an explanation of the observed differences.

4️⃣ House Price Prediction using Regression:

Given a CSV file for house price prediction (Dataset).

Split the dataset: 80% training, 20% validation.

Train a Linear Regression model:

Compute MSE loss on validation data.

Print feature coefficient names and values.

Apply L1 (Lasso) Regularization:

Compute MSE loss and print feature coefficients.

Apply L2 (Ridge) Regularization:

Compute MSE loss and print feature coefficients.

Explain and justify observed feature coefficient values.

5️⃣ Implement an MLP with Different Initializations:

Architecture:

Inputs: 3

Hidden Layers: 2 (each with 8 neurons, ReLU activation)

Output Layer: Scalar (Sigmoid activation)

Optimizer: Adam

Loss Function: Binary Cross-Entropy

Use three parameter initialization methods:

Xavier

He

Normal

Display loss and accuracy values for each initialization method.
