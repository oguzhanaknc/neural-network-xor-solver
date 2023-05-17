# Neural Network XOR Solver

This repository contains a simple implementation of a neural network that solves the XOR logic gate. The code is written in Rust and demonstrates how to train a neural network using gradient descent.

## Getting Started

To run the code, make sure you have Rust installed on your system. You can clone this repository and compile the code using the following commands:


$ git clone https://github.com/oguzhanaknc/neural-network-xor-solver.git
$ cd neural-network-xor-solver
$ cargo build


## Usage

The code provides a command-line interface with an optional `--verbose` flag for enabling verbose mode. By default, verbose mode is disabled. You can run the program using the following command:



$ cargo run [--verbose]


The program will train a neural network to solve the XOR logic gate and print the results for various input combinations.

## Code Explanation

### NeuralNetwork Struct

The `NeuralNetwork` struct represents a simple neural network with three parameters: `w1`, `w2`, and `b`. These parameters correspond to the weights and bias of the neural network.

### Xor Struct

The `Xor` struct represents the XOR logic gate and contains three instances of the `NeuralNetwork` struct: `or`, `and`, and `nand`. These instances correspond to the neural networks responsible for performing the OR, AND, and NAND operations, respectively.

### Main Function

The `main` function is the entry point of the program. It sets up the program's verbosity based on the command-line arguments and then trains the neural network for a specified number of iterations. After training, it prints the results for various input combinations.

### calculate_cost Function

The `calculate_cost` function calculates the cost of the neural network model based on the training data. It iterates over the XOR gate training data, performs a forward pass through the neural network, and compares the predicted output with the expected output.

### sigmoid Function

The `sigmoid` function implements the sigmoid activation function, which is used to introduce non-linearity in the neural network.

### forward Function

The `forward` function performs a forward pass through the neural network. It takes input values `x1` and `x2` and computes the output of the XOR logic gate.

### rand_xor Function

The `rand_xor` function initializes a new `Xor` struct with random weights and biases for the neural networks.

### finite_difference_method Function

The `finite_difference_method` function calculates the gradients of the neural network using the finite difference method. It perturbs each weight and bias parameter of the neural network and calculates the difference in cost. The gradients are then used in the gradient descent step.

### train Function

The `train` function applies the gradient descent step to update the weights and biases of the neural network based on the calculated gradients.

## Conclusion

This project demonstrates how to train a neural network to solve the XOR logic gate using gradient descent. It provides a simple implementation in Rust that can be used as a starting point for more complex neural network applications. Feel free to explore the code and experiment with different parameters and training data.
