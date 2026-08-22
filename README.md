# Machine Learning Library From Scratch

A collection of core machine learning algorithms implemented from scratch in Python — no NumPy, no PyTorch, no external ML libraries. Built to understand the math behind each algorithm, not just use it.

## Why

Most ML libraries hide the math behind function calls. This project implements everything manually — matrix operations, gradient descent, backpropagation — using plain Python lists and loops, to actually understand what's happening at each step.

## Contents

- **`linear_algebra.py`** — Matrix operations from scratch (matrix multiplication, transpose, elementwise operations, validation) used as the foundation for everything else.
- **`perceptron.py`** — A single-layer perceptron with configurable learning rate, bias, and epochs. Uses step activation and updates weights based on prediction error.
- **`logistic_regression.py`** — Logistic regression using sigmoid activation and binary cross-entropy loss, trained with gradient descent.
- **`mlp.py`** — A multilayer perceptron (dynamic number of layers) with forward propagation, backpropagation, and gradient descent, built entirely on the matrix operations above.
- **`ML_Library_From_Scratch(3).ipynb`** — Notebook version used for development, testing, and experimentation. Each cell represents a different version of the same code. In other words, any improvement I make to the V1.0 of the code will be implemented in V2.0 in the subsequent cell.

## How it's built

Each algorithm follows the same core pipeline:

```
input → weighted sum (+ bias) → activation → prediction → loss → gradients → parameter update
```

For the MLP specifically:

```
X → W1·X + b1 → sigmoid → A1 → W2·A1 + b2 → sigmoid → ŷ → loss → backprop → update weights
```

## Status

- [x] Perceptron
- [x] Logistic regression
- [x] Matrix operations
- [x] MLP (forward pass + backpropagation, dynamic layers)
- [ ] Batch training / multiple samples per epoch
- [ ] Additional activation functions

## Usage

Each module can be imported and used independently. See the notebook for example usage and test runs (e.g. XOR for the MLP).

## Requirements

Python 3, standard library only (`math`).