# Self-Pruning Neural Network

A PyTorch implementation of a neural network that learns to prune itself during training using learnable sigmoid gates and an L1 sparsity penalty.


## Overview

This project builds a feed-forward neural network for CIFAR-10 classification where each weight has a learnable gate.  
The gate value is passed through a sigmoid and multiplied with the weight, so weights with low gate values become effectively pruned during training.

The goal is to study the trade-off between:
- test accuracy,
- model sparsity,
- and pruning strength controlled by lambda.


## Problem Statement

Traditional pruning is usually done after training.  
In this project, pruning happens **during training itself**.

For each weight (w_{ij}), there is a corresponding gate score (g_{ij}).  
The effective weight becomes: w_{ij} * sigma(g_{ij})

Where:
- sigma is the sigmoid function,
- gates close to 0 deactivate weights,
- gates close to 1 keep weights active.

To encourage pruning, an L1-style sparsity penalty is added to the loss function.

## Model Architecture

- Flatten input image
- `PrunableLinear(3072 -> 512)`
- BatchNorm + ReLU
- `PrunableLinear(512 -> 256)`
- BatchNorm + ReLU
- `PrunableLinear(256 -> 128)`
- BatchNorm + ReLU
- `PrunableLinear(128 -> 10)`


## How It Works

Each custom linear layer contains:
- `weight`
- `bias`
- `gate_scores`

During forward pass:
1. Gate scores are converted into gate values using sigmoid.
2. Weights are multiplied by the gates.
3. The pruned weights are used in the linear layer.

The sparsity loss pushes gates toward 0, which makes more weights inactive.


## Loss Function

The total loss is:

Total Loss = Classification Loss + lambda * Sparsity Loss

Where:
- Classification Loss = CrossEntropyLoss
- Sparsity Loss = mean of gate values across all prunable layers
- lambda\ controls the balance between accuracy and sparsity

A higher lambda usually creates more sparsity, but may reduce accuracy.

## Results Summary

| Lambda | Test Accuracy | Sparsity Level (%) |
|---|---:|---:|
| 1e-07 | 56.6% | 0.0% |
| 5e-07 | 57.3% | 2.6% |
| 1e-06 | 56.9% | 10.2% |
| 5e-06 | 57.4% | 41.9% |
| 1e-05 | 56.4% | 59.0% |

### Observation
- Smaller lambda values keep most gates active.
- Larger lambda values prune more aggressively.
- In this run, `5e-06` gave the best balance between accuracy and sparsity.

---

## Gate Distribution Plot

The best model shows a gate distribution with:
- a large spike near 0,
- another cluster away from 0.

This confirms that the model has learned to prune many weights while keeping the important ones active.

The final plot is saved as: self_pruning_report.png

---

## Files in This Repo

- `train_self_pruning.py` — main training and evaluation script
- `README.md` — project documentation
- `self_pruning_report.png` — final gate distribution and results plot

---

## Requirements

```bash
pip install torch torchvision matplotlib numpy
```

---

## How to Run

```bash
python train_self_pruning.py
```

This will:
- train the self-pruning network,
- run multiple lambda values,
- print accuracy and sparsity results,
- generate the gate distribution plot,
- save the plot to `self_pruning_report.png`.


## Why This Is Useful

This project demonstrates how a neural network can learn to reduce its own complexity during training.  
It is a simple but effective example of sparse model learning and regularization-based pruning.


## Conclusion

This self-pruning approach shows a clear trade-off between sparsity and accuracy.  
It is useful when you want a model that automatically removes unimportant connections while still learning the task.

## Author

Built as part of an AI Engineering case study focused on pruning, sparsity, and efficient PyTorch model design.
