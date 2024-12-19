import torch
import torch.nn as nn

# Example data: 3 classes
logits = torch.tensor(
    [
        [2.0, 0.5, 0.1],  # Prediction logits for sample 1
        [0.1, 1.5, 2.1],  # Prediction logits for sample 2
        [1.0, 2.0, 0.1],
    ]
)  # Prediction logits for sample 3

targets = torch.tensor([0, 2, 1])  # Ground truth labels

# Define CrossEntropyLoss without weights
criterion_no_weight = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)

# Define CrossEntropyLoss with weights
weights = torch.tensor([0.7, 1.0, 1.5])  # Assign different weights to each class
criterion_with_weight = nn.CrossEntropyLoss(
    weight=weights, reduction="none", label_smoothing=0.1
)

# Compute loss without weights
loss_no_weight = criterion_no_weight(logits, targets)

# Compute loss with weights
loss_with_weight = criterion_with_weight(logits, targets)

selected_weight = torch.gather(weights, dim=0, index=targets)
print(f"{selected_weight=}")
print("Loss without weights:", loss_no_weight)
print("Loss with weights:", loss_with_weight)
print("====================================================")
# Define CrossEntropyLoss without weights
criterion_no_weight = nn.CrossEntropyLoss(reduction="none")

# Define CrossEntropyLoss with weights
weights = torch.tensor([0.7, 1.0, 1.5])  # Assign different weights to each class
criterion_with_weight = nn.CrossEntropyLoss(weight=weights, reduction="none")

# Compute loss without weights
loss_no_weight = criterion_no_weight(logits, targets)

# Compute loss with weights
loss_with_weight = criterion_with_weight(logits, targets)
print(f"{selected_weight=}")

print("Loss without weights:", loss_no_weight)
print("Loss with weights:", loss_with_weight)
