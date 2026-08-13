# Step 2 FedAvg Results

This step simulates Federated Learning (FedAvg) without DP noise using ResNet50.

- **Local Epochs (E)**: 5
- **Total Rounds (T)**: 100

## Test Accuracy Comparison

| Model | K (Clients) | Mode | Final Test Accuracy | Target (Paper No-Noise) | Difference |
|---|---|---|---|---|---|
| ResNet50 | 3 | binary | 77.67% | 83.05% | -5.38% |
| ResNet50 | 5 | binary | 78.64% | 83.05% | -4.41% |
