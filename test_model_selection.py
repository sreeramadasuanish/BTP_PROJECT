
import torch
from src.models.registry import get_model

# Create each dataset's model
fashion = get_model("Fashion-MNIST")
cifar = get_model("CIFAR-10")
mnist = get_model("MNIST")
sst = get_model("SST")

# Check the correct models are selected
assert type(fashion).__name__ == "FedCNN"
assert type(cifar).__name__ == "CIFAR10CNN"
assert type(mnist).__name__ == "MLPMnist"
assert type(sst).__name__ == "LSTMAttention"

# Check CIFAR-10 accepts RGB images and returns 10 class scores
output = cifar(torch.randn(2, 3, 32, 32))
assert output.shape == (2, 10), output.shape

print("All model-selection checks passed!")
print("Fashion-MNIST:", type(fashion).__name__)
print("CIFAR-10:", type(cifar).__name__)
print("MNIST:", type(mnist).__name__)
print("SST:", type(sst).__name__)
print("CIFAR-10 output shape:", output.shape)
