import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define AI Model for Mesh Optimization
class AIMeshOptimizer(nn.Module):
    def __init__(self):
        super(AIMeshOptimizer, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output: Optimized (x, y, z)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Create Model
mesh_optimizer_model = AIMeshOptimizer()

# Generate Training Data (Replace with real data)
X_train = torch.randn(5000, 3)  # 5000 random (x, y, z) positions
y_train = X_train + torch.randn(5000, 3) * 0.005  # Small mesh refinements

# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(mesh_optimizer_model.parameters(), lr=0.001)

# Train Model
print("Training AI Mesh Optimization Model...")
for epoch in range(200):
    optimizer.zero_grad()
    outputs = mesh_optimizer_model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.6f}")

# Save Only State Dict Properly
torch.save(mesh_optimizer_model.state_dict(), "mesh_optimizer.pth")
print("AI Mesh Optimization Model saved successfully.")
