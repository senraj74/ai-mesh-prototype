import torch
import torch.nn as nn

# Define the AI Model
class AIModel(nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

# Create Model
model = AIModel()

# Save the state_dict properly
torch.save(model.state_dict(), "ai_model.pth")
print("Model state_dict saved successfully as 'ai_model.pth'")
