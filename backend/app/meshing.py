import torch
import torch.nn as nn
import open3d as o3d
import numpy as np

# Define Model Again Before Loading
class AIMeshOptimizer(nn.Module):
    def __init__(self):
        super(AIMeshOptimizer, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load model with weights_only=False
mesh_optimizer_model = AIMeshOptimizer()
mesh_optimizer_model.load_state_dict(torch.load("app/models/mesh_optimizer.pth", map_location=torch.device("cpu"), weights_only=False))
mesh_optimizer_model.eval()

def mesh_optimization(mesh):
    """
    AI-driven mesh optimization using deep learning.
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # Convert to PyTorch tensor
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)

    # Pass through AI model
    with torch.no_grad():
        optimized_vertices = mesh_optimizer_model(vertices_tensor).numpy()

    # Update mesh with optimized vertices
    optimized_mesh = o3d.geometry.TriangleMesh()
    optimized_mesh.vertices = o3d.utility.Vector3dVector(optimized_vertices)
    optimized_mesh.triangles = o3d.utility.Vector3iVector(faces)

    return optimized_mesh
