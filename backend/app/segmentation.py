import open3d as o3d
import torch
import numpy as np
# from pointnet2.models import Pointnet2SSG

class AIModel(torch.nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()
        self.fc1 = torch.nn.Linear(3, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

# Load AI Model
model = AIModel()
model.load_state_dict(torch.load("app/models/ai_model.pth", map_location="cpu"))
model.eval()

def segment_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Validate mesh
    if mesh.is_empty():
        raise ValueError(" Error: Loaded mesh is empty!")

    # Convert to Point Cloud
    point_cloud = mesh.sample_points_uniformly(number_of_points=8000)  # Increased points for better meshing

    # Check if Point Cloud is Empty
    if len(point_cloud.points) < 50:
        raise ValueError(" Error: Not enough points to generate a mesh!")

    # Clean Up Noisy Points (Remove Outliers)
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)

    # Estimate Normals for Poisson
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Apply AI Segmentation (Color Labeling)
    points = np.asarray(point_cloud.points)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        output = model(points_tensor).numpy()

    labels = (output[:, 1] > 0.5).astype(int)
    colors = np.zeros((points.shape[0], 3))
    colors[labels == 1] = [1, 0, 0]  # Red for segmented part
    colors[labels == 0] = [0, 0, 1]  # Blue for non-segmented part
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Try Poisson Surface Reconstruction with Lower Depth
    try:
        print("⚠️ Running Poisson Surface Reconstruction (depth=8)...")
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=8)
        
        # Filter low-density areas to remove unwanted artifacts
        density_threshold = np.percentile(np.asarray(densities), 25)
        vertices_to_keep = np.asarray(densities) > density_threshold
        poisson_mesh.remove_vertices_by_mask(vertices_to_keep)

        # Compute normals for better rendering
        poisson_mesh.compute_vertex_normals()

        print("Poisson Reconstruction Succeeded!")
        return poisson_mesh

    except Exception as e:
        raise RuntimeError(f" Poisson reconstruction failed: {e}")
