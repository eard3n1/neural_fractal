import os
import torch
from PIL import Image

from torch.utils.data import DataLoader

from src.config import Config
from src.architecture.model import FractalNet
from src.dataset.fractal import FractalDataset
from src.targets.procedural import targets

def render_fractal(model, resolution=512, device="cpu"):
    x = torch.linspace(-2, 2, resolution)
    y = torch.linspace(-2, 2, resolution)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)
    with torch.no_grad():
        rgb = model(coords).clamp(0, 1).cpu().numpy()
    return Image.fromarray((rgb.reshape(resolution, resolution, 3) * 255).astype("uint8"))

default = Config("configs/default.yaml")
fractals = Config("configs/fractal.yaml")
preset = default.get("generation", "preset")
fractal = fractals.get(preset)
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FractalDataset(
    num_samples=default.get("dataset", "num_samples"),
    scale=default.get("dataset", "scale")
)

dataloader = DataLoader(
    dataset,
    batch_size=default.get("dataset", "batch_size"),
    shuffle=True
)

model = FractalNet(
    hidden_dim=default.get("model", "hidden_dim"),
    num_layers=default.get("model", "num_layers"),
    num_frequencies=default.get("model", "num_frequencies")   
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=default.get("training", "lr"))
criterion = torch.nn.MSELoss()

output_path = default.get("paths", "output")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
log_path = default.get("paths", "log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

resolution = default.get("generation", "resolution")
epochs = default.get("training", "epochs")

loss_history = []

if __name__ == "__main__":
    for epoch in range(epochs):
        total_loss = 0
        for coords in dataloader:
            coords = coords.to(device)
            target = targets(coords, fractal).to(device)

            optimizer.zero_grad()
            output = model(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

        img = render_fractal(model, resolution=resolution, device=device)
        img.save(output_path + "fractal.png")

        if (epoch + 1) % 10 == 0 or ((epoch + 1) == epochs):
            img.save(log_path + f"epoch_{epoch + 1}.png")

    with open(log_path + "log.csv", "w") as c:
        c.write("epoch,loss\n")
        for i, loss in enumerate(loss_history):
            c.write(f"{i + 1},{loss:.6f}\n")
