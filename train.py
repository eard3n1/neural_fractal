import os
import torch
from PIL import Image
from torch.utils.data import DataLoader

from architecture.config import load_config
from architecture.model import FractalNet
from architecture.dataset import FractalDataset
from architecture.targets import targets


def render_fractal(model: torch.nn.Module, resolution: int, device: str) -> Image.Image:
    x = torch.linspace(-2, 2, resolution)
    y = torch.linspace(-2, 2, resolution)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)
    with torch.no_grad():
        rgb = model(coords).clamp(0, 1).cpu().numpy()
    return Image.fromarray((rgb.reshape(resolution, resolution, 3) * 255).astype("uint8"))


defaults = load_config("configs/default.yaml")
fractals = load_config("configs/fractal.yaml")

model_cfg = defaults["model"]
dataset_cfg = defaults["dataset"]
training_cfg = defaults["training"]
generation_cfg = defaults["generation"]
paths_cfg = defaults["paths"]

fractal = fractals[generation_cfg["preset"]]
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FractalDataset(num_samples=dataset_cfg["num_samples"],scale=dataset_cfg["scale"])
dataloader = DataLoader(dataset, batch_size=dataset_cfg["batch_size"], shuffle=True)

model = FractalNet(
    hidden_dim=model_cfg["hidden_dim"],
    num_layers=model_cfg["num_layers"],
    num_frequencies=model_cfg["num_frequencies"],
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["lr"])
criterion = torch.nn.MSELoss()

live_path = paths_cfg["live"]
os.makedirs(os.path.dirname(live_path), exist_ok=True)
log_path = paths_cfg["log"]
os.makedirs(os.path.dirname(log_path), exist_ok=True)

resolution = generation_cfg["resolution"]
epochs = training_cfg["epochs"]

loss_history: list[float] = []

if __name__ == "__main__":
    for epoch in range(epochs):
        total_loss = 0.0
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
        img.save(live_path + "fractal.png")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            img.save(log_path + f"epoch_{epoch + 1}.png")

    with open(log_path + "log.csv", "w") as f:
        f.write("epoch,loss\n")
        for i, loss in enumerate(loss_history):
            f.write(f"{i + 1},{loss:.6f}\n")