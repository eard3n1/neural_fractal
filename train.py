import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
from src.config import Config
from src.architecture.model import FractalNet
from src.datasets.fractal import FractalDataset
from src.targets.procedural import targets

def render_fractal(model, resolution=512, device="cpu"):
    x = torch.linspace(-2, 2, resolution)
    y = torch.linspace(-2, 2, resolution)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)

    with torch.no_grad():
        rgb = model(coords).clamp(0, 1).cpu().numpy()

    img_array = (rgb.reshape(resolution, resolution, 3) * 255).astype("uint8")
    return Image.fromarray(img_array)

if __name__ == "__main__":
    cfg_train = Config("configs/default.yaml")
    cfg_fractal_all = Config("configs/fractal.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preset_name = cfg_train.get("generation", "preset")
    cfg_fractal = cfg_fractal_all.get(preset_name)

    dataset = FractalDataset(
        num_samples=cfg_train.get("dataset", "num_samples"),
        scale=cfg_train.get("dataset", "scale")
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg_train.get("dataset", "batch_size"),
        shuffle=True
    )

    model = FractalNet(
        input_dim=cfg_train.get("model", "input_dim"),
        hidden_dim=cfg_train.get("model", "hidden_dim"),
        num_layers=cfg_train.get("model", "num_layers")
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.get("training", "lr"))
    criterion = torch.nn.MSELoss()

    model_path = cfg_train.get("training", "model_path")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    output_path = cfg_train.get("generation", "output_path")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    resolution = cfg_train.get("generation", "resolution")
    epochs = cfg_train.get("training", "epochs")

    for epoch in range(epochs):
        total_loss = 0
        for coords in dataloader:
            coords = coords.to(device)
            target = targets(coords, cfg_fractal).to(device)

            optimizer.zero_grad()
            output = model(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

        img = render_fractal(model, resolution=resolution, device=device)
        img.save(output_path)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")