import torch

def targets(coords, cfg):
    preset_name = cfg.get("name", "mandelbrot").lower()
    max_iter = cfg.get("max_iter", 50)
    color_map = cfg.get("color_map", {})
    r_freq = color_map.get("r_freq", 5)
    g_freq = color_map.get("g_freq", 7)
    g_phase = color_map.get("g_phase", 2)
    b_freq = color_map.get("b_freq", 11)
    b_phase = color_map.get("b_phase", 4)

    if preset_name == "julia":
        c = cfg.get("c", {"x": -0.7, "y": 0.27015})
        cx = c.get("x", -0.7)
        cy = c.get("y", 0.27015)
        zx, zy = coords[:, 0].clone(), coords[:, 1].clone()
        div_time = torch.zeros_like(zx)
        for _ in range(max_iter):
            zx_new = zx * zx - zy * zy + cx
            zy_new = 2 * zx * zy + cy
            zx, zy = zx_new, zy_new
            mask = (zx * zx + zy * zy) < 4
            div_time += mask.float()

    elif preset_name == "sine_swirl":
        x = coords[:, 0]
        y = coords[:, 1]
        div_time = torch.zeros_like(x)

        for _ in range(max_iter):
            x_new = torch.sin(3 * x) - torch.cos(5 * y)
            y_new = torch.sin(4 * y) - torch.cos(7 * x)
            dist = torch.sqrt(x_new**2 + y_new**2)
            div_time += dist

            x, y = x_new, y_new

    else: # mandelbrot
        x0 = coords[:, 0]
        y0 = coords[:, 1]
        zx = torch.zeros_like(x0)
        zy = torch.zeros_like(y0)
        div_time = torch.zeros_like(x0)
        for _ in range(max_iter):
            zx_new = zx * zx - zy * zy + x0
            zy_new = 2 * zx * zy + y0
            zx, zy = zx_new, zy_new
            mask = (zx * zx + zy * zy) < 4
            div_time += mask.float()

    norm = div_time / max_iter
    r = torch.sin(r_freq * norm)
    g = torch.sin(g_freq * norm + g_phase)
    b = torch.sin(b_freq * norm + b_phase)
    rgb = torch.stack([(r + 1) / 2, (g + 1) / 2, (b + 1) / 2], dim=1)
    return rgb