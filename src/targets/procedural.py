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
        """
        Julia set

        Iteration:
            z_{n+1} = z_n^2 + c
            z = x + iy,  c = c_x + i c_y

        Real form:
            x_{n+1} = x_n^2 - y_n^2 + c_x
            y_{n+1} = 2 x_n y_n + c_y
        """
        c = cfg.get("c", {"x": -0.7, "y": 0.27015})
        cx = c.get("x", -0.7)
        cy = c.get("y", 0.27015)

        zx = coords[:, 0].clone()
        zy = coords[:, 1].clone()
        div_time = torch.zeros_like(zx)

        for _ in range(max_iter):
            zx_new = zx * zx - zy * zy + cx
            zy_new = 2 * zx * zy + cy
            zx, zy = zx_new, zy_new
            mask = (zx * zx + zy * zy) < 4
            div_time += mask.float()

    elif preset_name == "burning_ship":
        """
        Burning Ship fractal

        Iteration:
            z_{n+1} = (|Re(z_n)| + i|Im(z_n)|)^2 + c

        Real form:
            x_{n+1} = |x_n|^2 - |y_n|^2 + x_0
            y_{n+1} = 2 |x_n| |y_n| + y_0
        """
        x0 = coords[:, 0]
        y0 = coords[:, 1]

        zx = torch.zeros_like(x0)
        zy = torch.zeros_like(y0)
        div_time = torch.zeros_like(x0)

        for _ in range(max_iter):
            zx_abs = torch.abs(zx)
            zy_abs = torch.abs(zy)
            zx_new = zx_abs * zx_abs - zy_abs * zy_abs + x0
            zy_new = 2 * zx_abs * zy_abs + y0
            zx, zy = zx_new, zy_new
            mask = (zx * zx + zy * zy) < 4
            div_time += mask.float()

    elif preset_name == "newton":
        """
        Newton fractal

        Fractal: (f(z) = z^3 - 1)

        Iteration:
            z_{n+1} = z_n - f(z_n) / f'(z_n)

        Where:
            f(z)  = z^3 - 1
            f'(z) = 3 z^2

        Expanded real form:
            x_{n+1} = (2/3)x_n + (x_n^2 - y_n^2) / (3(x_n^2 + y_n^2)^2)
            y_{n+1} = (2/3)y_n - (2 x_n y_n) / (3(x_n^2 + y_n^2)^2)
        """
        zx = coords[:, 0].clone()
        zy = coords[:, 1].clone()

        div_time = torch.zeros_like(zx)

        for _ in range(max_iter):
            r2 = zx * zx + zy * zy
            denom = 3 * r2 * r2 + 1e-6

            zx_new = ((2 / 3) * zx + (zx * zx - zy * zy) / denom)
            zy_new = ((2 / 3) * zy - (2 * zx * zy) / denom)

            diff = torch.sqrt((zx_new - zx) ** 2 + (zy_new - zy) ** 2)
            div_time += (diff < 1e-3).float()

            zx, zy = zx_new, zy_new

    elif preset_name == "mandelbrot":
        """
        Mandelbrot set

        Iteration:
            z_{n+1} = z_n^2 + c
            z_0 = 0,  c = x_0 + i y_0

        Real form:
            x_{n+1} = x_n^2 - y_n^2 + x_0
            y_{n+1} = 2 x_n y_n + y_0
        """
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