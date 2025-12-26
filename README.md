# neural_fractal
A neural fractal approximator built with <a href="https://pytorch.org/">PyTorch</a> that learns fractals such as: <b>Mandelbrot</b> & <b>Julia</b>. Images are rendered progressively during training, allowing monitoring the fractal evolution in real-time.

<img src="fractals/mandelbrot.png" width="128" height="128">  <img src="fractals/julia.png" width="128" height="128">  <img src="fractals/burning_ship.png" width="128" height="128">  <img src="fractals/newton.png" width="128" height="128">

## Features
- Fractal presets: <code>mandelbrot</code>, <code>julia</code>, <code>burning_ship</code> & <code>newton</code>
- Image updates every epoch, reflecting current training process
- Interactive exploration using <b>Jupyter Notebook</b>
- Most parameters are configurable via <b>YAML</b>
- Supports both CPU & GPU

## Requirements
- Python 3.10+
- Torch
- Torchvision
- Numpy
- Pillow
- Pyyaml
- Matplotlib
- Ipywidgets

## Configuration
1. Install requirements: 
```bash
pip install -r requirements.txt
```
2. Most parameters are configurable via <b>YAML</b>:
- <code>configs/default.yaml</code>: training & generation settings, including fractal preset.
- <code>configs/fractal.yaml</code>: fractal-specific parameters for each preset.

3. Train the model:
```bash
python train.py
```

4. Open <code>notebooks/exploration.ipynb</code> to pan, zoom, and adjust resolution interactively.

## License
MIT License