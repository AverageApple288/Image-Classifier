# GPU-Accelerated Image Classifier

This application uses CuPy to accelerate image processing operations on NVIDIA GPUs.

## Requirements

- Python 3.13.5
- NVIDIA GPU (Optimized for RTX 3060)
- CUDA Toolkit (compatible with CuPy version)

## Setup for Arch Linux

1. Install CUDA Toolkit:

```bash
sudo pacman -S cuda
```

2. Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

If you have issues installing CuPy from pip, you can try installing it from the AUR:

```bash
yay -S python-cupy
```

## Running the Application

```bash
python app.py
```

The application will be available at http://127.0.0.1:5000/

## GPU Memory Management

This application automatically manages GPU memory, but you can monitor GPU usage with:

```bash
nvidia-smi
```

## Troubleshooting

If you encounter CUDA errors, check that:

1. Your NVIDIA drivers are properly installed
2. Your CUDA Toolkit version is compatible with CuPy
3. You have sufficient GPU memory available
