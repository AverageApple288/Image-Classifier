import os
import tarfile
import zipfile
from glob import glob
import matplotlib.pylab as plt
import numpy as np
import time
from functools import partial
from multiprocessing import Pool, cpu_count

from backend.convolve import convolve
from backend.filter_generation import generate_filter
from backend.max_pool import max_pool

from flask import Flask, render_template, request

filename1 = None
filename2 = None

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_file1 = request.files.get('zipfile')
    uploaded_file2 = request.files.get('zipfile2')
    messages = []

    # Dataset one
    if uploaded_file1 and uploaded_file1.filename:
        global filename1
        filename1 = uploaded_file1.filename
        upload_path1 = os.path.join(app.static_folder, 'uploads1', filename1)
        extract_path1 = os.path.join(app.static_folder, 'extracted_files1')
        os.makedirs(os.path.dirname(upload_path1), exist_ok=True)
        os.makedirs(extract_path1, exist_ok=True)
        uploaded_file1.save(upload_path1)
        if filename1.endswith('.zip'):
            with zipfile.ZipFile(upload_path1, 'r') as zip_ref:
                zip_ref.extractall(extract_path1)
            messages.append('Dataset one: Zip file processed and extracted.')
        elif filename1.endswith('.tar.gz') or filename1.endswith('.tar.xz'):
            mode = 'r:gz' if filename1.endswith('.tar.gz') else 'r:xz'
            with tarfile.open(upload_path1, mode) as tar_ref:
                tar_ref.extractall(extract_path1)
            messages.append(f'Dataset one: {os.path.splitext(filename1)[1]} file processed and extracted.')
        else:
            messages.append('Dataset one: Invalid file type.')
    else:
        messages.append('Dataset one: No file uploaded.')

    # Dataset two
    if uploaded_file2 and uploaded_file2.filename:
        global filename2
        filename2 = uploaded_file2.filename
        upload_path2 = os.path.join(app.static_folder, 'uploads2', filename2)
        extract_path2 = os.path.join(app.static_folder, 'extracted_files2')
        os.makedirs(os.path.dirname(upload_path2), exist_ok=True)
        os.makedirs(extract_path2, exist_ok=True)
        uploaded_file2.save(upload_path2)
        if filename2.endswith('.zip'):
            with zipfile.ZipFile(upload_path2, 'r') as zip_ref:
                zip_ref.extractall(extract_path2)
            messages.append('Dataset two: Zip file processed and extracted.')
        elif filename2.endswith('.tar.gz') or filename2.endswith('.tar.xz'):
            mode = 'r:gz' if filename2.endswith('.tar.gz') else 'r:xz'
            with tarfile.open(upload_path2, mode) as tar_ref:
                tar_ref.extractall(extract_path2)
            messages.append(f'Dataset two: {os.path.splitext(filename2)[1]} file processed and extracted.')
        else:
            messages.append('Dataset two: Invalid file type.')
    else:
        messages.append('Dataset two: No file uploaded.')

    trainbutton = "<a href='/train' id='trainButton' style='width:75%'><button>Train Model</button></a>"

    filename1 = os.path.splitext(filename1)[0]
    filename2 = os.path.splitext(filename2)[0]

    return render_template('index.html', trainbutton=trainbutton, message=' '.join(messages))

numProcessed = 0

def process_image(image_path, layers):
    """Processes a single image through all layers."""
    pixelinfo = plt.imread(image_path)
    normal_maps = [[] for _ in range(len(layers))]

    # Layer 0
    out_channels_layer0 = layers[0].shape[3]
    for j in range(out_channels_layer0):
        filter_slice = layers[0][:, :, :, j]
        normal_map = convolve(pixelinfo, filter_slice) # Output is 2D
        normal_map = np.maximum(0, normal_map)
        # Add channel dimension for max_pool
        normal_map = max_pool(normal_map[:, :, np.newaxis], pool_size=2, stride=2)
        normal_maps[0].append(normal_map)

    # Subsequent layers
    for i in range(1, len(layers)):
        # Concatenate along the channel axis
        input_maps = np.concatenate(normal_maps[i-1], axis=-1)
        out_channels = layers[i].shape[3]
        for j in range(out_channels):
            filter_slice = layers[i][:, :, :, j]
            normal_map = convolve(input_maps, filter_slice) # Output is 2D
            normal_map = np.maximum(0, normal_map)
            # Add channel dimension for max_pool
            normal_map = max_pool(normal_map[:, :, np.newaxis], pool_size=2, stride=2)
            normal_maps[i].append(normal_map)

    # Return the final feature maps for the image
    return normal_maps

@app.route('/train', methods=['GET'])
def train_model():
    global filename1, filename2
    upload1_images = glob(os.path.join(app.static_folder, 'extracted_files1', filename1, '*'))
    upload2_images = glob(os.path.join(app.static_folder, 'extracted_files2', filename2, '*'))

    filter_height = 3
    filter_width = 3
    in_channels = 3
    out_channels = 16

    layers = []
    for i in range(5):
        layer = generate_filter(filter_height, filter_width, in_channels, out_channels)
        layers.append(layer)
        print(f"Generated filters for layer {i + 1} with shape: {layer.shape}")
        # For subsequent layers, in_channels will be the out_channels of the previous one
        in_channels = out_channels
        out_channels *= 2

    print("Generated filters for 5 layers.")
    startTime = time.time()

    # Use multiprocessing to process images in parallel
    num_processes = cpu_count()
    print(f"Starting image processing with {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        # We use partial to pass the 'layers' argument to process_image
        process_func = partial(process_image, layers=layers)

        print("Processing dataset 1...")
        results1 = pool.map(process_func, upload1_images)

        print("Processing dataset 2...")
        results2 = pool.map(process_func, upload2_images)

    endTime = time.time()
    print(f"Finished processing all images in {endTime - startTime:.2f} seconds.")

    # `results1` and `results2` now contain the processed feature maps for each image
    # Example: access the first image's final layer's first feature map
    if results1:
        print("Shape of a final feature map from dataset 1:", results1[0][-1][0].shape)


    return render_template('index.html', message='Training complete.')

if __name__ == '__main__':
    app.run(debug=True)