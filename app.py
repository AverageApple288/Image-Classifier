import os
import tarfile
import zipfile
from glob import glob
import matplotlib.pylab as plt
import cupy as cp
import numpy as np  # Keep numpy for compatibility with matplotlib
import time

from backend.convolve import convolve
from backend.convolve_optimized import convolve_optimized, batch_convolve
from backend.filter_generation import generate_layer
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

@app.route('/train', methods=['GET'])
def train_model():
    upload1_images = glob(os.path.join(app.static_folder, 'extracted_files1', filename1, '*'))
    upload2_images = glob(os.path.join(app.static_folder, 'extracted_files2', filename2, '*'))

    try:
        x_gpu = cp.array([1, 2, 3])
        print("CuPy array created successfully on the GPU.")
        print(x_gpu)
        print("Number of GPUs available:", cp.cuda.runtime.getDeviceCount())
        std_dev = cp.sqrt(2.0 / 0.5)
        print("Standard deviation calculated successfully:", std_dev)
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"Error: CuPy could not initialize. Check your CUDA installation. Error details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    # Initialize lists to store images
    pixelinfo1 = []
    pixelinfo2 = []

    # Read images and store them in lists
    for image_path in upload1_images:
        pixelinfo1.append(plt.imread(image_path))
    
    for image_path in upload2_images:
        pixelinfo2.append(plt.imread(image_path))

    filter_height = 3
    filter_width = 3
    in_channels = 3
    out_channels = 16

    # Initialize list for layers
    layers = []

    # Generate filters and store them
    for i in range(5):
        layer = generate_layer(filter_height, filter_width, in_channels, out_channels)
        layers.append(layer)
        print(f"Generated filters for layer {i + 1} with shape: {layer.shape}")
        out_channels *= 2

    print("Generated filters for 5 layers.")

    # Convert numpy arrays to cupy arrays for GPU processing
    pixelinfo1_gpu = [cp.asarray(img) for img in pixelinfo1]
    pixelinfo2_gpu = [cp.asarray(img) for img in pixelinfo2]
    layers_gpu = [cp.asarray(layer) for layer in layers]

    normal_maps_1 = [[[] for j in range(5)] for i in range(len(pixelinfo1_gpu))]

    for i in range(len(pixelinfo1_gpu)):
        startTime = time.time()
        print(f"Processing image {i + 1}/{len(pixelinfo1_gpu)}")

        for j in range(16):
            #print("Started processing layer 0: " + str(time.time() - startTime))
            filter = layers_gpu[0][:,:,:,j]
            #print("Filter array initialised: " + str(time.time() - startTime))
            #print("Convolution started: " + str(time.time() - startTime))
            normal_map = convolve_optimized(pixelinfo1_gpu[i], filter)  # Using optimized convolution
            #print("Convolution done: " + str(time.time() - startTime))
            #print("Applying ReLU activation: " + str(time.time() - startTime))
            normal_map = cp.maximum(0, normal_map)
            #print("ReLU activation appli" + str(time.time() - startTime))
            #print("Max pooling started: " + str(time.time() - startTime))
            normal_map = max_pool(normal_map, pool_size=2, stride=2)
            #print("Max pooling done: " + str(time.time() - startTime))
            #print("Appending normal map to the list: " + str(time.time() - startTime))
            normal_maps_1[i][0].append(normal_map)
            #print("Normal map appended: " + str(time.time() - startTime))
        first_layer = time.time()
        print(f"Processed layer 0 for image {i + 1}: {first_layer - startTime} seconds")

        for j in range(32):
            filter = layers_gpu[1][:,:,:,j]
            input_maps = cp.concatenate(normal_maps_1[i][0], axis=-1)
            normal_map = convolve_optimized(input_maps, filter)  # Using optimized convolution
            normal_map = cp.maximum(0, normal_map)
            normal_map = max_pool(normal_map, pool_size=2, stride=2)
            normal_maps_1[i][1].append(normal_map)

        second_layer = time.time()
        print(f"Processed layer 1 for image {i + 1}: {second_layer - first_layer} seconds")

        for j in range(64):
            filter = layers_gpu[2][:,:,:,j]
            input_maps = cp.concatenate(normal_maps_1[i][1], axis=-1)
            normal_map = convolve_optimized(input_maps, filter)  # Using optimized convolution
            normal_map = cp.maximum(0, normal_map)
            normal_map = max_pool(normal_map, pool_size=2, stride=2)
            normal_maps_1[i][2].append(normal_map)

        third_layer = time.time()
        print(f"Processed layer 2 for image {i + 1}: {third_layer - second_layer} seconds")

        for j in range(128):
            filter = layers_gpu[3][:,:,:,j]
            input_maps = cp.concatenate(normal_maps_1[i][2], axis=-1)
            normal_map = convolve_optimized(input_maps, filter)  # Using optimized convolution
            normal_map = cp.maximum(0, normal_map)
            normal_map = max_pool(normal_map, pool_size=2, stride=2)
            normal_maps_1[i][3].append(normal_map)

        fourth_layer = time.time()
        print(f"Processed layer 3 for image {i + 1}: {fourth_layer - third_layer} seconds")

        for j in range(256):
            filter = layers_gpu[4][:,:,:,j]
            input_maps = cp.concatenate(normal_maps_1[i][3], axis=-1)
            normal_map = convolve_optimized(input_maps, filter)  # Using optimized convolution
            normal_map = cp.maximum(0, normal_map)
            normal_map = max_pool(normal_map, pool_size=2, stride=2)
            normal_maps_1[i][4].append(normal_map)

        fifth_layer = time.time()
        print(f"Processed layer 4 for image {i + 1}: {fifth_layer - fourth_layer} seconds")

    # Convert the output shape back to numpy for compatibility with the rest of the code
    print(normal_maps_1[0][4][0].shape)

    # Clear GPU memory
    cp.get_default_memory_pool().free_all_blocks()

    return render_template('index.html', message='Training complete using GPU acceleration.')

if __name__ == '__main__':
    app.run(debug=True)