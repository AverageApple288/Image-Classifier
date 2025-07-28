import os
import tarfile
import zipfile
from glob import glob
import matplotlib.pylab as plt
import numpy as np

from backend.convolve import convolve
from backend.filter_generation import generate_filter

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
        layer = generate_filter(filter_height, filter_width, in_channels, out_channels)
        layers.append(layer)
        print(f"Generated filters for layer {i + 1} with shape: {layer.shape}")
        out_channels *= 2

    print("Generated filters for 5 layers.")

    # Example of accessing pixel information
    single_filter = layers[0][:, :, :, 0]
    print(single_filter.shape)
    print(pixelinfo1[0].shape)
    print(pixelinfo2[0].shape)

    normal_map = convolve(pixelinfo1[0], single_filter)
    normal_map = np.maximum(0, normal_map)

    print(f"Convolved image shape: {normal_map.shape}")
    print(normal_map)

    return render_template('index.html', message='Training complete.')

if __name__ == '__main__':
    app.run(debug=True)