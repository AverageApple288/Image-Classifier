import os
import tarfile
import zipfile
from glob import glob
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

from backend.neural_network import flatten_output, hidden_dense_layer, sigmoid, final_dense_layer, binary_cross_entropy_loss
from backend.process_image import process_image
from backend.filter_generation import generate_filter

from flask import Flask, render_template, request, session

filename1 = None
filename2 = None

app = Flask(__name__)
app.secret_key = 'CrazySecretKey123'

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

@app.route('/train', methods=['GET', 'POST'])
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

    if results2:
        print("Shape of a final feature map from dataset 2:", results2[0][-1][0].shape)

    lenDataset1 = len(results1)
    lenDataset2 = len(results2)

    real_values = np.array([1] * lenDataset1 + [0] * lenDataset2).reshape(-1, 1)

    flattened_results1 = []
    flattened_results2 = []

    for i in range(len(results1)):
        flattened_results1.append(flatten_output(results1[i][-1][0]))
        flattened_results2.append(flatten_output(results2[i][-1][0]))

    batch_input = np.concatenate([flattened_results1, flattened_results2], axis=0)

    epochs = 1000
    #learning_rate = 0.001
    learning_rate = 0.0005
    num_neurons = 128
    num_input_features = flattened_results1[0].shape[0]
    hidden_weights = np.random.randn(num_input_features, num_neurons) # * np.sqrt(1.0 / num_input_features)
    hidden_bias = np.random.randn(num_neurons)
    final_weights = np.random.randn(num_neurons, 1) # * np.sqrt(1.0 / num_neurons)
    final_bias = np.random.randn(1)

    for epoch in range(epochs):
        activated_hidden_output, hidden_output_before_activation = hidden_dense_layer(batch_input, hidden_weights, hidden_bias)

        activated_final_output = final_dense_layer(activated_hidden_output, final_weights, final_bias)

        #print(activated_final_output)

        loss = binary_cross_entropy_loss(real_values, activated_final_output)

        d_loss = activated_final_output - real_values

        d_weights_final = np.dot(activated_hidden_output.T, d_loss)
        d_bias_final = np.sum(d_loss, axis=0)

        d_activated_hidden = np.dot(d_loss, final_weights.T)

        d_hidden_before_activation = d_activated_hidden * (hidden_output_before_activation > 0)

        d_weights_hidden = np.dot(batch_input.T, d_hidden_before_activation)
        d_bias_hidden = np.sum(d_hidden_before_activation, axis=0)

        final_weights -= learning_rate * d_weights_final
        final_bias -= learning_rate * d_bias_final

        hidden_weights -= learning_rate * d_weights_hidden
        hidden_bias -= learning_rate * d_bias_hidden

        print(f"Epoch {epoch + 1}, Loss: {loss}")

    activated_hidden_output, hidden_output_before_activation = hidden_dense_layer(batch_input, hidden_weights, hidden_bias)

    activated_final_output = final_dense_layer(activated_hidden_output, final_weights, final_bias)

    print(activated_final_output.flatten())

    session['final_weights'] = final_weights
    session['final_bias'] = final_bias
    session['hidden_weights'] = hidden_weights
    session['hidden_bias'] = hidden_bias

    return render_template('classify.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify_image():
    final_weights = session.get('final_weights')
    final_bias = session.get('final_bias')
    hidden_weights = session.get('hidden_weights')
    hidden_bias = session.get('hidden_bias')



    return render_template('classify.html')

if __name__ == '__main__':
    app.run(debug=True)