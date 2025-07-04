import os
import tarfile
import zipfile

from flask import Flask, render_template, request

app = Flask(__name__)

filename1 = None
filename2 = None

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload1', methods=['POST'])
def upload_file1():
    uploaded_file = request.files['zipfile']
    if uploaded_file:
        global filename1
        filename1 = uploaded_file.filename
        upload_path = os.path.join(app.static_folder, 'uploads1', filename1)
        extract_path = os.path.join(app.static_folder, 'extracted_files1')
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        os.makedirs(extract_path, exist_ok=True)
        uploaded_file.save(upload_path)
        if filename1.endswith('.zip'):
            with zipfile.ZipFile(upload_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            return render_template('index.html', message='Zip file processed and extracted.')
        elif filename1.endswith('.tar.gz') or filename1.endswith('.tar.xz'):
            mode = 'r:gz' if filename1.endswith('.tar.gz') else 'r:xz'
            with tarfile.open(upload_path, mode) as tar_ref:
                tar_ref.extractall(extract_path)
            return render_template('index.html', message=f'{os.path.splitext(filename1)[1]} file processed and extracted.')
        else:
            return render_template('index.html', message='Invalid file type.')
    return render_template('index.html', message='No file uploaded.')

@app.route('/upload2', methods=['POST'])
def upload_file2():
    uploaded_file = request.files['zipfile2']
    if uploaded_file:
        global filename2
        filename2 = uploaded_file.filename
        upload_path = os.path.join(app.static_folder, 'uploads2', filename2)
        extract_path = os.path.join(app.static_folder, 'extracted_files2')
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        os.makedirs(extract_path, exist_ok=True)
        uploaded_file.save(upload_path)
        if filename2.endswith('.zip'):
            with zipfile.ZipFile(upload_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            return render_template('index.html', message='Zip file processed and extracted (dataset two).')
        elif filename2.endswith('.tar.gz') or filename2.endswith('.tar.xz'):
            mode = 'r:gz' if filename2.endswith('.tar.gz') else 'r:xz'
            with tarfile.open(upload_path, mode) as tar_ref:
                tar_ref.extractall(extract_path)
            return render_template('index.html', message=f'{os.path.splitext(filename2)[1]} file processed and extracted (dataset two).')
        else:
            return render_template('index.html', message='Invalid file type (dataset two).')
    return render_template('index.html', message='No file uploaded (dataset two).')

if __name__ == '__main__':
    app.run()
