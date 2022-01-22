from logic import *
from flask_cors import CORS
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/hdd2/shantanuCodeData/data/demo_parquet/train_demo'
ALLOWED_EXTENSIONS = {'.tif', '.tiff'}
app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


class Server:
    def __init__(self):
        self.path = None
        self.lfs = None
        self.spark = None
        # self.spatial = None
        # self.texture = None


loader = Server()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/satlab/load', methods=['GET', 'POST'])
def load():


    # data = request.get_json(force=True)
    # print(data)
    # path = data['path']
    # loader.path = path
    # print(path)
    # return {}
    # print("helloooooooooooooooooo")
    if request.method == 'POST':
        # check if the post request has the file part
        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(request.url)

        print(request.FileList)
        # file = request.FileList['file']
        # # If the user does not select a file, the browser submits an
        # # empty file without a filename.
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #     return redirect(url_for('download_file', name=filename))
    return {}



@app.route('/satlab/label', methods=['POST'])
def user():
    if request.method == 'POST':
        try:
            result = run_job(loader.path, loader.spark, loader.lfs)
        except ValueError:
            print("wrong path value")

        print(result)

        return {"data": [
            {"ID": image['origin'], "Geom": image['Geom'], "Label": image['Label']}
            for image in result
        ]}
        # response = []
        # record = dict()
        # for row in result:
        #     record["id"] = row[0]
        #     record["label"] = row[1]
        #     response.append(record)
        #
        # return response


@app.route('/satlab/labelingfunctions', methods=['POST'])
def texture():
    data = request.get_json()
    loader.lfs = data['lf_index']
    return {}


# @app.route('/satlab/spatial', method=['POST'])
# def spatial():
#     data = request.form
#     loader.spatial = data['spatial']
#     return {}


if __name__ == '__main__':
    loader.spark = initiate_session()


    app.run(debug=True)
