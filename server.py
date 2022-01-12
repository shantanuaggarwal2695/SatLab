from flask import Flask
from flask import request
from logic import *

app = Flask(__name__)
spark = initiate_session()


class Server:
    def __init__(self):
        self.path = None
        self.spatial = None
        self.texture = None


loader = Server()


@app.route('/satlab/load', methods=['POST'])
def load():
    data = request.form
    path = data['path']
    loader.path = path
    return {}


@app.route('/satlab/label', methods=['POST'])
def user():
    if request.method == 'POST':
        try:
            result = run_job(loader.path, spark, texture, spatial)
        except ValueError:
            print("wrong path value")

        print(result)

        return {}
        # response = []
        # record = dict()
        # for row in result:
        #     record["id"] = row[0]
        #     record["label"] = row[1]
        #     response.append(record)
        #
        # return response


@app.route('/satlab/textural', methods=['POST'])
def texture():
    data = request.form
    loader.texture = data['texture']
    return {}


@app.route('/satlab/spatial', method=['POST'])
def spatial():
    data = request.form
    loader.spatial = data['spatial']
    return {}


if __name__ == '__main__':
    app.run(debug=True)
