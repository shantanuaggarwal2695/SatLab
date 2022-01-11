from flask import Flask
from flask import request
from logic import *

app = Flask(__name__)
spark = initiate_session()


class Loader:
    def __init__(self, path):
        self.path = path

    def __int__(self):
        self.path = ""


loader = Loader()


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
            result = run_job(loader.path, spark)
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


if __name__ == '__main__':
    app.run(debug=True)
