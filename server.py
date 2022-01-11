from flask import Flask
from flask import request
from logic import *

app = Flask(__name__)
spark = initiate_session()


@app.route('/satlab', methods=['POST'])
def user():
    if request.method == 'POST':
        data = request.form
        path = data["path"]
        result = run_job(path, spark)
        response = []
        record = dict()
        for row in result:
            record["id"] = row[0]
            record["label"] = row[1]
            response.append(record)

        return response


if __name__ == '__main__':
    app.run(debug=True)
