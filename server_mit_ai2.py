from flask import Flask, jsonify, request, send_from_directory
import numpy as np
from pprint import pprint

from model import predict_and_draw

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_img():
    file = request.data
    cur_id = str(np.random.randint(0, int(1e10))).zfill(10)

    with open(f"to_prc/{cur_id}.jpg", "wb") as f:
        f.write(file)
    result = predict_and_draw(f"to_prc/{cur_id}.jpg", f"gen_img/{cur_id}.jpg")

    pprint(result)
    return jsonify(result)


@app.route("/gen_img/<path:filepath>")
def gen_img(filepath):
    return send_from_directory("gen_img", filepath)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
