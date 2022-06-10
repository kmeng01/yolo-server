from flask import Flask, jsonify, request, send_from_directory
import numpy as np
import sys
from pprint import pprint
from pathlib import Path

from model import predict_and_draw

app = Flask(__name__)
BASE_PATH = Path("/home/ubuntu/mit-ai2-server")


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_img():
    file = request.data
    cur_id = str(np.random.randint(0, int(1e10))).zfill(10)

    with open(BASE_PATH / f"to_prc/{cur_id}.jpg", "wb") as f:
        f.write(file)
    result = predict_and_draw(
        BASE_PATH / "to_prc" / f"{cur_id}.jpg",
        BASE_PATH / "gen_img" / f"{cur_id}.jpg",
        f"/gen_img/{cur_id}.jpg",
    )

    pprint(result)
    return jsonify(result)


@app.route("/gen_img/<path:filepath>")
def gen_img(filepath):
    return send_from_directory(BASE_PATH / "gen_img", filepath)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
