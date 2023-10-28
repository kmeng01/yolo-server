import base64
import time
from pathlib import Path
from pprint import pprint

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from model import predict_and_draw

# Create necessary directories
BASE_PATH = Path("data")
to_create = [str(BASE_PATH), "data/to_prc", "data/gen_img"]
for fpath in to_create:
    if not (p := Path(fpath)).exists():
        p.mkdir()

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict_img():
    file = request.data
    cur_id = (
        str(int(time.time())) + "_" + str(np.random.randint(0, int(1e10))).zfill(10)
    )

    with open(BASE_PATH / f"to_prc/{cur_id}.jpg", "wb") as f:
        f.write(base64.b64decode(file))
    result = predict_and_draw(
        BASE_PATH / "to_prc" / f"{cur_id}.jpg",
        BASE_PATH / "gen_img" / f"{cur_id}.jpg",
    ) | {"gen_img": f"gen_img/{cur_id}.jpg"}

    pprint(result)
    return jsonify(result)


@app.route("/gen_img/<path:filepath>")
def gen_img(filepath):
    return send_from_directory(BASE_PATH / "data" / "gen_img", filepath)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
