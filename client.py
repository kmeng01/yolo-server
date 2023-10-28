import base64

import requests

with open("assets/burger.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

response = requests.post("http://127.0.0.1:5001/predict", data=encoded_string)
