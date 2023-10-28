import base64

import requests

with open("assets/burger.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

response = requests.post(
    "https://kmeng01--main-py-flask-app-dev.modal.run/predict", data=encoded_string
)
print(response.json())
