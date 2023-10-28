import cv2
import matplotlib
import PIL.ImageOps
import torch
from PIL import Image
from skimage import io

yolo_model = torch.hub.load(
    "ultralytics/yolov5", "yolov5s"
)  # or yolov5n - yolov5x6, custom
cmap = matplotlib.cm.get_cmap("jet")


def predict(img_path):
    results = yolo_model(img_path)
    return results.pandas().xyxy[0], results.names


def predict_and_draw(img_path, out_img_path):
    img = io.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result, class_names = predict(img_path)
    result["color"] = result.apply(
        lambda x: tuple(c * 255 for c in cmap(x["class"] / len(class_names)))[:3],
        axis=1,
    )

    for _, el in result.iterrows():
        p1, p2 = (el["xmin"], el["ymin"]), (el["xmax"], el["ymax"])
        p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
        cv2.rectangle(img, p1, p2, el["color"], 2)

    cv2.imwrite(str(out_img_path), img)

    result_dict = result.to_dict(orient="index")
    return {
        "boxes": [v for _, v in result_dict.items()],
    }
