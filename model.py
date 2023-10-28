import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import io
import torch
import matplotlib

yolo_model = torch.hub.load(
    "ultralytics/yolov5", "yolov5s"
)  # or yolov5n - yolov5x6, custom
cmap = matplotlib.cm.get_cmap("jet")


def predict(img_path):
    results = yolo_model(img_path)
    return results.pandas().xyxy[0], results.names


def predict_and_draw(img_path, out_img_path):
    img = Image.open(img_path)
    img_np = np.array(img)

    result, class_names = predict(img_path)
    result["color"] = result.apply(
        lambda x: cmap(x["class"] / len(class_names))[:3],
        axis=1,
    )

    for _, el in result.iterrows():
        p1, p2 = (el["xmin"], el["ymin"]), (el["xmax"], el["ymax"])
        p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))
        
        # Drawing rectangle using matplotlib
        plt.gca().add_patch(plt.Rectangle(p1, p2[0] - p1[0], p2[1] - p1[1], color=el["color"], linewidth=2, fill=False))

    plt.imshow(img_np)
    plt.axis('off')  # to remove axis numbers and ticks
    plt.savefig(out_img_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()

    result_dict = result.to_dict(orient="index")
    return {
        "boxes": [v for _, v in result_dict.items()],
    }