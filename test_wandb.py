import os.path

import cv2
import numpy as np
from matplotlib import pyplot as plt

import wandb

img_path = r"C:\Users\tristan_cotte\PycharmProjects\coco2yolo_obb\Dataset_solo\train\images\train_1.png"
ann_path = r"C:\Users\tristan_cotte\PycharmProjects\coco2yolo_obb\Dataset_solo\train\labelTxt\train_1.txt"

img_dir = r"C:\Users\tristan_cotte\PycharmProjects\coco2yolo_obb\Dataset_solo\train\images"
ann_dir = r"C:\Users\tristan_cotte\PycharmProjects\coco2yolo_obb\Dataset_solo\train\labelTxt"

if __name__ == "__main__":

    def get_mask(ann_path, img_path, classes):
        with open(ann_path) as f:
            lines = f.readlines()

        img = cv2.imread(img_path)

        mask = np.full(shape=(img.shape[:2]), fill_value=len(classes) + 1, dtype=np.int32)

        for annotation in lines:
            pts = np.array(annotation.split(" ")[:8]).astype(float)
            pts = np.array([(pts[0], pts[1]), (pts[2], pts[3]), (pts[4], pts[5]), (pts[6], pts[7])], np.int32).reshape(
                -1, 1, 2)
            mask = cv2.fillPoly(mask, [pts], 0)

        return mask


    wandb.init()

    class_labels = {
        0: "ct"
    }

    table = wandb.Table(columns=['ID', 'Image'])

    for id, img_name, label_name in zip(list(range(0, 2)), os.listdir(img_dir)[:2], os.listdir(ann_dir)[:2]):
        label = os.path.join(ann_dir, label_name)
        img_path = os.path.join(img_dir, img_name)
        print(label, img_path)
        img = cv2.imread(img_path)

        mask_img = wandb.Image(img, masks={
            "prediction": {
                "mask_data": get_mask(ann_path=label, img_path=img_path, classes=class_labels),
                "class_labels": class_labels
            }
        })

        table.add_data(id, mask_img)

    wandb.log({"Table": table})

