import os
import random
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from ultralytics import YOLO

dataset_dir = '/content/dataset'
images_dir = '/content/images'
labels_dir = '/content/labels'
video_path = '/content/test_video.mp4'
weights_path = '/content/yolo11n.pt'

for split in ['train', 'val']:
    os.makedirs(f'{dataset_dir}/{split}/images', exist_ok=True)
    os.makedirs(f'{dataset_dir}/{split}/labels', exist_ok=True)

yaml_path = os.path.join(dataset_dir, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(f"""\
path: {dataset_dir}
train: train/images
val: val/images
nc: 1
names: ['dog']
""")

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=25, p=0.5),
    A.RandomGamma(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.Blur(blur_limit=3, p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]

total_augmented = 1000
split_ratio = 0.8
i = 0
pbar = tqdm(total=total_augmented, desc='Generating augmented images')

while i < total_augmented:
    file_name = random.choice(image_files)
    base_name = os.path.splitext(file_name)[0]

    label_file = f'{base_name}.txt'
    bboxes, class_labels = [], []
    with open(os.path.join(labels_dir, label_file), 'r') as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            bboxes.append([x, y, w, h])
            class_labels.append(int(cls))

    img = cv2.imread(os.path.join(images_dir, file_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
    aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']

    split = 'train' if i < total_augmented * split_ratio else 'val'

    out_img_path = f'{dataset_dir}/{split}/images/{base_name}_aug{i}.jpg'
    out_label_path = f'{dataset_dir}/{split}/labels/{base_name}_aug{i}.txt'

    cv2.imwrite(out_img_path, aug_img)
    with open(out_label_path, 'w') as f:
        for cls, bbox in zip(aug_labels, aug_bboxes):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")

    i += 1
    pbar.update(1)

pbar.close()
print("Augmented images generated")

model = YOLO(weights_path)

model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=32,
    name='dog_detector_yolo11'
)

class_names = ['dog']

cap = cv2.VideoCapture(video_path)
out_video_path = '/content/test_video_out.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for r in results:
        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cls_idx = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i])
            label = f"{class_names[cls_idx]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
print(f"Processed video saved as: {out_video_path}")
