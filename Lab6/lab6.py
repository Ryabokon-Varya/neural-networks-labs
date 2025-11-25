import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

logo_dir = '/content/logos'
background_dir = '/content/data'
output_dir = '/content/dataset'

splits = ['train', 'val', 'test']
classes = ['positive', 'negative']

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

def augment_logo(logo):
    angle = random.uniform(-30, 30)
    logo = logo.rotate(angle, expand=True)
    scale = random.uniform(0.5, 1.5)
    w, h = logo.size
    logo = logo.resize((int(w * scale), int(h * scale)))
    alpha = random.uniform(0.5, 1.0)
    logo.putalpha(int(255 * alpha))
    return logo

logo_files = [os.path.join(logo_dir, f) for f in os.listdir(logo_dir)
              if f.endswith(('.png', '.jpg', '.jpeg'))]

background_files = [os.path.join(background_dir, f) for f in os.listdir(background_dir)
                    if f.endswith(('.png', '.jpg', '.jpeg'))]

total_positive = 1000
total_negative = 1000
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

def get_split_name(idx, total):
    val_thresh = int(total * split_ratios['train'])
    test_thresh = int(total * (split_ratios['train'] + split_ratios['val']))
    if idx < val_thresh:
        return 'train'
    elif idx < test_thresh:
        return 'val'
    else:
        return 'test'

i = 0
pbar = tqdm(total=total_positive, desc='Generating positive samples')
while i < total_positive:
    bg_path = random.choice(background_files)
    logo_path = random.choice(logo_files)
    try:
        bg = Image.open(bg_path).convert('RGB')
        logo = Image.open(logo_path).convert('RGBA')
    except:
        continue
    logo = augment_logo(logo)
    if logo.width > bg.width or logo.height > bg.height:
        logo.thumbnail((bg.width, bg.height))
    max_x = bg.width - logo.width
    max_y = bg.height - logo.height
    if max_x < 0 or max_y < 0:
        continue
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    bg.paste(logo, (x, y), logo)
    split = get_split_name(i, total_positive)
    bg.save(os.path.join(output_dir, split, 'positive', f'pos_{i}.jpg'))
    i += 1
    pbar.update(1)
pbar.close()

i = 0
pbar = tqdm(total=total_negative, desc='Generating negative samples')
while i < total_negative:
    bg_path = random.choice(background_files)
    try:
        bg = Image.open(bg_path).convert('RGB')
    except:
        continue
    split = get_split_name(i, total_negative)
    bg.save(os.path.join(output_dir, split, 'negative', f'neg_{i}.jpg'))
    i += 1
    pbar.update(1)
pbar.close()

dataset_dir = '/content/dataset'
img_size = (150, 150)
batch_size = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary')

val_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, 'val'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    os.path.join(dataset_dir, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

def build_xception(input_shape=(150, 150, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    skip = layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same')(x)
    skip = layers.BatchNormalization()(skip)

    x = layers.SeparableConv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = layers.add([x, skip])

    skip = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
    skip = layers.BatchNormalization()(skip)

    x = layers.SeparableConv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = layers.add([x, skip])

    x = layers.SeparableConv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(num_classes)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(0.0005), 
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
        metrics=['accuracy']
    )
    return model

model = build_xception()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
history = model.fit(train_generator, validation_data=val_generator, epochs=30, callbacks=[early_stop])

test_logits = model.predict(test_generator)
test_probs = tf.nn.sigmoid(test_logits).numpy().flatten()
test_preds = (test_probs > 0.5).astype(int)
test_labels = test_generator.labels

print("\nClassification Report:\n")
print(classification_report(test_labels, test_preds, target_names=classes))

conf_mat = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

num_samples = 5
class_names = list(train_generator.class_indices.keys())
num_test_samples = len(test_generator.labels)

random_indices = random.sample(range(num_test_samples), num_samples)

plt.figure(figsize=(15, 10))

for i, index in enumerate(random_indices):
    image = test_generator.filepaths[index]
    true_label = test_generator.labels[index]
    img = tf.keras.utils.load_img(image, target_size=model.input_shape[1:3])
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    probabilities = tf.nn.sigmoid(predictions).numpy().flatten()
    predicted_class = 1 if probabilities[0] > 0.5 else 0
    predicted_label = class_names[predicted_class]
    true_label_name = class_names[true_label]
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(img)
    plt.title(f"True: {true_label_name}\nPred: {predicted_label} ({probabilities[0]:.2f})")
    plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.hist(test_probs, bins=20, alpha=0.7)
plt.title('Test Predictions Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

video_path = '/content/toyota_video.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
batch_size = 32
frames, frames_ids, frame_predictions, frame_numbers = [], [], [], []
frame_count = 0

def process_batch(batch_frames, batch_ids):
    batch_np = np.array(batch_frames, dtype=np.float32)/255.0
    predictions = model.predict(batch_np)
    for i in range(len(predictions)):
        score = 1/(1+np.exp(-predictions[i][0]))
        frame_predictions.append(score)
        frame_numbers.append(batch_ids[i])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (150, 150))
    frames.append(resized_frame)
    frames_ids.append(frame_count)
    if len(frames) == batch_size:
        process_batch(frames, frames_ids)
        frames, frames_ids = [], []
    frame_count += 1
if frames:
    process_batch(frames, frames_ids)
cap.release()

time_seconds = [f/fps for f in frame_numbers]
plt.figure(figsize=(12,6))
plt.plot(time_seconds, frame_predictions, label='Confidence', color='blue')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.xlabel('Time (s)')
plt.ylabel('Prediction Score')
plt.title('Model Predictions Over Video Time')
plt.legend()
plt.grid(True)
plt.show()

threshold = 0.5
intervals = []
start = None

for t, score in zip(time_seconds, frame_predictions):
    if score > threshold and start is None:
        start = t
    elif score <= threshold and start is not None:
        end = t
        intervals.append((start, end))
        start = None

if start is not None:
    intervals.append((start, time_seconds[-1]))

print("Logo appearance intervals (seconds):")
for start, end in intervals:
    print(f"{start:.2f} s - {end:.2f} s")
