import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cat_dir = '/content/cat'
dog_dir = '/content/dog'
output_dir = '/content/dataset'

splits = ['train', 'val', 'test']
classes = ['cat', 'dog']

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

def augment_image(img):
    img = img.rotate(random.uniform(-30, 30), expand=True)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    scale = random.uniform(0.8, 1.2)
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)))
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    return img

cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
dog_files = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

total_cat = 1000
total_dog = 1000
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

def generate_augmented_images(files, class_name, total_images):
    i = 0
    pbar = tqdm(total=total_images, desc=f'Generating {class_name} samples')
    while i < total_images:
        file_path = random.choice(files)
        try:
            img = Image.open(file_path).convert('RGB')
        except:
            continue
        img = augment_image(img)
        split = get_split_name(i, total_images)
        img.save(os.path.join(output_dir, split, class_name, f'{class_name}_{i}.jpg'))
        i += 1
        pbar.update(1)
    pbar.close()

generate_augmented_images(cat_files, 'cat', total_cat)
generate_augmented_images(dog_files, 'dog', total_dog)

img_size = (150, 150)
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(output_dir, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    os.path.join(output_dir, 'val'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    os.path.join(output_dir, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

def build_inception(input_shape=(150,150,3), num_classes=1):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='valid', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
    x = tf.keras.layers.Conv2D(80, 1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2D(192, 3, padding='valid', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
    
    def inception_block_a(x):
        b1 = tf.keras.layers.Conv2D(64, 1, activation='relu')(x)
        b2 = tf.keras.layers.Conv2D(48, 1, activation='relu')(x)
        b2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(b2)
        b3 = tf.keras.layers.Conv2D(64, 1, activation='relu')(x)
        b3 = tf.keras.layers.Conv2D(96, 3, padding='same', activation='relu')(b3)
        b3 = tf.keras.layers.Conv2D(96, 3, padding='same', activation='relu')(b3)
        b4 = tf.keras.layers.AveragePooling2D(3, strides=1, padding='same')(x)
        b4 = tf.keras.layers.Conv2D(32, 1, activation='relu')(b4)
        return tf.keras.layers.concatenate([b1, b2, b3, b4], axis=-1)
    
    x = inception_block_a(x)
    x = inception_block_a(x)
    
    def inception_block_b(x):
        b1 = tf.keras.layers.Conv2D(192, 1, activation='relu')(x)
        b2 = tf.keras.layers.Conv2D(128, 1, activation='relu')(x)
        b2 = tf.keras.layers.Conv2D(128, (1,7), padding='same', activation='relu')(b2)
        b2 = tf.keras.layers.Conv2D(192, (7,1), padding='same', activation='relu')(b2)
        b3 = tf.keras.layers.Conv2D(128, 1, activation='relu')(x)
        b3 = tf.keras.layers.Conv2D(128, (7,1), padding='same', activation='relu')(b3)
        b3 = tf.keras.layers.Conv2D(128, (1,7), padding='same', activation='relu')(b3)
        b3 = tf.keras.layers.Conv2D(192, (7,1), padding='same', activation='relu')(b3)
        b4 = tf.keras.layers.MaxPooling2D(3, strides=1, padding='same')(x)
        b4 = tf.keras.layers.Conv2D(192, 1, activation='relu')(b4)
        return tf.keras.layers.concatenate([b1, b2, b3, b4], axis=-1)
    
    x = inception_block_b(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

model = build_inception()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
history = model.fit(train_generator, validation_data=val_generator, epochs=30, callbacks=[early_stop])

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

test_images_dir = '/content/test_images'
test_files = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

plt.figure(figsize=(15,5))
for i, file_path in enumerate(test_files[:10]):
    try:
        img = Image.open(file_path).convert('RGB')
    except:
        continue
    img_resized = img.resize(img_size)
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)
    pred_logits = model.predict(img_array)
    prob = tf.nn.sigmoid(pred_logits).numpy()[0][0]
    label = 'dog' if prob > 0.5 else 'cat'

    plt.subplot(1, min(10, len(test_files)), i+1)
    plt.imshow(img)
    plt.title(f"{label}\n({prob:.2f})")
    plt.axis('off')
plt.tight_layout()
plt.show()
