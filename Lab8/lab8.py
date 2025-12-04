import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer

data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
data_path = keras.utils.get_file(
    fname="LJSpeech-1.1",
    origin=data_url,
    untar=True,
    cache_dir="/content",
    cache_subdir=""
)
data_path = os.path.join(data_path, "LJSpeech-1.1")
wavs_path = os.path.join(data_path, "wavs")
metadata_path = os.path.join(data_path, "metadata.csv")

metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]
metadata_df = metadata_df[:3000].sample(frac=1).reset_index(drop=True)

split_train = int(len(metadata_df) * 0.8)
split_val = int(len(metadata_df) * 0.1)
split_test = len(metadata_df) - split_train - split_val

df_train = metadata_df[:split_train]
df_val = metadata_df[split_train:split_train+split_val]
df_test = metadata_df[split_train+split_val:]

print(f"Size of the training set: {len(df_train)}")
print(f"Size of the validation set: {len(df_val)}")
print(f"Size of the test set: {len(df_test)}")

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)
print(f"The vocabulary is: {char_to_num.get_vocabulary()} (size ={char_to_num.vocabulary_size()})")

frame_length = 256
frame_step = 160
fft_length = 384

def encode_single_sample(wav_file, label):
    file = tf.io.read_file(tf.strings.join([wavs_path, "/", wav_file, ".wav"]))
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = char_to_num(label)
    return spectrogram, label

batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((list(df_train["file_name"]), list(df_train["normalized_transcription"])))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size, padded_shapes=([None, fft_length // 2 + 1], [None]), padding_values=(0.0, tf.cast(0, tf.int64)))
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((list(df_val["file_name"]), list(df_val["normalized_transcription"])))
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size, padded_shapes=([None, fft_length // 2 + 1], [None]), padding_values=(0.0, tf.cast(0, tf.int64)))
    .prefetch(tf.data.AUTOTUNE)
)

def show_audio_prediction(spectrogram, waveform, label=None, prediction=None):
    spectrogram = np.array(spectrogram)
    if spectrogram.ndim != 2:
        max_len = max(len(row) for row in spectrogram)
        spectrogram = np.array([np.pad(row, (0, max_len - len(row))) for row in spectrogram])

    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(2, 1, 1)
    ax.imshow(np.transpose(spectrogram), vmax=1, aspect='auto')
    title = ""
    if label is not None:
        title += f"Target: {label}\n"
    if prediction is not None:
        title += f"Prediction: {prediction}"
    ax.set_title(title.strip())
    ax.axis("off")

    ax = plt.subplot(2, 1, 2)
    plt.plot(waveform)
    ax.set_title("Signal Wave")
    ax.set_xlim(0, len(waveform))
    display.display(display.Audio(np.transpose(waveform), rate=16000))
    plt.show()

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones((batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones((batch_len, 1), dtype="int64")
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    input_spectrogram = layers.Input((None, input_dim), name="input")
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    x = layers.Conv2D(32, [11, 41], strides=[2, 2], padding="same", use_bias=False, name="conv_1")(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    x = layers.Conv2D(32, [11, 21], strides=[1, 2], padding="same", use_bias=False, name="conv_2")(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(rnn_units, activation="tanh", recurrent_activation="sigmoid",
                               use_bias=True, return_sequences=True, reset_after=True, name=f"gru_{i}")
        x = layers.Bidirectional(recurrent, name=f"bidirectional_{i}", merge_mode="concat")(x)
        if i < rnn_layers:
            x = layers.Dropout(0.5)(x)
    x = layers.Dense(rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(output_dim + 2, activation="softmax")(x)
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=CTCLoss)
    return model

model = build_model(input_dim=fft_length // 2 + 1, output_dim=char_to_num.vocabulary_size() - 1, rnn_units=512)
model.summary(line_length=110)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = [tf.strings.reduce_join(num_to_char(r)).numpy().decode("utf-8") for r in results]
    return output_text

class CallbackEval(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.last_epoch_wer = None

    def on_epoch_end(self, epoch, logs=None):
        predictions, targets = [], []
        for batch in self.dataset.take(1):
            X, y = batch
            batch_predictions = decode_batch_predictions(self.model.predict(X))
            predictions.extend(batch_predictions)
            for label in y:
                targets.append(tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8"))
        wer_score = wer(targets, predictions) if targets else float("nan")
        self.last_epoch_wer = wer_score
        print("-"*100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-"*100)
        for i in range(min(2, len(predictions))):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-"*100)

epochs = 15
validation_callback = CallbackEval(validation_dataset)
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[validation_callback])

print("-"*100)
print(f"Final Word Error Rate: {validation_callback.last_epoch_wer:.4f}")
print("-"*100)

print("-"*100)
print("Testing on random samples from test dataset")
print("-"*100)
test_samples = df_test.sample(3)
for idx, row in test_samples.iterrows():
    file_path = os.path.join(wavs_path, row["file_name"] + ".wav")
    audio_file = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_file)
    audio = tf.squeeze(audio, axis=-1)
    spectrogram, label_encoded = encode_single_sample(row["file_name"], row["normalized_transcription"])
    prediction = decode_batch_predictions(model.predict(tf.expand_dims(spectrogram, 0)))[0]
    show_audio_prediction(spectrogram.numpy(), audio.numpy(), label=row["normalized_transcription"], prediction=prediction)

print("-"*100)
print("Testing on random samples not from test dataset")
print("-"*100)

my_audio_folder = "/content/my_audios_wav"
my_audio_files = [f for f in os.listdir(my_audio_folder) if f.endswith(".wav")][:3]

target_len = model.input_shape[1]

for f in my_audio_files:
    file_path = os.path.join(my_audio_folder, f)
    audio_file = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_file)
    audio = tf.squeeze(audio, axis=-1)

    spectrogram = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    spectrogram = (spectrogram - tf.reduce_mean(spectrogram, axis=1, keepdims=True)) / \
                  (tf.math.reduce_std(spectrogram, axis=1, keepdims=True) + 1e-10)

    current_len = spectrogram.shape[1]
    if target_len is None:
        target_len = current_len

    if current_len < target_len:
        pad_amount = target_len - current_len
        spectrogram = tf.pad(spectrogram, paddings=[[0,0],[0,pad_amount]], mode="CONSTANT")
    else:
        spectrogram = spectrogram[:, :target_len]

    prediction = decode_batch_predictions(model.predict(tf.expand_dims(spectrogram, 0)))[0]

    show_audio_prediction(spectrogram.numpy(), audio.numpy(), prediction=prediction)
