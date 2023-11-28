"""

"""

# inference.py

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

def predict(image):
    image = np.expand_dims(image, axis=0)
    pred_mask = model.predict(image)[0].argmax(axis=-1)  
    return pred_mask

def mask_to_rle(img, shape=(768, 768)) -> str:
    img = img.astype('float32')
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    img = np.stack(np.vectorize(lambda x: 0 if x < 0.1 else 1)(img), axis=1)
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# UNet model
class UNetModel:
    def __init__(self, input_shape=(128, 128, 3), num_classes=NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self._model = self._build_model()

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    def _conv_block(self, x, filters, size, apply_batch_norm=False, apply_instance_norm=False, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, size, strides=1, padding='same', use_bias=False, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization() if apply_batch_norm else tf.keras.layers.Lambda(lambda x: x),
            tfa.layers.InstanceNormalization() if apply_instance_norm else tf.keras.layers.Lambda(lambda x: x),
            tf.keras.layers.Activation(tfa.activations.mish),
            tf.keras.layers.Dropout(0.55) if apply_dropout else tf.keras.layers.Lambda(lambda x: x),
        ])
        return result(x)

    def _upsample_block(self, x, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=False, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1) if apply_dropout else tf.keras.layers.Lambda(lambda x: x),
            tf.keras.layers.Activation(tfa.activations.mish),
        ])
        return result(x)

    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = inputs
        filters_list = [16, 32, 64]

        # Encoder
        encoder_outputs = []
        for i, filters in enumerate(filters_list):
            x = self._conv_block(x, filters, size=3, apply_batch_norm=True, apply_instance_norm=True)
            print(f"Encoder Block {i+1} Output Shape: {x.shape}")
            x = self._conv_block(x, filters, size=1, apply_batch_norm=True, apply_instance_norm=True)
            encoder_outputs.append(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = self._conv_block(x, filters=128, size=3, apply_batch_norm=True)
        print(f"Encoder Block {len(filters_list)+1} Output Shape: {x.shape}")
        encoder_outputs.append(x)

        # Decoder
        x = encoder_outputs[-1]
        for i, (filters, skip) in enumerate(zip(filters_list[::-1], encoder_outputs[-2::-1])):
            x = self._upsample_block(x, filters, 3)
            print(f"Decoder Upsample Block {i+1} Output Shape: {x.shape}")
            x = tf.keras.layers.Concatenate()([x, skip])
            print(f"Decoder Concatenate Block {i+1} Output Shape: {x.shape}")
            x = self._conv_block(x, filters, size=3, apply_batch_norm=True)
            print(f"Decoder Conv Block {i+1} Output Shape: {x.shape}")
            x = self._conv_block(x, filters, size=1, apply_batch_norm=True)
            print(f"Decoder Conv Block {i+1} Output Shape: {x.shape}")

        # Output layer
        last = self._conv_block(x, filters=self.num_classes, size=1)
        print(f"Output Block Output Shape: {last.shape}")
        outputs = tf.keras.layers.Activation('softmax')(last)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

def load_test_image(tensor) -> np.ndarray:
    path = tf.get_static_value(tensor).decode("utf-8")
    input_image = cv2.imread(path)
    input_image = cv2.resize(input_image, IMG_SHAPE, interpolation=cv2.INTER_AREA)
    input_image = input_image / 255.0
    return input_image

# Load your trained model
IMG_SHAPE = (256, 256)  # Adjust this to match the input shape of your model
TEST_DIR = '/kaggle/input/airbus-ship-detection/test_v2/'
model = UNetModel(IMG_SHAPE + (3,)).model
model.load_weights('checkpoints/model-checkpoint')

# Load test data or submission file
submission = pd.read_csv("/kaggle/input/airbus-ship-detection/sample_submission_v2.csv")

def set_model_prediction(row: pd.Series) -> pd.Series:
    image = load_test_image(f'{TEST_DIR}{row["ImageId"]}')
    pred_mask = predict(image)
    row['EncodedPixels'] = mask_to_rle(pred_mask)
    if row['EncodedPixels'] == '':
        row['EncodedPixels'] = np.nan
    return row

# Apply model predictions
submission = submission.apply(lambda x: set_model_prediction(x), axis=1).set_index("ImageId")

# Save submission file
submission.to_csv("submission.csv")
print(submission)
