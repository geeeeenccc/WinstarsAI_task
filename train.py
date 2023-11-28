"""
In this code we're gonna train more advanced u-net architecture
"""

# train.py

import os
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import tensorflow_addons as tfa
from model import UNetModel
from utils import load_train_image, rle_to_mask, one_hot

# Read Data Set
segmentations = pd.read_csv("/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv")
segmentations['EncodedPixels'] = segmentations['EncodedPixels'].astype('string')

def get_train_image(name: str):
    path = f'/kaggle/input/airbus-ship-detection/train_v2/{name}'
    return cv2.imread(path)

def extract_features_from_image(row: pd.Series) -> pd.Series:
    image = np.zeros((768, 768, 3))  # get_train_image(row['ImageId'])
    row['ImageHeight'], row['ImageWidth'], _ = image.shape
    return row

segmentations = segmentations.apply(lambda x: extract_features_from_image(x), axis=1)

# Train Unet Model
RANDOM_SEED = 77
random.seed(RANDOM_SEED)

TRAIN_DIR = '/kaggle/input/airbus-ship-detection/train_v2/'
TEST_DIR = '/kaggle/input/airbus-ship-detection/test_v2/'

df = pd.read_csv("/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv")
df['EncodedPixels'] = df['EncodedPixels'].astype('string')

# Delete corrupted images
CORRUPTED_IMAGES = ['6384c3e78.jpg']
df = df.drop(df[df['ImageId'].isin(CORRUPTED_IMAGES)].index)

# Dataframe that contains the segmentation for each ship in the image.
instance_segmentation = df

# Dataframe that contains the segmentation of all ships in the image.
image_segmentation = df.groupby(by=['ImageId'])['EncodedPixels'].apply(lambda x: np.nan if pd.isna(x).any() else ' '.join(x)).reset_index()

# Utils
def rle_to_mask(rle: str, shape=(768, 768)):
    encoded_pixels = np.array(rle.split(), dtype=int)
    starts = encoded_pixels[::2] - 1
    ends = starts + encoded_pixels[1::2]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask_to_rle(img, shape=(768, 768)) -> str:
    img = img.astype('float32')
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    img = np.stack(np.vectorize(lambda x: 0 if x < 0.1 else 1)(img), axis=1)
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Data preprocessing
IMAGES_WITHOUT_SHIPS_NUMBER = 25000

images_without_ships = image_segmentation[image_segmentation['EncodedPixels'].isna()]['ImageId'].values[:IMAGES_WITHOUT_SHIPS_NUMBER]
images_with_ships = image_segmentation[image_segmentation['EncodedPixels'].notna()]['ImageId'].values
images_list = np.append(images_without_ships, images_with_ships)

images_list = np.array(list(filter(lambda x: x not in CORRUPTED_IMAGES, images_list)))

VALIDATION_LENGTH = 2000
TEST_LENGTH = 2000
TRAIN_LENGTH = len(images_list) - VALIDATION_LENGTH - TEST_LENGTH
BATCH_SIZE = 16
BUFFER_SIZE = 1000
IMG_SHAPE = (256, 256)
NUM_CLASSES = 2

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])

def load_train_image(tensor) -> tuple:
    path = tf.get_static_value(tensor).decode("utf-8")

    image_id = path.split('/')[-1]
    input_image = cv2.imread(path)
    input_image = tf.image.resize(input_image, IMG_SHAPE)
    input_image = tf.cast(input_image, tf.float32) / 255.0

    encoded_mask = image_segmentation[image_segmentation['ImageId'] == image_id].iloc[0]['EncodedPixels']
    input_mask = np.zeros(IMG_SHAPE + (1,), dtype=np.int8)
    if not pd.isna(encoded_mask):
        input_mask = rle_to_mask(encoded_mask)
        input_mask = cv2.resize(input_mask, IMG_SHAPE, interpolation=cv2.INTER_AREA)
        input_mask = np.expand_dims(input_mask, axis=2)
    one_hot_segmentation_mask = one_hot(input_mask, NUM_CLASSES)
    input_mask_tensor = tf.convert_to_tensor(one_hot_segmentation_mask, dtype=tf.float32)
    
    class_weights = tf.constant([0.0005, 0.9995], tf.float32)
    sample_weights = tf.gather(class_weights, indices=tf.cast(input_mask_tensor, tf.int32), name='cast_sample_weights')

    return input_image, input_mask_tensor, sample_weights

images_list = tf.data.Dataset.list_files([f'{TRAIN_DIR}{name}' for name in images_list], shuffle=True)
train_images = images_list.map(lambda x: tf.py_function(load_train_image, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)

validation_dataset = train_images.take(VALIDATION_LENGTH)
test_dataset = train_images.skip(VALIDATION_LENGTH).take(TEST_LENGTH)
train_dataset = train_images.skip(VALIDATION_LENGTH + TEST_LENGTH)

train_batches = (
    train_dataset
    .repeat()
    .batch(BATCH_SIZE))

validation_batches = validation_dataset.batch(BATCH_SIZE)
test_batches = test_dataset.batch(BATCH_SIZE)

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


EPOCHS = 4
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

optimizer = tfa.optimizers.RectifiedAdam(
    learning_rate=0.005,
    total_steps=EPOCHS * STEPS_PER_EPOCH,
    warmup_proportion=0.3,
    min_lr=0.00001,
)
optimizer = tfa.optimizers.Lookahead(optimizer)

loss = tf.keras.losses.CategoricalCrossentropy()

model = UNetModel(IMG_SHAPE + (3,)).model
model.compile(optimizer=optimizer, 
              loss=loss,
              metrics=[dice_coef],
)

trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
print(f'Trainable params: {trainable_params}')

checkpoint_filepath = 'checkpoints/model-checkpoint'
save_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_dice_coef',
    mode='max',
    save_best_only=True
)

model_history = model.fit(train_batches,
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=validation_batches,
                          callbacks=[save_callback])

model.load_weights(checkpoint_filepath)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'C2', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.show()

dice_coef_values = model_history.history['dice_coef']
val_dice_coef_values = model_history.history['val_dice_coef']

plt.figure()
plt.plot(model_history.epoch, dice_coef_values, 'm', label='Training Dice Coef')
plt.plot(model_history.epoch, val_dice_coef_values, 'y', label='Validation Dice Coef')

plt.title('Training and Validation Dice Coefficients')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient Value')
plt.legend()
plt.show()
