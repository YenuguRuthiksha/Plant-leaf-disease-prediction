# Importing Necessary Libraries
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

# Load the dataset
ruthahika_dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/plant/train.csv')

# Data Cleaning
ruthahika_dataset['image_id'] = ruthahika_dataset['image_id'] + '.jpg'

# Check for null values and data types
print(ruthahika_dataset.isnull().any())
print(ruthahika_dataset.dtypes)

# Data Visualization - Class Distribution
ruthahika_dataset.healthy.hist()
plt.title('Healthy Classes')
plt.show()

ruthahika_dataset.multiple_diseases.hist()
plt.title('Multiple Diseases Classes')
plt.show()

ruthahika_dataset.rust.hist()
plt.title('Rust Classes')
plt.show()

ruthahika_dataset.scab.hist()
plt.title('Scab Classes')
plt.show()

# Display Sample Images
def ruthahika_visualize_samples(dataset, image_dir):
    fig = plt.figure(figsize=(20, 14))
    columns, rows = 4, 4
    for i in range(1, columns * rows + 1):
        img = plt.imread(f'{image_dir}/Train_{i}.jpg')
        fig.add_subplot(rows, columns, i)
        if dataset.healthy[i] == 1:
            plt.title('Healthy')
        elif dataset.multiple_diseases[i] == 1:
            plt.title('Multiple Diseases')
        elif dataset.rust[i] == 1:
            plt.title('Rust')
        else:
            plt.title('Scab')
        plt.imshow(img)
        plt.axis('off')
    plt.show()

ruthahika_visualize_samples(ruthahika_dataset, '/content/drive/MyDrive/Colab Notebooks/plant/images')

# Image Augmentation
ruthahika_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=180,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    vertical_flip=True
)

# Splitting the Data
ruthahika_train_data, ruthahika_valid_data = train_test_split(ruthahika_dataset, test_size=0.05, shuffle=False)

# Create Data Generators
ruthahika_train_generator = ruthahika_datagen.flow_from_dataframe(
    ruthahika_train_data,
    directory='/content/drive/MyDrive/Colab Notebooks/plant/images/',
    x_col='image_id',
    y_col=['healthy', 'multiple_diseases', 'rust', 'scab'],
    target_size=(512, 512),
    class_mode='raw',
    batch_size=8,
    shuffle=False
)

ruthahika_valid_generator = ruthahika_datagen.flow_from_dataframe(
    ruthahika_valid_data,
    directory='/content/drive/MyDrive/Colab Notebooks/plant/images/',
    x_col='image_id',
    y_col=['healthy', 'multiple_diseases', 'rust', 'scab'],
    target_size=(512, 512),
    class_mode='raw',
    batch_size=8,
    shuffle=False
)

# Model Definition - Xception
ruthahika_xception_model = tf.keras.models.Sequential([
    tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation='softmax')
])

ruthahika_xception_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Definition - DenseNet
ruthahika_densenet_model = tf.keras.models.Sequential([
    tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation='softmax')
])

ruthahika_densenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Ensembling the Models
inputs = tf.keras.Input(shape=(512, 512, 3))
xception_output = ruthahika_xception_model(inputs)
densenet_output = ruthahika_densenet_model(inputs)
outputs = tf.keras.layers.average([xception_output, densenet_output])

ruthahika_ensemble_model = tf.keras.Model(inputs=inputs, outputs=outputs)
ruthahika_ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Learning Rate Scheduler
def ruthahika_lrfn(epoch):
    LR_START, LR_MAX, LR_MIN = 0.00001, 0.0001, 0.00001
    LR_RAMPUP_EPOCHS, LR_SUSTAIN_EPOCHS, LR_EXP_DECAY = 15, 3, 0.8
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

ruthahika_lr_callback = tf.keras.callbacks.LearningRateScheduler(ruthahika_lrfn, verbose=True)

# Model Training
ruthahika_model_history = ruthahika_ensemble_model.fit(
    ruthahika_train_generator,
    epochs=50,
    validation_data=ruthahika_valid_generator,
    callbacks=[ruthahika_lr_callback]
)

# Saving the Model and History
ruthahika_ensemble_model.save('ruthahika_ensemble_model.h5')
pd.DataFrame(ruthahika_model_history.history).to_csv('ruthahika_model_history.csv')

# Visualizing Training History
plt.plot(pd.DataFrame(ruthahika_model_history.history)['accuracy'], label='Train Accuracy')
plt.plot(pd.DataFrame(ruthahika_model_history.history)['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()
