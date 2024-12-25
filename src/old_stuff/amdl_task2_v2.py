from matplotlib import pyplot as plt
from tensorflow.keras import datasets as tfd  # type: ignore
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
import json
from tensorflow.keras.optimizers import SGD, RMSprop, Adam  # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau  # type: ignore
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def load_data_all():
    train_data, test_data = tfd.cifar100.load_data(label_mode="coarse")

    (x_tr_color, y_tr), (x_test_color, y_test) = train_data, test_data

    # Normalize the data
    x_tr_gy = np.mean(x_tr_color, axis=3) / 255.0
    x_test_gy = np.mean(x_test_color, axis=3) / 255.0
    print(y_tr.shape)
    print(y_test.shape)
    return x_tr_color, x_test_color, x_tr_gy, x_test_gy, y_tr, y_test


def load_data_perturbed():
    with open('data/cifar20_perturb_test.pkl', 'rb') as file:
        data_dict = pickle.load(file)
    x_perturb_color, y_perturb = data_dict['x_perturb'], data_dict['y_perturb']
    x_perturb_gy = np.mean(x_perturb_color, axis=3)  # Grayscale version
    return x_perturb_color, x_perturb_gy, y_perturb


def display_image(x_data, x_data2, y_data, index=0):
    image = x_data[index].astype(np.uint8)
    grayscale_image = x_data2[index].astype(np.uint8)
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image: {y_data[index]}")
    axes[0].axis("off")
    axes[1].imshow(np.repeat(grayscale_image[..., np.newaxis], 3, axis=-1))
    axes[1].set_title(f"Grayscale Image: {y_data[index]}")
    axes[1].axis("off")
    plt.show()


def validation_set(data, target, t_size):
    # create a stratified validation set.
    data_train, data_val, target_train, target_val = train_test_split(
        data, target, test_size=t_size, random_state=1, stratify=target
    )
    return data_train, data_val, target_train, target_val


def m1(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def m2(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (16, 16, 32)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (8, 8, 64)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (4, 4, 128)
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def m3(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def m4(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (2, 2), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (16, 16, 32)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (8, 8, 64)
        layers.Conv2D(128, (4, 4), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (4, 4, 128)
        layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def m5(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (5, 5), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (16, 16, 32)
        layers.Conv2D(64, (4, 4), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (8, 8, 64)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (4, 4, 128)
        layers.Conv2D(256, (2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def m6(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (2, 2), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (16, 16, 32)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (8, 8, 64)
        layers.Conv2D(128, (4, 4), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (4, 4, 128)
        layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Dropout in dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Dropout in dense layers
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def m7(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (2, 2), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (16, 16, 32)
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (8, 8, 64)
        layers.Dropout(0.3),
        layers.Conv2D(128, (4, 4), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (4, 4, 128)
        layers.Dropout(0.4),
        layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Dropout in dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Dropout in dense layers
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def m8(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (2, 2), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (16, 16, 32)
 
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (8, 8, 64)

        layers.Conv2D(128, (4, 4), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (4, 4, 128)

        layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01)),

        layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def m9(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (2, 2), activation='relu',  padding='same',kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (16, 16, 32)
 
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (8, 8, 64)

        layers.Conv2D(128, (4, 4), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (4, 4, 128)

        layers.Conv2D(256, (5, 5), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01)),

        layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def m10(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (2, 2), activation='relu',  padding='same',kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (16, 16, 32)
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (8, 8, 64)
        layers.Dropout(0.2),
        layers.Conv2D(128, (4, 4), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Output: (4, 4, 128)
        layers.Dropout(0.2),
        layers.Conv2D(256, (5, 5), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def m11(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),  # Dropout after max-pooling

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Dropout after max-pooling

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),  # Dropout after max-pooling

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.5),  # Dropout after flattening

        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2()),
        layers.Dropout(0.5),  # Dropout in dense layers
        layers.Dense(128, activation='relu',  kernel_regularizer=regularizers.l2()),
        layers.Dropout(0.3),  # Dropout in dense layers
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_cnn_model(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def large_kernel_first(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(128, (6, 6), activation='relu', ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (4, 4), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),  # Same adjustment
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def small_kernel_first(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def big_model(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        # Adjusted kernel size or padding
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # Padding ensures kernel fits
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def bm_2(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),  # Dropout after max-pooling

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Dropout after max-pooling

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),  # Dropout after max-pooling

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.5),  # Dropout after flattening

        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Dropout in dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Dropout in dense layers
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def bm_3(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Layer 1: Small kernel for fine details
        layers.Conv2D(32, (3, 3), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Layer 2: Slightly larger kernel
        layers.Conv2D(64, (4, 4), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Layer 3: Even larger kernel
        layers.Conv2D(128, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Layer 4: Largest kernel for deeper features
        layers.Conv2D(256, (6, 6), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def bm_4(input_shape=(32, 32, 1), num_classes=20):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu',  padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),  # Dropout after max-pooling

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Dropout after max-pooling

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),  # Dropout after max-pooling

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # Dropout after flattening

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.3),  # Dropout after flattening

        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Dropout in dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Dropout in dense layers
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def model_compile_adam(model):
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    model.summary()
    return model


def train_model(model, x_tr, y_tr, x_val, y_val, epochs=100, batch_size=32):
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10)

    history = model.fit(
        x_tr, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping]
    )

    return model, history




def main():
    # Load Data
    x_tr_color, x_test_color, x_tr_gy, x_test_gy, y_tr, y_test = load_data_all()
    
    
    # x_perturb_color, x_perturb_gy, y_perturb = load_data_perturbed()
    # Display Data
    # for i in range(40, 45):
    #     display_image(x_tr_color, x_tr_gy, y_tr, index=i)
    #     display_image(x_perturb_color, x_perturb_gy, y_perturb, index=i)

    # Split validation set
    x_tr, x_val, y_tr, y_val = validation_set(x_tr_gy, y_tr, 0.2)

    # Define models to run
    m_names = ["m11"]
    functions = [m11]  

    for i in range(len(functions)):
        function = functions[i]  
        model = function()  

        # Compile the model
        model = model_compile_adam(model)

        # One-hot encode the labels
        y_tr_categorical = to_categorical(y_tr, num_classes=20)
        y_val_categorical = to_categorical(y_val, num_classes=20)

        # Train the model
        model, history = train_model(model, x_tr, y_tr_categorical, x_val, y_val_categorical)

        # Save training history and model
        with open(f"data/{m_names[i]}.json", 'w') as file:
            json.dump(history.history, file)
        model.save(f"data/{m_names[i]}.keras")



if __name__ == "__main__":
    main()
