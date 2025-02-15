import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_model(image_size, num_classes):
    """
    Создает сверточную нейронную сеть (CNN) для распознавания цифр.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, images, labels, image_size, num_classes, learning_rate, epochs, batch_size, progress_callback=None):
    """
    Обучает модель на предоставленных данных.
    """
    try:
        labels = to_categorical(labels, num_classes=num_classes)
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
        x_train = np.expand_dims(x_train, axis=-1)
        x_val = np.expand_dims(x_val, axis=-1)

        # Аугментация данных
        data_generator = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
        )
        data_generator.fit(x_train)

        # Добавление EarlyStopping и ModelCheckpoint
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Использование data_generator.flow для обучения
        history = model.fit(data_generator.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            steps_per_epoch=len(x_train) // batch_size,
                            validation_data=(x_val, y_val),
                            callbacks=[early_stopping, checkpoint])

        if progress_callback:
            progress_callback(100)

        return history

    except Exception as e:
        print(f"Ошибка во время обучения модели: {e}")
        return None

def save_model(model, filename):
    """
    Сохраняет обученную модель в файл.
    """
    try:
        model.save(filename)
        print(f"Модель сохранена в {filename}")
    except Exception as e:
        print(f"Ошибка при сохранении модели: {e}")

def load_model(filename):
    """
    Загружает обученную модель из файла.
    """
    try:
        model = tf.keras.models.load_model(filename)
        print(f"Модель загружена из {filename}")
        return model
    except FileNotFoundError:
        print("Файл модели не найден. Пожалуйста, обучите модель сначала.")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None
