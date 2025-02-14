import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.ttk import Button, Label, Progressbar
from PIL import Image, ImageDraw, ImageGrab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import threading
import asyncio

# ------------------------------
# Параметры
# ------------------------------
IMAGE_SIZE = 28
NUM_CLASSES = 10
DATA_DIR = 'data'
MODEL_FILE = 'digit_recognizer.h5'
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 64


# ------------------------------
# Функции для работы с данными
# ------------------------------

def create_data_directory():
    """Создает директорию для сохранения данных, если она не существует."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        for i in range(NUM_CLASSES):
            os.makedirs(os.path.join(DATA_DIR, str(i)))


def save_drawing(image, label):
    """Сохраняет изображение в указанную папку."""
    create_data_directory()
    count = len(os.listdir(os.path.join(DATA_DIR, str(label))))
    filename = os.path.join(DATA_DIR, str(label), f"{count}.png")
    image.save(filename)
    print(f"Изображение сохранено как {filename}")


def save_data_npz(images, labels):
    """Сохранение данных в формат .npz"""
    np.savez_compressed('digit_data.npz', images=images, labels=labels)


def load_data_npz():
    """Загрузка данных из файла .npz"""
    data = np.load('digit_data.npz')
    return data['images'], data['labels']


def load_data():
    """Загружает данные из директории и подготавливает их для обучения."""
    images = []
    labels = []
    for i in range(NUM_CLASSES):
        digit_dir = os.path.join(DATA_DIR, str(i))
        for filename in os.listdir(digit_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(digit_dir, filename)
                img = Image.open(img_path).convert('L')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(i)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# ------------------------------
# Функции для создания и обучения модели
# ------------------------------

def create_model():
    """Создает сверточную нейронную сеть."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


def train_model(model, images, labels, progress_callback=None):
    """Обучает модель на предоставленных данных."""
    labels = to_categorical(labels, num_classes=NUM_CLASSES)
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)

    # Добавление EarlyStopping и ModelCheckpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val),
                        callbacks=[early_stopping, checkpoint])

    if progress_callback:
        progress_callback(100)  # Завершаем прогресс-бар

    save_model(model)
    return history


def save_model(model, filename=MODEL_FILE):
    """Сохраняет обученную модель."""
    model.save(filename)
    print(f"Модель сохранена в {filename}")


def load_model(filename=MODEL_FILE):
    """Загружает обученную модель."""
    try:
        model = tf.keras.models.load_model(filename)
        print(f"Модель загружена из {filename}")
        return model
    except OSError:
        print("Файл модели не найден. Пожалуйста, обучите модель сначала.")
        return None


# ------------------------------
# Графический интерфейс
# ------------------------------

class DigitRecognizerApp:
    """Основной класс для приложения распознавания рукописных цифр."""

    def __init__(self, master):
        self.master = master
        master.title("Распознавание рукописных цифр")
        master.geometry("450x500")

        self.model = load_model()  # Загружаем модель при старте

        # Создание холста для рисования
        self.canvas_width = 250
        self.canvas_height = 250
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white", bd=2,
                                relief="solid")
        self.canvas.pack(pady=10)

        # Подсказка
        self.label = Label(master, text="Нарисуйте цифру здесь", font=("Helvetica", 12, "italic"))
        self.label.pack(pady=5)

        # Кнопки управления
        self.clear_button = Button(master, text="Очистить", command=self.clear_canvas, width=18)
        self.clear_button.pack(side=tk.LEFT, padx=10)

        self.predict_button = Button(master, text="Распознать", command=self.async_predict, width=18)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.save_button = Button(master, text="Сохранить для обучения", command=self.save_for_training, width=20)
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.train_button = Button(master, text="Обучить модель", command=self.train_model, width=20)
        self.train_button.pack(side=tk.LEFT, padx=10)

        # Ввод цифры для сохранения
        self.digit_var = tk.StringVar(value="0")
        self.digit_entry = tk.Entry(master, textvariable=self.digit_var, width=5, font=("Helvetica", 14))
        self.digit_entry.pack(pady=10)

        # Метка для предсказания
        self.prediction_label = Label(master, text="Предсказание: ", font=("Helvetica", 14))
        self.prediction_label.pack(pady=10)

        # Прогресс-бар для обучения
        self.progress = Progressbar(master, length=200, mode="indeterminate")
        self.progress.pack(pady=10)

        # Настройка холста для рисования
        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x = None
        self.last_y = None

    def paint(self, event):
        """Рисует на холсте."""
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=10, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
        self.last_x = event.x
        self.last_y = event.y

    def clear_canvas(self):
        """Очищает холст для рисования."""
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.prediction_label.config(text="Предсказание: ")

    def async_predict(self):
        """Асинхронное предсказание с обновлением UI."""
        async def run_prediction():
            await self.predict_digit()

        asyncio.run(run_prediction())

    async def predict_digit(self):
        """Распознает цифру, нарисованную на холсте."""
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена. Пожалуйста, обучите модель сначала.")
            return

        if self.last_x is None or self.last_y is None:
            messagebox.showerror("Ошибка", "Пожалуйста, нарисуйте цифру перед распознаванием.")
            return

        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)

        self.prediction_label.config(text=f"Предсказание: {digit}")

    def save_for_training(self):
        """Сохраняет нарисованное изображение для обучения."""
        digit = self.digit_var.get()
        if not digit.isdigit() or int(digit) < 0 or int(digit) > 9:
            messagebox.showerror("Ошибка", "Пожалуйста, введите цифру от 0 до 9.")
            return

        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

        save_drawing(img, int(digit))
        messagebox.showinfo("Сохранено", "Изображение сохранено для обучения.")

    def train_model(self):
        """Обучает модель на собранных данных."""
        images, labels = load_data()
        if len(images) == 0:
            messagebox.showerror("Ошибка", "Нет данных для обучения. Сначала соберите данные.")
            return

        self.progress.start()
        self.model = create_model()

        def update_progress_bar(progress):
            """Обновляет прогресс-бар по мере обучения"""
            self.progress["value"] = progress

        train_model(self.model, images, labels, progress_callback=update_progress_bar)
        self.progress.stop()
        messagebox.showinfo("Успех", "Модель обучена и сохранена.")


# ------------------------------
# Запуск приложения
# ------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
