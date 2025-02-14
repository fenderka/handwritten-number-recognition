import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Button, Label, Progressbar
from PIL import Image, ImageGrab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import threading
import logging

# ------------------------------
# Настройка логирования
# ------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------
# Параметры (Константы)
# ------------------------------
IMAGE_SIZE = 28
NUM_CLASSES = 10
DATA_DIR = 'data'
MODEL_FILE = 'digit_recognizer.keras'  # Используем .keras формат
BEST_MODEL_FILE = 'best_model.keras' # Для checkpoint
LEARNING_RATE = 0.001
EPOCHS = 20  # Сократил для демонстрации. Можно вернуть к 100.
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 5
RANDOM_STATE = 42

# ------------------------------
# Функции для работы с данными
# ------------------------------

def create_data_directory():
    """Создает директорию для сохранения данных, если она не существует."""
    try:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            for i in range(NUM_CLASSES):
                os.makedirs(os.path.join(DATA_DIR, str(i)))
        logging.info(f"Директория данных '{DATA_DIR}' успешно создана/проверена.")
    except OSError as e:
        logging.error(f"Ошибка при создании директории данных: {e}")
        raise


def save_drawing(image, label):
    """Сохраняет изображение в указанную папку."""
    try:
        create_data_directory()
        count = len(os.listdir(os.path.join(DATA_DIR, str(label))))
        filename = os.path.join(DATA_DIR, str(label), f"{count}.png")
        image.save(filename)
        logging.info(f"Изображение сохранено как {filename}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении изображения: {e}")
        messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить изображение: {e}")


def load_data():
    """Загружает данные из директории и подготавливает их для обучения."""
    images = []
    labels = []
    try:
        for i in range(NUM_CLASSES):
            digit_dir = os.path.join(DATA_DIR, str(i))
            for filename in os.listdir(digit_dir):
                if filename.endswith(".png"):
                    img_path = os.path.join(digit_dir, filename)
                    try:
                        img = Image.open(img_path).convert('L')
                        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                        img_array = np.array(img) / 255.0
                        images.append(img_array)
                        labels.append(i)
                    except Exception as e:
                        logging.warning(f"Не удалось загрузить изображение {img_path}: {e}")

        images = np.array(images)
        labels = np.array(labels)
        logging.info(f"Загружено {len(images)} изображений из директории {DATA_DIR}")
        return images, labels
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        messagebox.showerror("Ошибка загрузки данных", f"Не удалось загрузить данные: {e}")
        return np.array([]), np.array([])  # Возвращаем пустые массивы, чтобы не сломать обучение


# ------------------------------
# Функции для создания и обучения модели
# ------------------------------

def create_model():
    """Создает сверточную нейронную сеть."""
    try:
        model = Sequential([
            Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),  # Добавляем Input слой
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        logging.info("Модель успешно создана.")
        return model
    except Exception as e:
        logging.error(f"Ошибка при создании модели: {e}")
        messagebox.showerror("Ошибка", f"Не удалось создать модель: {e}")
        return None


def train_model(model, images, labels, progress_callback=None):
    """Обучает модель на предоставленных данных."""
    try:
        labels = to_categorical(labels, num_classes=NUM_CLASSES)

        x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
        x_train = np.expand_dims(x_train, axis=-1)
        x_val = np.expand_dims(x_val, axis=-1)

        # Добавление EarlyStopping и ModelCheckpoint
        early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
        checkpoint = ModelCheckpoint(BEST_MODEL_FILE, save_best_only=True, monitor='val_loss', verbose=0) # verbose=0 чтобы не спамить в консоль

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        logging.info("Начинаем обучение модели...")

        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, checkpoint],
            verbose=0  # Отключаем вывод в консоль, чтобы прогресс-бар работал корректно
        )

        logging.info("Обучение модели завершено.")

        if progress_callback:
            progress_callback(100)  # Завершаем прогресс-бар

        save_model(model)
        return history
    except Exception as e:
        logging.error(f"Ошибка при обучении модели: {e}")
        messagebox.showerror("Ошибка", f"Не удалось обучить модель: {e}")
        return None


def save_model(model, filename=MODEL_FILE):
    """Сохраняет обученную модель."""
    try:
        model.save(filename)
        logging.info(f"Модель сохранена в {filename}")
        messagebox.showinfo("Сохранение модели", f"Модель успешно сохранена в {filename}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении модели: {e}")
        messagebox.showerror("Ошибка", f"Не удалось сохранить модель: {e}")


def load_model(filename=MODEL_FILE):
    """Загружает обученную модель."""
    try:
        model = tf.keras.models.load_model(filename)
        logging.info(f"Модель загружена из {filename}")
        return model
    except OSError:
        logging.warning("Файл модели не найден. Пожалуйста, обучите модель сначала.")
        messagebox.showinfo("Внимание", "Файл модели не найден.  Модель будет создана заново.")
        return None
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели: {e}")
        messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")
        return None


# ------------------------------
# Графический интерфейс
# ------------------------------

class DigitRecognizerApp:
    """Основной класс для приложения распознавания рукописных цифр."""

    def __init__(self, master):
        self.master = master
        master.title("Распознавание рукописных цифр")
        master.geometry("450x550")  # Увеличил высоту окна

        self.model = load_model()  # Загружаем модель при старте
        if self.model is None:
            self.model = create_model() # Если загрузка не удалась, создаем новую модель

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

        self.predict_button = Button(master, text="Распознать", command=self.predict_digit, width=18)  # Убрал async
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.save_button = Button(master, text="Сохранить для обучения", command=self.save_for_training, width=20)
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.train_button = Button(master, text="Обучить модель", command=self.start_training_thread, width=20) # Запускаем обучение в потоке
        self.train_button.pack(side=tk.LEFT, padx=10)

        # Ввод цифры для сохранения
        self.digit_var = tk.StringVar(value="0")
        self.digit_entry = tk.Entry(master, textvariable=self.digit_var, width=5, font=("Helvetica", 14))
        self.digit_entry.pack(pady=10)

        # Метка для предсказания
        self.prediction_label = Label(master, text="Предсказание: ", font=("Helvetica", 14))
        self.prediction_label.pack(pady=10)

        # Прогресс-бар для обучения
        self.progress = Progressbar(master, length=200, mode="determinate", maximum=100)
        self.progress.pack(pady=10)

        # Настройка холста для рисования
        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x = None
        self.last_y = None

        self.is_training = False # Флаг, чтобы избежать одновременного обучения

    def paint(self, event):
        """Рисует на холсте."""
        try:
            if self.last_x and self.last_y:
                self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                        width=10, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.last_x = event.x
            self.last_y = event.y
        except Exception as e:
            logging.error(f"Ошибка при рисовании: {e}")

    def clear_canvas(self):
        """Очищает холст для рисования."""
        try:
            self.canvas.delete("all")
            self.last_x = None
            self.last_y = None
            self.prediction_label.config(text="Предсказание: ")
        except Exception as e:
            logging.error(f"Ошибка при очистке холста: {e}")

    def predict_digit(self):
        """Распознает цифру, нарисованную на холсте."""
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена. Пожалуйста, обучите модель сначала.")
            return

        if self.last_x is None or self.last_y is None:
            messagebox.showerror("Ошибка", "Пожалуйста, нарисуйте цифру перед распознаванием.")
            return

        try:
            x = self.master.winfo_rootx() + self.canvas.winfo_x()
            y = self.master.winfo_rooty() + self.canvas.winfo_y()
            x1 = x + self.canvas_width
            y1 = y + self.canvas_height
            img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            img_array = np.expand_dims(img_array, axis=-1)

            prediction = self.model.predict(img_array, verbose=0) # verbose=0 чтобы не спамить в консоль
            digit = np.argmax(prediction)

            self.prediction_label.config(text=f"Предсказание: {digit}")

        except Exception as e:
            logging.error(f"Ошибка при предсказании цифры: {e}")
            messagebox.showerror("Ошибка", f"Не удалось распознать цифру: {e}")

    def save_for_training(self):
        """Сохраняет нарисованное изображение для обучения."""
        digit = self.digit_var.get()
        if not digit.isdigit() or int(digit) < 0 or int(digit) > 9:
            messagebox.showerror("Ошибка", "Пожалуйста, введите цифру от 0 до 9.")
            return

        try:
            x = self.master.winfo_rootx() + self.canvas.winfo_x()
            y = self.master.winfo_rooty() + self.canvas.winfo_y()
            x1 = x + self.canvas_width
            y1 = y + self.canvas_height
            img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

            save_drawing(img, int(digit))
            messagebox.showinfo("Сохранено", "Изображение сохранено для обучения.")

        except Exception as e:
            logging.error(f"Ошибка при сохранении для обучения: {e}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить изображение для обучения: {e}")

    def start_training_thread(self):
        """Запускает обучение модели в отдельном потоке."""
        if self.is_training:
            messagebox.showinfo("Внимание", "Обучение уже идет!")
            return

        if self.model is None:
            messagebox.showerror("Ошибка", "Невозможно начать обучение. Модель не существует.")
            return

        self.is_training = True
        threading.Thread(target=self.train_model_thread, daemon=True).start()

    def train_model_thread(self):
        """Обучает модель на собранных данных (выполняется в отдельном потоке)."""
        try:
            images, labels = load_data()
            if len(images) == 0:
                messagebox.showerror("Ошибка", "Нет данных для обучения. Сначала соберите данные.")
                return

            # Блокируем кнопку "Обучить модель"
            self.train_button.config(state="disabled")

            # Функция для обновления прогресс-бара (нужно вызывать через `after`)
            def update_progress_bar(progress):
                self.progress["value"] = progress
                self.master.update_idletasks()  # Обновляем UI

            # Callback, который вызывает обновление прогресс-бара
            def progress_callback(progress):
                self.master.after(0, update_progress_bar, progress)  # Вызываем в главном потоке

            # Запускаем обучение
            train_model(self.model, images, labels, progress_callback=progress_callback)

            # Выводим сообщение об успехе
            messagebox.showinfo("Успех", "Модель обучена и сохранена.")

        except Exception as e:
            logging.error(f"Ошибка в потоке обучения: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Произошла ошибка во время обучения: {e}")
        finally:
            # Разблокируем кнопку "Обучить модель" и сбрасываем флаг
            self.train_button.config(state="normal")
            self.is_training = False
            self.master.after(0, self.progress.stop) # Останавливаем прогрессбар
            self.master.after(0, self.progress.config, {"value": 0})  # Сбрасываем значение прогрессбара


# ------------------------------
# Запуск приложения
# ------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
