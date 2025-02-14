import tkinter as tk
from tkinter import messagebox, colorchooser
from tkinter.ttk import Button, Label, Progressbar
from PIL import Image, ImageGrab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import threading
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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
MODEL_FILE = 'digit_recognizer.keras'
BEST_MODEL_FILE = 'best_model.keras'
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10
RANDOM_STATE = 42
ROTATION_RANGE = 15
ZOOM_RANGE = 0.1
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
INITIAL_LINE_WIDTH = 10
DEFAULT_COLOR = "black"

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
            Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        logging.info("Модель успешно создана.")
        return model
    except Exception as e:
        logging.error(f"Ошибка при создании модели: {e}")
        messagebox.showerror("Ошибка", f"Не удалось создать модель: {e}")
        return None

def learning_rate_scheduler(epoch):
    """Уменьшает learning rate каждые 10 эпох."""
    if epoch % 10 == 0 and epoch > 0:
        return LEARNING_RATE * 0.5
    return LEARNING_RATE


def train_model(model, images, labels, progress_callback=None, history_callback=None):
    """Обучает модель на предоставленных данных."""
    try:
        labels = to_categorical(labels, num_classes=NUM_CLASSES)

        x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
        x_train = np.expand_dims(x_train, axis=-1)
        x_val = np.expand_dims(x_val, axis=-1)

        # Аугментация данных
        datagen = ImageDataGenerator(
            rotation_range=ROTATION_RANGE,
            zoom_range=ZOOM_RANGE,
            width_shift_range=WIDTH_SHIFT_RANGE,
            height_shift_range=HEIGHT_SHIFT_RANGE)

        datagen.fit(x_train)

        # Колбэки
        early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
        checkpoint = ModelCheckpoint(BEST_MODEL_FILE, save_best_only=True, monitor='val_loss', verbose=0)
        lr_scheduler = LearningRateScheduler(learning_rate_scheduler)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        logging.info("Начинаем обучение модели...")

        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, checkpoint, lr_scheduler],
            verbose=0)

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
        master.geometry("650x700")  # Увеличил размер окна

        self.model = load_model()
        if self.model is None:
            self.model = create_model()

        # Параметры рисования
        self.line_width = INITIAL_LINE_WIDTH
        self.paint_color = DEFAULT_COLOR

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
        self.clear_button = Button(master, text="Очистить", command=self.clear_canvas, width=15)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.predict_button = Button(master, text="Распознать", command=self.predict_digit, width=15)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.save_button = Button(master, text="Сохранить для обучения", command=self.save_for_training, width=20)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.train_button = Button(master, text="Обучить модель", command=self.start_training_thread, width=20)
        self.train_button.pack(side=tk.LEFT, padx=5)

        # Настройки рисования
        self.width_label = Label(master, text="Толщина линии:", font=("Helvetica", 10))
        self.width_label.pack()

        self.width_scale = tk.Scale(master, from_=1, to=30, orient=tk.HORIZONTAL, command=self.change_width, length=150)
        self.width_scale.set(self.line_width)
        self.width_scale.pack()

        self.color_button = Button(master, text="Выбрать цвет", command=self.choose_color, width=15)
        self.color_button.pack(pady=5)

        # Ввод цифры для сохранения
        self.digit_var = tk.StringVar(value="0")
        self.digit_entry = tk.Entry(master, textvariable=self.digit_var, width=5, font=("Helvetica", 14))
        self.digit_entry.pack(pady=5)

        # Метка для предсказания
        self.prediction_label = Label(master, text="Предсказание: ", font=("Helvetica", 14))
        self.prediction_label.pack(pady=5)

        # Прогресс-бар для обучения
        self.progress = Progressbar(master, length=200, mode="determinate", maximum=100)
        self.progress.pack(pady=5)

        # Область для графика обучения
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget.get_tk_widget().pack(pady=5)

        # Настройка холста для рисования
        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x = None
        self.last_y = None
        self.is_training = False

    def paint(self, event):
        """Рисует на холсте."""
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=self.line_width, fill=self.paint_color, capstyle=tk.ROUND, smooth=tk.TRUE)
        self.last_x = event.x
        self.last_y = event.y

    def clear_canvas(self):
        """Очищает холст для рисования."""
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.prediction_label.config(text="Предсказание: ")

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

            prediction = self.model.predict(img_array, verbose=0)
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
        self.clear_plot()  # Очистить график перед новым обучением
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

            # Функция для обновления прогресс-бара
            def update_progress_bar(progress):
                self.progress["value"] = progress
                self.master.update_idletasks()

            # Callback для обновления прогресс-бара
            def progress_callback(progress):
                self.master.after(0, update_progress_bar, progress)

            def history_callback(history):
                 self.master.after(0, self.update_plot, history)

            history = train_model(self.model, images, labels, progress_callback=progress_callback)

            if history:
                history_callback(history.history)

            # Выводим сообщение об успехе
            messagebox.showinfo("Успех", "Модель обучена и сохранена.")

        except Exception as e:
            logging.error(f"Ошибка в потоке обучения: {e}", exc_info=True)
            messagebox.showerror("Ошибка", f"Произошла ошибка во время обучения: {e}")
        finally:
            # Разблокируем кнопку "Обучить модель" и сбрасываем флаг
            self.train_button.config(state="normal")
            self.is_training = False
            self.master.after(0, self.progress.stop)
            self.master.after(0, self.progress.config, {"value": 0})

    def change_width(self, value):
        """Изменяет толщину линии рисования."""
        self.line_width = int(value)

    def choose_color(self):
        """Выбирает цвет линии рисования."""
        color_code = colorchooser.askcolor(title="Выберите цвет")
        if color_code:
            self.paint_color = color_code[1]

    def update_plot(self, history):
        """Обновляет график обучения."""
        self.plot.clear()
        self.plot.plot(history['accuracy'])
        self.plot.plot(history['val_accuracy'])
        self.plot.set_title('Точность модели')
        self.plot.set_ylabel('Точность')
        self.plot.set_xlabel('Эпоха')
        self.plot.legend(['Обучение', 'Валидация'], loc='upper left')
        self.canvas_widget.draw()

    def clear_plot(self):
         """Очищает график обучения"""
         self.plot.clear()
         self.canvas_widget.draw()

# ------------------------------
# Запуск приложения
# ------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
