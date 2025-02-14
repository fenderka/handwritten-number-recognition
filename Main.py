import tkinter as tk
from tkinter import messagebox, filedialog, Canvas, StringVar, Entry
from tkinter.ttk import Button, Label, Progressbar, Frame, LabelFrame, Style
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import threading
import time

# Отключение аппаратного ускорения (если необходимо)
os.environ['TK_SILENCE_DEPRECATION'] = '1'
os.environ['PYGAME_DISPLAY'] = 'x11'

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
    return image


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
        progress_callback(100)

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
    def __init__(self, master):
        self.master = master
        master.title("Распознавание рукописных цифр")

        style = Style()
        style.theme_use('clam')  # clam - более легкая тема

        self.model = load_model()
        self.prediction_thread = None

        # --- Frames ---
        canvas_frame = LabelFrame(master, text="Рисование", padding=10)
        canvas_frame.pack(pady=10, padx=10, fill=tk.X)

        prediction_frame = LabelFrame(master, text="Результат распознавания", padding=10)
        prediction_frame.pack(pady=10, padx=10, fill=tk.X)

        button_frame = Frame(master, padding=10)
        button_frame.pack(pady=10, padx=10, fill=tk.X)

        training_frame = LabelFrame(master, text="Обучение модели", padding=10)
        training_frame.pack(pady=10, padx=10, fill=tk.X)

        thumbnail_frame = LabelFrame(master, text="Сохраненное изображение", padding=10)
        thumbnail_frame.pack(pady=10, padx=10, fill=tk.X)

        # --- Canvas ---
        self.canvas_width = 300
        self.canvas_height = 300
        self.canvas = Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="white", bd=2,
                             relief="solid")
        self.canvas.pack()

        # --- Prediction Labels ---
        self.prediction_label = Label(prediction_frame, text="Предсказание: ", font=("Helvetica", 16))
        self.prediction_label.pack(pady=5)

        self.confidence_label = Label(prediction_frame, text="Уверенность: ", font=("Helvetica", 16))
        self.confidence_label.pack(pady=5)

        # --- Buttons ---
        self.clear_button = Button(button_frame, text="Очистить", command=self.clear_canvas, width=15)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.predict_button = Button(button_frame, text="Распознать", command=self.async_predict, width=15)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.save_png_button = Button(button_frame, text="Сохранить PNG", command=self.save_png, width=15)
        self.save_png_button.pack(side=tk.LEFT, padx=5)

        # --- Training Widgets ---
        self.digit_var = StringVar(value="0")
        self.digit_entry = Entry(training_frame, textvariable=self.digit_var, width=5, font=("Helvetica", 14))
        self.digit_entry.pack(side=tk.LEFT, padx=5)
        self.digit_entry.insert(0, '0')

        self.save_button = Button(training_frame, text="Сохранить для обучения", command=self.save_for_training,
                                  width=20)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.train_button = Button(training_frame, text="Обучить модель", command=self.train_model_thread, width=20)
        self.train_button.pack(pady=10, padx=5)

        self.progress = Progressbar(master, length=200, mode="indeterminate")
        self.progress.pack(pady=10)

        # --- Thumbnail ---
        self.thumbnail_label = Label(thumbnail_frame)
        self.thumbnail_label.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x = None
        self.last_y = None

    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=10, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
        self.last_x = event.x
        self.last_y = event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.prediction_label.config(text="Предсказание: ")
        self.confidence_label.config(text="Уверенность: ")
        self.thumbnail_label.config(image=None)
        self.thumbnail_label.image = None

    def async_predict(self):
        if self.prediction_thread and self.prediction_thread.is_alive():
            return

        self.prediction_thread = threading.Thread(target=self.predict_digit)
        self.prediction_thread.start()

    def predict_digit(self):
        if self.model is None:
            self.update_gui_after("Ошибка", "Модель не загружена.")
            return

        if self.last_x is None or self.last_y is None:
            self.update_gui_after("Ошибка", "Пожалуйста, нарисуйте цифру.")
            return

        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)

        start_time = time.time()
        prediction = self.model.predict(img_array)
        end_time = time.time()
        print(f"Prediction time: {end_time - start_time:.4f} seconds")

        digit = np.argmax(prediction)
        confidence = prediction[0][digit] * 100

        self.update_gui_after("Предсказание", f"Предсказание: {digit}, Уверенность: {confidence:.2f}%")

    def update_label(self, label, text):
        self.master.after(0, lambda: label.config(text=text))

    def update_gui_after(self, title, message):
        self.master.after(0, lambda: self.update_gui(title, message))

    def update_gui(self, title, message):
        if title == "Ошибка":
            messagebox.showerror(title, message)
        elif title == "Предсказание":
            self.update_label(self.prediction_label, message.split(',')[0])
            self.update_label(self.confidence_label, message.split(',')[1])

    def save_for_training(self):
        digit = self.digit_var.get()
        if not digit.isdigit() or int(digit) < 0 or int(digit) > 9:
            messagebox.showerror("Ошибка", "Пожалуйста, введите цифру от 0 до 9.")
            return

        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

        saved_image = save_drawing(img, int(digit))

        saved_image.thumbnail((50, 50))
        photo = ImageTk.PhotoImage(saved_image)
        self.thumbnail_label.config(image=photo)
        self.thumbnail_label.image = photo

        messagebox.showinfo("Сохранено", "Изображение сохранено для обучения.")

    def train_model_thread(self):
        self.progress.start()
        threading.Thread(target=self.train_model).start()

    def train_model(self):
        images, labels = load_data()
        if len(images) == 0:
            messagebox.showerror("Ошибка", "Нет данных для обучения. Сначала соберите данные.")
            return

        self.model = create_model()

        def update_progress_bar(progress):
            self.progress["value"] = progress

        train_model(self.model, images, labels, progress_callback=update_progress_bar)
        self.progress.stop()
        messagebox.showinfo("Успех", "Модель обучена и сохранена.")

    def save_png(self):
        if self.last_x is None or self.last_y is None:
            messagebox.showerror("Ошибка", "Пожалуйста, нарисуйте цифру перед сохранением.")
            return

        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height
        img = ImageGrab.grab().crop((x, y, x1, y1))

        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            img.save(file_path)
            messagebox.showinfo("Успех", f"Изображение сохранено как {file_path}")


# ------------------------------
# Запуск приложения
# ------------------------------

if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
