
import tkinter as tk
from tkinter import messagebox, filedialog, Canvas, StringVar, Entry, simpledialog, Toplevel, Label
from tkinter.ttk import Button, Label, Progressbar, Frame, LabelFrame, Style
from PIL import Image, ImageTk, ImageGrab, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import threading
import time
import json  # Для сохранения и загрузки настроек горячих клавиш

# Отключение аппаратного ускорения (если необходимо)
os.environ['TK_SILENCE_DEPRECATION'] = '1'
os.environ['PYGAME_DISPLAY'] = 'x11'

# ------------------------------
# Параметры
# ------------------------------
IMAGE_SIZE = 28  # Размер изображения, к которому приводятся все входные данные.
NUM_CLASSES = 10 # Количество классов (цифры от 0 до 9).
DATA_DIR = 'data' # Директория для сохранения данных для обучения.
MODEL_FILE = 'digit_recognizer.h5' # Имя файла для сохранения обученной модели.
LEARNING_RATE = 0.001 # Скорость обучения для оптимизатора Adam.
EPOCHS = 100 # Максимальное количество эпох обучения.
BATCH_SIZE = 64 # Размер пакета данных для обучения.
HOTKEY_CONFIG_FILE = "hotkeys.json"  # Файл для сохранения настроек горячих клавиш


# ------------------------------
# Функции для работы с данными
# ------------------------------

def create_data_directory():
    """
    Создает директорию для сохранения данных, если она не существует.

    Внутри создается структура папок для каждой цифры (0-9), чтобы хранить изображения, соответствующие этим цифрам.
    """
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            for i in range(NUM_CLASSES):
                os.makedirs(os.path.join(DATA_DIR, str(i)))
        except OSError as e:
            print(f"Ошибка при создании директории: {e}")
            return False
    return True


def save_drawing(image, label):
    """
    Сохраняет изображение в указанную папку, соответствующую метке (цифре).

    Args:
        image (PIL.Image): Изображение для сохранения.
        label (int): Метка изображения (цифра от 0 до 9).

    Returns:
        PIL.Image: Возвращает сохраненное изображение в случае успеха, None в случае ошибки.
    """
    if not create_data_directory():
        return None
    try:
        count = len(os.listdir(os.path.join(DATA_DIR, str(label))))
        filename = os.path.join(DATA_DIR, str(label), f"{count}.png")
        image.save(filename)
        print(f"Изображение сохранено как {filename}")
        return image
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}")
        return None


def save_data_npz(images, labels):
    """Сохранение данных в формат .npz"""
    try:
        np.savez_compressed('digit_data.npz', images=images, labels=labels)
    except Exception as e:
        print(f"Ошибка при сохранении в .npz: {e}")


def load_data_npz():
    """Загрузка данных из файла .npz"""
    try:
        data = np.load('digit_data.npz')
        return data['images'], data['labels']
    except FileNotFoundError:
        print("Файл digit_data.npz не найден.")
        return None, None
    except Exception as e:
        print(f"Ошибка при загрузке из .npz: {e}")
        return None, None


def load_data():
    """
    Загружает данные из директории DATA_DIR и подготавливает их для обучения.

    Читает изображения из папок, соответствующих цифрам (0-9), изменяет размер до IMAGE_SIZE x IMAGE_SIZE,
    нормализует значения пикселей и возвращает массивы NumPy для изображений и меток.

    Returns:
        tuple: Кортеж из двух массивов NumPy: images (изображения) и labels (метки).
               Возвращает пустые массивы в случае ошибки.
    """
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
                    except FileNotFoundError:
                        print(f"Файл не найден: {img_path}")
                    except Exception as e:
                        print(f"Ошибка при обработке изображения {img_path}: {e}")

        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return np.array([]), np.array([])


# ------------------------------
# Функции для создания и обучения модели
# ------------------------------

def create_model():
    """
    Создает сверточную нейронную сеть (CNN) для распознавания цифр.

    Архитектура модели:
        - Conv2D: 32 фильтра, ядро 3x3, функция активации ReLU, входная форма (IMAGE_SIZE, IMAGE_SIZE, 1).
        - MaxPooling2D: Размер пула 2x2.
        - Conv2D: 64 фильтра, ядро 3x3, функция активации ReLU.
        - MaxPooling2D: Размер пула 2x2.
        - Flatten: Преобразует выходные данные в одномерный вектор.
        - Dense: 128 нейронов, функция активации ReLU.
        - Dense: NUM_CLASSES нейронов, функция активации Softmax (для классификации).

    Returns:
        tf.keras.models.Sequential: Скомпилированная модель CNN.
    """
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
    """
    Обучает модель на предоставленных данных.

    Использует разделение на обучающую и валидационную выборки, аугментацию данных,
    EarlyStopping и ModelCheckpoint для улучшения процесса обучения.

    Args:
        model (tf.keras.models.Sequential): Модель для обучения.
        images (np.array): Массив NumPy с изображениями.
        labels (np.array): Массив NumPy с метками.
        progress_callback (function, optional): Функция обратного вызова для обновления прогресса обучения. Defaults to None.

    Returns:
        tf.keras.callbacks.History: Объект History, содержащий информацию об обучении.
    """
    try:
        labels = to_categorical(labels, num_classes=NUM_CLASSES)
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

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Использование data_generator.flow для обучения
        history = model.fit(data_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
                            epochs=EPOCHS,
                            steps_per_epoch=len(x_train) // BATCH_SIZE,
                            validation_data=(x_val, y_val),
                            callbacks=[early_stopping, checkpoint])

        if progress_callback:
            progress_callback(100)

        save_model(model)
        return history

    except Exception as e:
        print(f"Ошибка во время обучения модели: {e}")
        return None


def save_model(model, filename=MODEL_FILE):
    """
    Сохраняет обученную модель в файл.

    Args:
        model (tf.keras.models.Sequential): Модель для сохранения.
        filename (str, optional): Имя файла для сохранения модели. Defaults to MODEL_FILE.
    """
    try:
        model.save(filename)
        print(f"Модель сохранена в {filename}")
    except Exception as e:
        print(f"Ошибка при сохранении модели: {e}")


def load_model(filename=MODEL_FILE):
    """
    Загружает обученную модель из файла.

    Args:
        filename (str, optional): Имя файла модели. Defaults to MODEL_FILE.

    Returns:
        tf.keras.models.Sequential: Загруженная модель.
                                     Возвращает None, если файл не найден или произошла ошибка.
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


# ------------------------------
# Графический интерфейс
# ------------------------------

class DigitRecognizerApp:
    """
    Класс, представляющий графическое приложение для распознавания рукописных цифр.

    Содержит методы для рисования, очистки холста, загрузки изображений, сохранения данных,
    обучения модели и распознавания цифр.  Также предоставляет интерфейс для настройки горячих клавиш.
    """
    def __init__(self, master):
        self.master = master
        master.title("Распознавание рукописных цифр")

        style = Style()
        style.theme_use('clam')  # clam - более легкая тема

        self.model = load_model()
        self.prediction_thread = None

        # --- Загрузка настроек горячих клавиш из файла ---
        self.hotkey_bindings = self.load_hotkey_config()

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
        self.clear_button.bind("<Enter>", lambda event: self.show_tooltip(self.clear_button, "Очистить холст"))

        self.predict_button = Button(button_frame, text="Распознать", command=self.async_predict, width=15)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        self.predict_button.bind("<Enter>", lambda event: self.show_tooltip(self.predict_button, "Распознать цифру"))

        self.save_png_button = Button(button_frame, text="Сохранить PNG", command=self.save_png, width=15)
        self.save_png_button.pack(side=tk.LEFT, padx=5)
        self.save_png_button.bind("<Enter>", lambda event: self.show_tooltip(self.save_png_button, "Сохранить изображение в формате PNG"))

        self.load_image_button = Button(button_frame, text="Загрузить изображение", command=self.load_image, width=20)
        self.load_image_button.pack(side=tk.LEFT, padx=5)
        self.load_image_button.bind("<Enter>", lambda event: self.show_tooltip(self.load_image_button, "Загрузить изображение с диска"))

        # --- Training Widgets ---
        self.digit_var = StringVar(value="0")
        self.digit_entry = Entry(training_frame, textvariable=self.digit_var, width=5, font=("Helvetica", 14))
        self.digit_entry.pack(side=tk.LEFT, padx=5)
        self.digit_entry.insert(0, '0')
        self.digit_entry.bind("<Enter>", lambda event: self.show_tooltip(self.digit_entry, "Введите цифру от 0 до 9"))

        self.save_button = Button(training_frame, text="Сохранить для обучения", command=self.save_for_training, width=20)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.save_button.bind("<Enter>", lambda event: self.show_tooltip(self.save_button, "Сохранить нарисованную цифру для обучения модели"))

        self.train_button = Button(training_frame, text="Обучить модель", command=self.train_model_thread, width=20)
        self.train_button.pack(pady=10, padx=5)
        self.train_button.bind("<Enter>", lambda event: self.show_tooltip(self.train_button, "Обучить модель на собранных данных"))

        self.progress = Progressbar(master, length=200, mode="indeterminate")
        self.progress.pack(pady=10)

        # --- Thumbnail ---
        self.thumbnail_label = Label(thumbnail_frame)
        self.thumbnail_label.pack()

        # --- Hotkey Configuration ---
        self.hotkey_button = Button(button_frame, text="Настроить горячие клавиши", command=self.configure_hotkeys, width=25)
        self.hotkey_button.pack(side=tk.LEFT, padx=5)

        # --- Help Button ---
        self.help_button = Button(button_frame, text="Справка", command=self.show_help, width=10)
        self.help_button.pack(side=tk.LEFT, padx=5)

        self.bind_hotkeys()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x = None
        self.last_y = None

    def load_hotkey_config(self):
        """Загружает настройки горячих клавиш из файла."""
        try:
            with open(HOTKEY_CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config
        except FileNotFoundError:
            # Если файл не найден, возвращаем настройки по умолчанию
            return {
                "clear": "<Control-c>",
                "predict": "<Control-p>",
                "save": "<Control-s>",
                "load_image": "<Control-o>",
                "save_training": "<Control-t>",
                "train_model": "<Control-m>"
            }
        except json.JSONDecodeError:
            print("Ошибка при чтении файла конфигурации горячих клавиш. Используются настройки по умолчанию.")
            return {
                "clear": "<Control-c>",
                "predict": "<Control-p>",
                "save": "<Control-s>",
                "load_image": "<Control-o>",
                "save_training": "<Control-t>",
                "train_model": "<Control-m>"
            }

    def save_hotkey_config(self):
        """Сохраняет настройки горячих клавиш в файл."""
        try:
            with open(HOTKEY_CONFIG_FILE, "w") as f:
                json.dump(self.hotkey_bindings, f, indent=4)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки горячих клавиш: {e}")


    def bind_hotkeys(self):
        """Привязывает горячие клавиши к функциям."""
        for action, key in self.hotkey_bindings.items():
            try:
                # Попытка отвязать предыдущую горячую клавишу, если она была назначена
                self.master.unbind(key)
            except tk.TclError:
                # Игнорируем ошибку, если клавиша не была привязана ранее
                pass

            # Определяем, какую функцию вызвать в зависимости от действия
            if action == "clear":
                command = self.clear_canvas
            elif action == "predict":
                command = self.async_predict
            elif action == "save":
                command = self.save_png
            elif action == "load_image":
                command = self.load_image
            elif action == "save_training":
                command = self.save_for_training
            elif action == "train_model":
                command = self.train_model_thread
            else:
                print(f"Неизвестное действие для горячей клавиши: {action}")
                continue  # Пропускаем это действие, если оно неизвестно

            # Привязываем новую горячую клавишу
            self.master.bind(key, lambda event, cmd=command: cmd())  # Замыкание для сохранения текущего значения command

        print("Горячие клавиши привязаны.")  # Debug-вывод

    def configure_hotkeys(self):
        config_window = tk.Toplevel(self.master)
        config_window.title("Настройка горячих клавиш")

        def update_hotkey(action, new_key):
            """Обновляет горячую клавишу и сохраняет конфигурацию."""
            if not new_key:
                return  # Если new_key пустой, ничего не делаем

            try:
                # Отвязываем старую горячую клавишу
                if self.hotkey_bindings.get(action):  # Проверяем, была ли ранее назначена горячая клавиша
                    self.master.unbind(self.hotkey_bindings[action])

                # Обновляем привязку в словаре
                self.hotkey_bindings[action] = new_key

                # Сохраняем конфигурацию в файл
                self.save_hotkey_config()

                # Привязываем новую горячую клавишу
                self.bind_hotkeys()  # Перепривязываем все горячие клавиши

                messagebox.showinfo("Успех", f"Горячая клавиша для '{action}' изменена на '{new_key}'")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось изменить горячую клавишу: {e}")
            finally:
                config_window.destroy()


        for action, key in self.hotkey_bindings.items():
            frame = Frame(config_window)
            frame.pack(pady=5, padx=5)

            label = Label(frame, text=f"{action.capitalize()}: {key}")
            label.pack(side=tk.LEFT)

            change_button = Button(frame, text="Изменить",
                                   command=lambda a=action: self.change_hotkey(config_window, a, update_hotkey))
            change_button.pack(side=tk.LEFT, padx=5)

    def change_hotkey(self, config_window, action, update_hotkey):
        new_key = simpledialog.askstring("Изменить горячую клавишу", f"Введите новую горячую клавишу для '{action}':")
        if new_key is not None:  # Проверяем, что пользователь не нажал "Отмена"
            update_hotkey(action, new_key)

    def show_help(self):
        """
        Открывает окно справки с информацией о приложении.
        """
        help_window = Toplevel(self.master)
        help_window.title("Справка")

        help_text = """
        **Распознавание рукописных цифр - Справка**

        Это приложение позволяет распознавать рукописные цифры, нарисованные на холсте или загруженные из файла.

        **Рисование:**
        - Рисуйте цифры на холсте, используя левую кнопку мыши.  Толщина линии рисования - 10 пикселей.

        **Основные элементы интерфейса:**
        - **Холст:** Область для рисования цифр.
        - **Метка "Предсказание":** Отображает предсказанную цифру.
        - **Метка "Уверенность":** Отображает уверенность модели в предсказании в процентах.
        - **Превью сохраненного изображения:** Отображает уменьшенную копию последнего сохраненного изображения для обучения.

        **Кнопки управления:**
        - **Очистить:** Очищает холст, удаляя все нарисованные линии. Горячая клавиша: {clear}
        - **Распознать:** Запускает процесс распознавания цифры, нарисованной на холсте.  Результат (предсказанная цифра и уверенность) отображается в соответствующих метках. Горячая клавиша: {predict}
        - **Сохранить PNG:** Сохраняет содержимое холста в файл в формате PNG.  Открывается диалоговое окно для выбора имени файла и места сохранения. Горячая клавиша: {save}
        - **Загрузить изображение:** Загружает изображение (PNG, JPG, JPEG) с диска и отображает его на холсте.  Изображение автоматически изменяется до размера холста. Горячая клавиша: {load_image}
        - **Настроить горячие клавиши:** Открывает окно для изменения сочетаний клавиш, используемых для быстрого доступа к функциям приложения.
        - **Справка:** Открывает данное окно справки.

        **Обучение модели:**

        Для обучения модели необходимо собрать набор данных, состоящий из изображений цифр и соответствующих им меток.
        1. Нарисуйте цифру на холсте.
        2. В поле ввода рядом с кнопкой "Сохранить для обучения" введите цифру, *соответствующую* нарисованному изображению.  Важно, чтобы введенная цифра соответствовала тому, что изображено на холсте.
        3. Нажмите кнопку "Сохранить для обучения".  Изображение будет сохранено в папку 'data' в подпапку, соответствующую введенной цифре.  Превью сохраненного изображения появится в нижней части окна.  Горячая клавиша: {save_training}
        4. Повторите шаги 1-3, чтобы собрать достаточное количество данных для каждой цифры (рекомендуется не менее 50 изображений каждой цифры).
        5. Нажмите кнопку "Обучить модель".  Начнется процесс обучения модели на собранных данных.  Индикатор прогресса покажет ход обучения.  *Во время обучения интерфейс может быть временно заблокирован*. Горячая клавиша: {train_model}

        **Важно:**
        - Перед обучением модели убедитесь, что в папке 'data' есть достаточное количество изображений для каждой цифры.
        - Чем больше данных, тем лучше будет работать модель.
        - Во время обучения модели не закрывайте приложение.

        **Настройка горячих клавиш:**

        Приложение позволяет настроить горячие клавиши для быстрого доступа к наиболее часто используемым функциям.

        1. Нажмите кнопку "Настроить горячие клавиши".
        2. В открывшемся окне для каждой функции будет указана текущая горячая клавиша и кнопка "Изменить".
        3. Нажмите кнопку "Изменить" для функции, горячую клавишу которой вы хотите изменить.
        4. В появившемся диалоговом окне введите новое сочетание клавиш (например, "<Control-Shift-A>").
        5. Нажмите "OK".
        6. Новая горячая клавиша будет применена немедленно.

        **Примечание:**
        - Горячие клавиши сохраняются в файле 'hotkeys.json' и будут автоматически загружены при следующем запуске приложения.
        - Если файл 'hotkeys.json' не найден или поврежден, будут использованы настройки по умолчанию.
        """

        help_text = help_text.format(
            clear=self.hotkey_bindings["clear"],
            predict=self.hotkey_bindings["predict"],
            save=self.hotkey_bindings["save"],
            load_image=self.hotkey_bindings["load_image"],
            save_training=self.hotkey_bindings["save_training"],
            train_model=self.hotkey_bindings["train_model"]
        )

        # Вместо Label используем Text и Scrollbar для лучшего отображения
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # Только для чтения

        scrollbar = tk.Scrollbar(help_window, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)



    def show_tooltip(self, widget, text):
        def on_enter(event):
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.withdraw()
            self.tooltip.overrideredirect(True)
            self.tooltip_label = Label(self.tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1,
                                       font=("Helvetica", 9))
            self.tooltip_label.pack()
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            self.tooltip.wm_geometry(f"+{x}+{y}")
            self.tooltip.deiconify()

        def on_leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

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

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            try:
                img = Image.open(file_path).convert('L')
                img = img.resize((self.canvas_width, self.canvas_height))
                self.canvas.delete("all")  # Очистить холст перед загрузкой нового изображения
                self.canvas.image = ImageTk.PhotoImage(img)  # сохранить ссылку!
                self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')
                self.last_x = 1
                self.last_y = 1
                # Обновить превью, если оно было
                img_small = img.resize((50, 50))
                photo = ImageTk.PhotoImage(img_small)
                self.thumbnail_label.config(image=photo)
                self.thumbnail_label.image = photo

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")

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
            self.update_gui_after("Ошибка", "Пожалуйста, нарисуйте цифру или загрузите изображение.")
            return

        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height
        try:
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

        except Exception as e:
            self.update_gui_after("Ошибка", f"Ошибка во время распознавания: {e}")

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
        try:
            img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

            saved_image = save_drawing(img, int(digit))

            if saved_image:
                saved_image.thumbnail((50, 50))
                photo = ImageTk.PhotoImage(saved_image)
                self.thumbnail_label.config(image=photo)
                self.thumbnail_label.image = photo

                messagebox.showinfo("Сохранено", "Изображение сохранено для обучения.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении: {e}")

    def save_png(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG files", "*.png")])
        if file_path:
            x = self.master.winfo_rootx() + self.canvas.winfo_x()
            y = self.master.winfo_rooty() + self.canvas.winfo_y()
            x1 = x + self.canvas_width
            y1 = y + self.canvas_height
            try:
                img = ImageGrab.grab().crop((x, y, x1, y1))
                img.save(file_path)
                messagebox.showinfo("Сохранено", "Изображение сохранено.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении: {e}")

    def train_model_thread(self):
        if self.model is None:
            self.model = create_model()

        images, labels = load_data()
        if not images.size or not labels.size:
            messagebox.showerror("Ошибка", "Недостаточно данных для обучения.  Пожалуйста, соберите данные, сохраняя рисунки для обучения.")
            return

        self.progress.start()
        self.train_thread = threading.Thread(target=self.train_model_process, args=(images, labels))
        self.train_thread.start()

    def train_model_process(self, images, labels):
        def progress_callback(progress):
            self.master.after(0, lambda: self.progress.config(value=progress))

        history = train_model(self.model, images, labels, progress_callback)

        self.master.after(0, self.training_complete)

    def training_complete(self):
        self.progress.stop()
        self.progress.config(value=0)
        messagebox.showinfo("Обучение завершено", "Обучение модели завершено.")
        self.model = load_model()  # Перезагрузка модели после обучения

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()