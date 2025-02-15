import tkinter as tk
from tkinter.ttk import Style
from gui_utils import DigitRecognizerApp  # Импорт класса из gui_utils.py
import model_utils  # Импорт функций для работы с моделью
import os

# Отключение аппаратного ускорения (если необходимо)
os.environ['TK_SILENCE_DEPRECATION'] = '1'
os.environ['PYGAME_DISPLAY'] = 'x11'


# ------------------------------
# Параметры
# ------------------------------
IMAGE_SIZE = 28  # Размер изображения, к которому приводятся все входные данные.
NUM_CLASSES = 10  # Количество классов (цифры от 0 до 9).
DATA_DIR = 'data'  # Директория для сохранения данных для обучения.
MODEL_FILE = 'digit_recognizer.h5'  # Имя файла для сохранения обученной модели.
LEARNING_RATE = 0.001  # Скорость обучения для оптимизатора Adam.
EPOCHS = 100  # Максимальное количество эпох обучения.
BATCH_SIZE = 64  # Размер пакета данных для обучения.
HOTKEY_CONFIG_FILE = "hotkeys.json"  # Файл для сохранения настроек горячих клавиш


if __name__ == "__main__":
    root = tk.Tk()
    # Используем класс DigitRecognizerApp из gui_utils.py
    app = DigitRecognizerApp(root, image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, data_dir=DATA_DIR,
                             model_file=MODEL_FILE, learning_rate=LEARNING_RATE, epochs=EPOCHS,
                             batch_size=BATCH_SIZE, hotkey_config_file=HOTKEY_CONFIG_FILE)
    root.mainloop()
