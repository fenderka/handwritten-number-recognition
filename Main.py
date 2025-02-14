# ------------------------------
#  Импортирование библиотек
# ------------------------------
import tkinter as tk
from tkinter import messagebox, filedialog  # Для отображения сообщений и работы с диалогами
from tkinter.ttk import Button, Label, Progressbar  # Для работы с элементами управления в интерфейсе
from PIL import Image, ImageDraw, ImageGrab  # Для работы с изображениями
import numpy as np  # Для работы с массивами данных
import tensorflow as tf  # Для работы с нейронными сетями
from tensorflow.keras.models import Sequential  # Для создания модели нейронной сети
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Для создания слоев нейронной сети
from tensorflow.keras.utils import to_categorical  # Для преобразования меток в формат one-hot encoding
import os  # Для работы с файловой системой

# ------------------------------
#  Параметры
# ------------------------------
IMAGE_SIZE = 28  # Размер изображений для обучения
NUM_CLASSES = 10  # Количество классов (цифры от 0 до 9)
DATA_DIR = 'data'  # Папка для сохранения данных
MODEL_FILE = 'digit_recognizer.h5'  # Файл для сохранения модели
LEARNING_RATE = 0.001  # Скорость обучения
EPOCHS = 100  # Количество эпох обучения
BATCH_SIZE = 64  # Размер пакета для обучения

# ------------------------------
#  Функции для работы с данными
# ------------------------------

def create_data_directory():
    """Создает директорию для сохранения данных, если она не существует."""
    if not os.path.exists(DATA_DIR):  # Проверяем, существует ли папка с данными
        os.makedirs(DATA_DIR)  # Создаем папку для данных
        for i in range(NUM_CLASSES):  # Для каждого класса (цифры от 0 до 9)
            os.makedirs(os.path.join(DATA_DIR, str(i)))  # Создаем подпапку для каждой цифры

def save_drawing(image, label):
    """Сохраняет изображение в указанную папку."""
    create_data_directory()  # Убедимся, что папка для данных существует
    count = len(os.listdir(os.path.join(DATA_DIR, str(label))))  # Считаем количество изображений в папке
    filename = os.path.join(DATA_DIR, str(label), f"{count}.png")  # Создаем имя файла
    image.save(filename)  # Сохраняем изображение
    print(f"Изображение сохранено как {filename}")  # Выводим сообщение в консоль

def load_data():
    """Загружает данные из директории и подготавливает их для обучения."""
    images = []  # Список для хранения изображений
    labels = []  # Список для хранения меток классов
    for i in range(NUM_CLASSES):  # Для каждого класса (цифры от 0 до 9)
        digit_dir = os.path.join(DATA_DIR, str(i))  # Папка с изображениями цифры
        for filename in os.listdir(digit_dir):  # Проходим по всем файлам в папке
            if filename.endswith(".png"):  # Проверяем, что файл - это изображение
                img_path = os.path.join(digit_dir, filename)  # Путь к изображению
                img = Image.open(img_path).convert('L')  # Открываем изображение в оттенках серого
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  # Изменяем размер изображения
                img_array = np.array(img) / 255.0  # Преобразуем изображение в массив и нормализуем
                images.append(img_array)  # Добавляем изображение в список
                labels.append(i)  # Добавляем метку (цифру) в список

    images = np.array(images)  # Преобразуем список изображений в массив
    labels = np.array(labels)  # Преобразуем список меток в массив
    return images, labels  # Возвращаем массивы изображений и меток

# ------------------------------
#  Функции для создания и обучения модели
# ------------------------------

def create_model():
    """Создает сверточную нейронную сеть."""
    model = Sequential([  # Создаем последовательную модель
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),  # Первый сверточный слой
        MaxPooling2D((2, 2)),  # Первый слой подвыборки
        Conv2D(64, (3, 3), activation='relu'),  # Второй сверточный слой
        MaxPooling2D((2, 2)),  # Второй слой подвыборки
        Flatten(),  # Преобразуем 2D в 1D для подачи в полносвязный слой
        Dense(128, activation='relu'),  # Полносвязный слой
        Dense(NUM_CLASSES, activation='softmax')  # Выходной слой с 10 нейронами (по одному для каждой цифры)
    ])
    return model  # Возвращаем модель

def train_model(model, images, labels):
    """Обучает модель на предоставленных данных."""
    labels = to_categorical(labels, num_classes=NUM_CLASSES)  # Преобразуем метки в формат one-hot encoding
    from sklearn.model_selection import train_test_split  # Импортируем функцию для разделения данных
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)  # Разделяем данные на обучающую и валидационную выборки
    x_train = np.expand_dims(x_train, axis=-1)  # Добавляем размерность для канала
    x_val = np.expand_dims(x_val, axis=-1)  # Добавляем размерность для канала
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),  # Компиляция модели с оптимизатором Adam
                  loss='categorical_crossentropy',  # Функция потерь для многоклассовой классификации
                  metrics=['accuracy'])  # Метрика точности
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))  # Обучаем модель

def save_model(model, filename=MODEL_FILE):
    """Сохраняет обученную модель."""
    model.save(filename)  # Сохраняем модель в файл
    print(f"Модель сохранена в {filename}")  # Выводим сообщение о сохранении модели

def load_model(filename=MODEL_FILE):
    """Загружает обученную модель."""
    try:
        model = tf.keras.models.load_model(filename)  # Пытаемся загрузить модель
        print(f"Модель загружена из {filename}")  # Если успешно, выводим сообщение
        return model
    except OSError:
        print("Файл модели не найден. Пожалуйста, обучите модель сначала.")  # Если модель не найдена
        return None  # Возвращаем None, если модель не была загружена

# ------------------------------
#  Графический интерфейс
# ------------------------------

class DigitRecognizerApp:
    """Основной класс для приложения распознавания рукописных цифр."""
    def __init__(self, master):
        self.master = master  # Главное окно приложения
        master.title("Распознавание рукописных цифр")  # Заголовок окна
        master.geometry("450x500")  # Размеры окна

        self.model = load_model()  # Пытаемся загрузить модель при запуске приложения

        # Создание холста для рисования
        self.canvas_width = 250  # Ширина холста
        self.canvas_height = 250  # Высота холста
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white", bd=2, relief="solid")
        self.canvas.pack(pady=10)  # Добавляем холст в окно приложения

        # Подсказка
        self.label = Label(master, text="Нарисуйте цифру здесь", font=("Helvetica", 12, "italic"))
        self.label.pack(pady=5)

        # Кнопки управления
        self.clear_button = Button(master, text="Очистить", command=self.clear_canvas, width=18)
        self.clear_button.pack(side=tk.LEFT, padx=10)

        self.predict_button = Button(master, text="Распознать", command=self.predict_digit, width=18)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.save_button = Button(master, text="Сохранить для обучения", command=self.save_for_training, width=20)
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.train_button = Button(master, text="Обучить модель", command=self.train_model, width=20)
        self.train_button.pack(side=tk.LEFT, padx=10)

        # Ввод цифры для сохранения
        self.digit_var = tk.StringVar(value="0")  # Переменная для хранения цифры, которую вводит пользователь
        self.digit_entry = tk.Entry(master, textvariable=self.digit_var, width=5, font=("Helvetica", 14))
        self.digit_entry.pack(pady=10)

        # Метка для предсказания
        self.prediction_label = Label(master, text="Предсказание: ", font=("Helvetica", 14))
        self.prediction_label.pack(pady=10)

        # Прогресс-бар для обучения
        self.progress = Progressbar(master, length=200, mode="indeterminate")
        self.progress.pack(pady=10)

        # Настройка холста для рисования
        self.canvas.bind("<B1-Motion>", self.paint)  # Привязываем функцию рисования к движению мыши по холсту
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
        self.canvas.delete("all")  # Удаляем все элементы на холсте
        self.last_x = None
        self.last_y = None
        self.prediction_label.config(text="Предсказание: ")  # Очищаем метку с предсказанием

    def predict_digit(self):
        """Распознает цифру, нарисованную на холсте."""
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена. Пожалуйста, обучите модель сначала.")
            return

        # Получаем изображение с холста
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

        # Предобработка изображения
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  # Изменяем размер изображения
        img_array = np.array(img) / 255.0  # Нормализуем изображение
        img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для батча
        img_array = np.expand_dims(img_array, axis=-1)  # Добавляем размерность для канала

        # Предсказание
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)  # Выбираем цифру с максимальной вероятностью

        self.prediction_label.config(text=f"Предсказание: {digit}")  # Выводим предсказание на экран

    def save_for_training(self):
        """Сохраняет нарисованное изображение для обучения."""
        digit = self.digit_var.get()  # Получаем цифру, введенную пользователем
        if not digit.isdigit() or int(digit) < 0 or int(digit) > 9:
            messagebox.showerror("Ошибка", "Пожалуйста, введите цифру от 0 до 9.")  # Проверка на правильность ввода
            return

        # Получаем изображение с холста
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height
        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')

        save_drawing(img, int(digit))  # Сохраняем изображение для обучения
        messagebox.showinfo("Сохранено", "Изображение сохранено для обучения.")  # Подтверждаем сохранение

    def train_model(self):
        """Обучает модель на собранных данных."""
        images, labels = load_data()  # Загружаем данные
        if len(images) == 0:
            messagebox.showerror("Ошибка", "Нет данных для обучения. Сначала соберите данные.")  # Проверка на наличие данных
            return

        self.progress.start()  # Запускаем прогресс-бар
        self.model = create_model()  # Создаем модель
        train_model(self.model, images, labels)  # Обучаем модель
        save_model(self.model)  # Сохраняем обученную модель
        self.progress.stop()  # Останавливаем прогресс-бар
        messagebox.showinfo("Успех", "Модель обучена и сохранена.")  # Подтверждаем успешное обучение

# ------------------------------
#  Запуск приложения
# ------------------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerApp(root)  # Создаем экземпляр приложения
    root.mainloop()  # Запускаем основной цикл приложения
