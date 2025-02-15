import os
from PIL import Image
import numpy as np

def create_data_directory(data_dir, num_classes):
    """
    Создает директорию для сохранения данных, если она не существует.
    Внутри создается структура папок для каждой цифры (0-9), чтобы хранить изображения, соответствующие этим цифрам.
    """
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            for i in range(num_classes):
                os.makedirs(os.path.join(data_dir, str(i)))
        except OSError as e:
            print(f"Ошибка при создании директории: {e}")
            return False
    return True

def save_drawing(image, label, data_dir, num_classes):
    """
    Сохраняет изображение в указанную папку, соответствующую метке (цифре).
    """
    if not create_data_directory(data_dir, num_classes):
        return None
    try:
        count = len(os.listdir(os.path.join(data_dir, str(label))))
        filename = os.path.join(data_dir, str(label), f"{count}.png")
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

def load_data(data_dir, image_size, num_classes):
    """
    Загружает данные из директории DATA_DIR и подготавливает их для обучения.
    """
    images = []
    labels = []
    try:
        for i in range(num_classes):
            digit_dir = os.path.join(data_dir, str(i))
            for filename in os.listdir(digit_dir):
                if filename.endswith(".png"):
                    img_path = os.path.join(digit_dir, filename)
                    try:
                        img = Image.open(img_path).convert('L')
                        img = img.resize((image_size, image_size))
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
