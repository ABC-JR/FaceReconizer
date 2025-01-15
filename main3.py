from deepface import DeepFace
import cv2
import os
from datetime import datetime

# Открытие веб-камеры
cap = cv2.VideoCapture(0)

# Чтение кадра с веб-камеры
ret, frame = cap.read()

if not ret:
    print("Не удалось захватить изображение с камеры.")
else:
    # Сохранение захваченного кадра во временный файл
    time =  datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_image_path = str(time) + "user_face.jpg"

    cv2.imwrite(temp_image_path, frame)

    try:
        # Сравнение захваченного изображения с изображением пользователя
        result = DeepFace.verify("user_face.jpg", temp_image_path)

        # Проверка на совпадение лиц
        if result["verified"]:
            print("Face recognized!")
            os.remove(temp_image_path)
        else:
            os.system('shutdown /s /t 0')
    except Exception as e:
        print(f"An error occurred: {e}")


cap.release()

