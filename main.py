import face_recognition
import cv2
import os
import asyncio
from telegram import Bot
from telegram.error import TelegramError
from datetime import datetime

# Инициализация Telegram бота
TOKEN = "7491361835:AAFcjeMHdHIAAEfC6ckrx6qKQZeXWveAvYI"
USER_ID = 5580150613  # Замените на ваш Telegram ID
bot = Bot(token=TOKEN)

# Загрузка фотографии для сравнения
imgmain = face_recognition.load_image_file('dataset/image_d.jpg')
imgmain = cv2.cvtColor(imgmain, cv2.COLOR_BGR2RGB)

# Получение кодировки лица для эталонного изображения
try:
    encodeElon = face_recognition.face_encodings(imgmain)[0]
    print('Лицо обнаружено')
except IndexError:
    print("Лицо не найдено на изображении для сравнения")
    exit(1)

# Папка для сохранения скриншотов
screenshots_directory = os.path.join(os.getcwd(), "screenshots")
if not os.path.exists(screenshots_directory):
    os.makedirs(screenshots_directory)

# Видеозахват с камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка при открытии веб-камеры.")
    exit(1)

# Функция для отправки фото в Telegram
async def send_photo_to_telegram(photo_path, message):
    with open(photo_path, 'rb') as photo_file:
        try:
            await bot.send_photo(chat_id=USER_ID, photo=photo_file)
            await bot.send_message(chat_id=USER_ID, text=message)
            print(f"Отправлено сообщение и скриншот в Telegram пользователю {USER_ID}")
        except TelegramError as e:
            print(f"Не удалось отправить фото: {e}")

# Основная функция для обработки видео
async def process_video():
    face_present_start_time = None  # Время начала обнаружения лица
    screenshot_taken = False  # Флаг для того, чтобы не делать скриншоты многократно

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка при захвате кадра.")
            break

        # Конвертация изображения из BGR в RGB
        rgb_frame = frame[:, :, ::-1]

        # Поиск всех лиц в текущем кадре
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            if face_present_start_time is None:
                face_present_start_time = datetime.now()

            elapsed_time = (datetime.now() - face_present_start_time).total_seconds()

            if elapsed_time >= 3 and not screenshot_taken:
                screenshot_path = os.path.join(screenshots_directory, f"face_{len(os.listdir(screenshots_directory)) + 1}.png")
                cv2.imwrite(screenshot_path, frame)  # Сохранение скриншота

                # Загрузка и кодирование сохраненного скриншота
                saved_screenshot = cv2.imread(screenshot_path)
                face_locations_test = face_recognition.face_locations(saved_screenshot)
                encodings_test = face_recognition.face_encodings(saved_screenshot)

                match_found = False  # Флаг, чтобы проверить, найдено ли совпадение

                for (top, right, bottom, left), encodeTest in zip(face_locations_test, encodings_test):
                    # Сравнение лиц
                    results = face_recognition.compare_faces([encodeElon], encodeTest)
                    if True in results:
                        label = "Совпадает лицо человека"
                        match_found = True
                    else:
                        label = "Не совпадает лицо человека"

                    # Отрисовка меток на изображении
                    cv2.rectangle(saved_screenshot, (left, top), (right, bottom), (255, 0, 255), 2)
                    cv2.putText(saved_screenshot, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

                # Сохранение скриншота с метками
                labeled_screenshot_path = os.path.join(screenshots_directory, f"labeled_face_{len(os.listdir(screenshots_directory)) + 1}.png")
                cv2.imwrite(labeled_screenshot_path, saved_screenshot)
                print(f"Сохранен скриншот с метками как {labeled_screenshot_path}")

                # Отправка изображения и сообщения в Telegram
                await send_photo_to_telegram(labeled_screenshot_path, label)

                screenshot_taken = True

            # Отрисовка лиц на текущем кадре
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        else:
            face_present_start_time = None  # Сброс времени начала обнаружения лица
            screenshot_taken = False  # Сброс флага скриншота

        # Отображение текущего кадра
        cv2.imshow('Video', frame)

        # Завершение по нажатию Enter
        if cv2.waitKey(25) & 0xFF == 13:
            break

    cap.release()  # Освобождение видеозахвата
    cv2.destroyAllWindows()  # Закрытие всех окон OpenCV

# Запуск обработки видео
asyncio.run(process_video())