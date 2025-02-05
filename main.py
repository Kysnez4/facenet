import cv2
import os
from src import face_recognition as fr

# Конфигурация
MODELS_DIR = 'models'
MODEL_FILENAME = 'efficientnet_model.h5'
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.9
SIMILARITY_THRESHOLD = 0.7
FACES_DIRECTORY = 'students'

def main():
    try:
        model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
        face_recognizer = fr.FaceRecognizer(model_path, IMAGE_SIZE, CONFIDENCE_THRESHOLD, SIMILARITY_THRESHOLD)
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        exit()

    face_database = face_recognizer.load_face_database(FACES_DIRECTORY)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Ошибка: Не удалось открыть видеокамеру.")
        exit()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Ошибка: Не удалось получить кадр.")
            break

        frame = face_recognizer.process_frame(frame, face_database)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
