import cv2
import tensorflow as tf
import numpy as np
import os
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self, model_path, image_size, confidence_threshold, similarity_threshold):
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise Exception(f"Ошибка при загрузке модели: {e}")

        self.detector = MTCNN()
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold

    def preprocess_image(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype('float32') / 255.0
        return image

    def get_face_embedding(self, face_image):
        processed_face = self.preprocess_image(face_image)
        processed_face = np.expand_dims(processed_face, axis=0)
        embedding = self.model.predict(processed_face)
        return embedding

    def load_face_database(self, faces_directory):
        face_embeddings = {}
        for student_id in os.listdir(faces_directory):
            student_dir = os.path.join(faces_directory, student_id)
            if os.path.isdir(student_dir):
                face_embeddings[student_id] = []
                for filename in os.listdir(student_dir):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            image_path = os.path.join(student_dir, filename)
                            face_image = cv2.imread(image_path)
                            if face_image is None:
                                continue
                            embedding = self.get_face_embedding(face_image)
                            face_embeddings[student_id].append(embedding)
                        except Exception as e:
                            print(f"Ошибка при обработке изображения: {e}")
        return face_embeddings

    def recognize_face(self, face_embedding, face_database):
        best_match_id = None
        best_similarity = -1
        for student_id, embeddings in face_database.items():
            for known_embedding in embeddings:
                similarity = cosine_similarity(face_embedding, known_embedding)[0][0]
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = student_id
        return best_match_id, best_similarity

    def process_frame(self, frame, face_database):
        results = self.detector.detect_faces(frame)
        for result in results:
            bounding_box = result['box']
            confidence = result['confidence']
            if confidence > self.confidence_threshold:
                x, y, w, h = bounding_box
                x, y, w, h = abs(x), abs(y), abs(w), abs(h)
                face = frame[y:y + h, x:x + w]
                try:
                    if face.size > 0:
                        face_embedding = self.get_face_embedding(face)
                        student_id, similarity = self.recognize_face(face_embedding, face_database)
                        if student_id:
                            label = f"ID: {student_id} (Похожесть: {similarity:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = "Неизвестно"
                            color = (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        print("Предупреждение: Лицо не извлечено.")
                except Exception as e:
                    print(f"Ошибка при обработке лица: {e}")
        return frame
