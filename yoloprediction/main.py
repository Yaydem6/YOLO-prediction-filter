import os
import cv2
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path, min_area_ratio=0.001, max_area_ratio=0.9):
        # YOLO modelini yükle
        self.model = YOLO(model_path)
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def _process_results(self, frame, results, frame_width, frame_height):
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = ((x2 - x1) / frame_width) * ((y2 - y1) / frame_height)

                if box_area < self.min_area_ratio or box_area > self.max_area_ratio:
                    continue

                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = self.model.names[class_id]

                # Kutuları ve etiketleri çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)

    def predict_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: The file '{image_path}' does not exist.")
            return None

        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load the image '{image_path}'.")
            return None

        frame_height, frame_width = image.shape[:2]
        frame_area = 1  # Normalized image area

        # Tahmin yap
        results = self.model(image)

        # Sonuçları işle
        self._process_results(image, results, frame_width, frame_height)

        return image

    def predict_video(self, video_path, output_path):
        if not os.path.exists(video_path):
            print(f"Error: The file '{video_path}' does not exist.")
            return None

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]
            frame_area = 1  # Normalized image area

            # Tahmin yap
            results = self.model(frame)

            # Sonuçları işle
            self._process_results(frame, results, frame_width, frame_height)

            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

            out.write(frame)

        cap.release()
        out.release()
        print(f"Output saved to {output_path}")

    def process_folder_images(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                image_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, f"output_{filename}")

                processed_image = self.predict_image(image_path)
                if processed_image is not None:
                    cv2.imwrite(output_path, processed_image)
                    print(f"Output saved to {output_path}")

        print("All images processed.")

    def process_folder_videos(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mov"):
                video_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, f"output_{filename}")

                self.predict_video(video_path, output_path)

        print("All videos processed.")


# Kullanım örneği
model_path = ""  # Modelin yolunu girin
input_folder_images = ""  # Fotoğraf klasörü yolunu girin
output_folder_images = ""  # Fotoğraf çıktı klasörü yolunu girin

input_folder_videos = ""  # Video klasörü yolunu girin
output_folder_videos = ""  # Video çıktı klasörü yolunu girin

predictor = YOLOModel(model_path, min_area_ratio=0.001, max_area_ratio=0.9)
predictor.process_folder_images(input_folder_images, output_folder_images)
predictor.process_folder_videos(input_folder_videos, output_folder_videos)
