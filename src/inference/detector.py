from ultralytics import YOLO
import cv2
import os


class CrackDetector:

    def __init__(
        self,
        model_path="models/detector/best_detector.pt"
    ):

        self.model = YOLO(model_path)

        os.makedirs("outputs/crops", exist_ok=True)

    def detect(
        self,
        image_path,
        conf=0.25,
        save=True
    ):

        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=save,
            verbose=False
        )

        image = cv2.imread(image_path)

        detections = []

        crack_id = 1

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])

                crop = image[y1:y2, x1:x2]

                crop_path = f"outputs/crops/crack_{crack_id}.jpg"

                cv2.imwrite(crop_path, crop)

                detections.append({

                    "bbox":[x1,y1,x2,y2],

                    "confidence":round(confidence,4),

                    "crop_path":crop_path

                })

                crack_id += 1

        return {

            "image_name":os.path.basename(image_path),

            "num_cracks":len(detections),

            "detections":detections

        }