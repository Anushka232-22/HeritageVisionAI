import os

from src.inference.detector import CrackDetector
from src.inference.classifier import DamageClassifier

from src.severity.crack_metrics import extract_crack_metrics
from src.severity.severity_score import calculate_severity
from src.severity.report_generator import generate_report


class HeritagePipeline:

    CONFIDENCE_THRESHOLD = 0.6  # below this, classification is marked "Uncertain"

    def __init__(self):

        self.detector = CrackDetector()
        self.classifier = DamageClassifier()

    def analyze(self, image_path):

        # -----------------------------
        # Crack Detection
        # -----------------------------

        detection_result = self.detector.detect(image_path)

        pipeline_result = {
            "image_name": detection_result["image_name"],
            "num_cracks": detection_result["num_cracks"],
            "cracks": []
        }

        # -----------------------------
        # Process every detected crack
        # -----------------------------

        for crack in detection_result["detections"]:

            crop_path = crack["crop_path"]

            # Damage Classification

            classification = self.classifier.classify(crop_path)

            # Apply confidence threshold — don't trust low-confidence class calls

            if classification["confidence"] < self.CONFIDENCE_THRESHOLD:
                damage_type = "Uncertain"
            else:
                damage_type = classification["class"]

            # Skip drawing/reporting boxes the classifier says are healthy surface,
            # even if confidence is high — these are false-positive detector boxes,
            # not real damage, and shouldn't appear in the report at all.

            if damage_type in ("Normal", "Healthy_Surface"):
                continue

            # Crack Metrics

            metrics = extract_crack_metrics(crop_path)

            crack_info = {

                "bbox": crack["bbox"],

                "detection_confidence": crack["confidence"],

                "damage_type": damage_type,

                "damage_confidence": classification["confidence"],

                "metrics": metrics

            }

            pipeline_result["cracks"].append(crack_info)

        # -----------------------------
        # Update crack count to reflect filtered results
        # -----------------------------

        pipeline_result["num_cracks"] = len(pipeline_result["cracks"])

        # -----------------------------
        # Severity Analysis
        # -----------------------------

        severity = calculate_severity(pipeline_result)

        pipeline_result["severity"] = severity


        # -----------------------------
        # Return detailed result
        # -----------------------------

        return pipeline_result