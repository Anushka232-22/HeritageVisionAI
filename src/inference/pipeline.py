import os

from src.inference.detector import CrackDetector
from src.inference.classifier import DamageClassifier

from src.severity.crack_metrics import extract_crack_metrics
from src.severity.severity_score import calculate_severity
from src.severity.report_generator import generate_report


class HeritagePipeline:

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

            # Crack Metrics

            metrics = extract_crack_metrics(crop_path)

            crack_info = {

                "bbox": crack["bbox"],

                "detection_confidence": crack["confidence"],

                "damage_type": classification["class"],

                "damage_confidence": classification["confidence"],

                "metrics": metrics

            }

            pipeline_result["cracks"].append(crack_info)

        # -----------------------------
        # Severity Analysis
        # -----------------------------

        severity = calculate_severity(pipeline_result)

        pipeline_result["severity"] = severity


        # -----------------------------
        # Return detailed result
        # -----------------------------

        return pipeline_result