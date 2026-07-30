import os

from src.inference.pipeline import HeritagePipeline
from src.severity.report_generator import generate_report
from src.visualization.visualize import Visualizer


pipeline = HeritagePipeline()

visualizer = Visualizer()


def inspect(image_path):

    report = pipeline.analyze(image_path)

    json_path = generate_report(report)

    image_path = visualizer.draw_predictions(
        image_path,
        report
    )

    print("="*50)
    print("HERITAGE INSPECTION COMPLETE")
    print("="*50)

    print()

    print("Image :", report["image_name"])

    print("Cracks:", report["num_cracks"])

    print()

    for i, crack in enumerate(report["cracks"],1):

        print(f"Crack {i}")

        print("Damage :", crack["damage_type"])

        print("Detection Confidence :", crack["detection_confidence"])

        print("Classification Confidence :", crack["damage_confidence"])

        print("Length :", crack["metrics"]["crack_length"])

        print("Area :", crack["metrics"]["crack_area"])

        print("Density :", crack["metrics"]["crack_density"])

        print()

    print("JSON Report :", json_path)

    print("Visualization :", image_path)


if __name__ == "__main__":

    inspect(
        r"C:\Users\ACER\Desktop\HeritageVisionAI\data\processed\heritage_yolo\test\images\stone_test661234_resize_jpg.rf.155a14574870b864e14a45ef3540e48d.jpg"
    )