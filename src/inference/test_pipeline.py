import json

from src.inference.pipeline import HeritagePipeline


pipeline = HeritagePipeline()

results = pipeline.analyze(

    r"C:\Users\ACER\Desktop\HeritageVisionAI\data\processed\heritage_yolo\test\images\brick_test44973_resize_jpg.rf.2bbe334a5b14048d168aee3cba6d1c72.jpg"

)

print(json.dumps(results, indent=4))