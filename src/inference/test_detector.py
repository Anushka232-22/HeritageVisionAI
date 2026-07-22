from pprint import pprint
from detector import CrackDetector

detector = CrackDetector()

image = r"C:\Users\ACER\Desktop\HeritageVisionAI\data\processed\heritage_yolo\test\images\brick_test44973_resize_jpg.rf.2bbe334a5b14048d168aee3cba6d1c72.jpg"

results = detector.detect(image)

pprint(results)