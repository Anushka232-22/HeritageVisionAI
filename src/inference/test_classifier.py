from pprint import pprint

from src.inference.classifier import DamageClassifier


classifier = DamageClassifier()

result = classifier.classify(

    "outputs/crops/crack_1.jpg"

)

pprint(result)