import json

from visualize import Visualizer

report = json.load(
    open(
        "outputs/reports/brick_test44973_resize_jpg.json"
    )
)

viz = Visualizer()

output = viz.draw_predictions(
    "data/processed/heritage_yolo/test/images/brick_test44973_resize_jpg.rf.2bbe334a5b14048d168aee3cba6d1c72.jpg",
    report
)

print(output)