import cv2
import os


import os
import cv2


class Visualizer:

    def __init__(self):

        self.output_dir = "outputs/visualizations"

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    def draw_predictions(self, image_path, report):

        image = cv2.imread(image_path)

        for crack in report["cracks"]:

            x1, y1, x2, y2 = crack["bbox"]

            label = (
                f'{crack["damage_type"]} '
                f'{crack["damage_confidence"]:.2f}'
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1,max(25,y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        image_name = report.get("image_name", report.get("image"))

        output_path = os.path.join(
            self.output_dir,
            os.path.basename(image_name)
        )

        cv2.imwrite(output_path,image)

        return output_path