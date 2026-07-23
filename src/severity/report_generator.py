import json
import os


def generate_report(result, save_dir="outputs/reports"):

    os.makedirs(save_dir, exist_ok=True)

    report = {

        "image": result["image_name"],

        "num_cracks": result["num_cracks"],

        "severity": result["severity"],

        "cracks": result["cracks"]

    }

    filename = os.path.join(

        save_dir,

        result["image_name"].split(".")[0] + ".json"

    )

    with open(filename, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Report saved: {filename}")

    return str(filename)