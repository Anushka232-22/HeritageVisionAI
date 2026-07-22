import json
from pathlib import Path


def generate_report(
        image_name,
        metrics,
        severity_score,
        risk_level,
        recommendation):

    report = {

        "image": image_name,

        "crack_length":
            metrics["crack_length"],

        "crack_area":
            metrics["crack_area"],

        "crack_density":
            metrics["crack_density"],

        "severity_score":
            severity_score,

        "risk_level":
            risk_level,

        "recommendation":
            recommendation

    }

    output_dir = Path(
        "outputs/reports"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = output_dir / (
        Path(image_name).stem + ".json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print(
        f"Report saved: {output_file}"
    )

    return report