from pathlib import Path

from crack_metrics import extract_crack_metrics

from severity_score import (
    compute_severity,
    risk_level,
    recommendation
)

from report_generator import (
    generate_report
)


# Path to test images
images = list(
    Path(
        "data/processed/heritage_yolo/test/images"
    ).glob("*.jpg")
)

print("\n" + "=" * 60)
print("HERITAGE VISION AI - SEVERITY ANALYSIS")
print("=" * 60)

# Process first 5 images
for img in images[:5]:

    # Extract crack metrics
    metrics = extract_crack_metrics(
        str(img)
    )

    # Compute severity score
    score = compute_severity(
        metrics["crack_length"],
        metrics["crack_area"],
        metrics["crack_density"]
    )

    # Determine risk level
    level = risk_level(score)

    # Generate recommendation
    action = recommendation(level)

    # Generate JSON report
    generate_report(
        img.name,
        metrics,
        score,
        level,
        action
    )

    # Print results
    print("\n" + "=" * 60)

    print("Image:", img.name)

    print(metrics)

    print("Severity Score:", score)

    print("Risk Level:", level)

    print("Recommendation:", action)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)