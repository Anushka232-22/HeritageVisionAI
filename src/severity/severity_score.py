def compute_severity(length, area, density):

    # Normalize values

    length_score = min(length / 10000, 1.0)

    area_score = min(area / 50000, 1.0)

    density_score = min(density / 0.30, 1.0)

    final_score = (
        0.4 * length_score +
        0.3 * area_score +
        0.3 * density_score
    ) * 100

    return round(final_score, 2)


def risk_level(score):

    if score < 25:
        return "LOW"

    elif score < 50:
        return "MODERATE"

    elif score < 75:
        return "HIGH"

    else:
        return "CRITICAL"


def recommendation(level):

    recommendations = {

        "LOW":
        "Monitor annually",

        "MODERATE":
        "Schedule inspection within 6 months",

        "HIGH":
        "Detailed structural assessment recommended",

        "CRITICAL":
        "Immediate restoration intervention required"

    }

    return recommendations[level]


# -------------------------------------------------------
# Main function used by the pipeline
# -------------------------------------------------------

def calculate_severity(pipeline_result):

    total_length = 0
    total_area = 0
    total_density = 0

    for crack in pipeline_result["cracks"]:

        metrics = crack["metrics"]

        total_length += metrics["crack_length"]
        total_area += metrics["crack_area"]
        total_density += metrics["crack_density"]

    if pipeline_result["num_cracks"] > 0:
        avg_density = total_density / pipeline_result["num_cracks"]
    else:
        avg_density = 0

    score = compute_severity(
        total_length,
        total_area,
        avg_density
    )

    level = risk_level(score)

    return {

        "crack_length": round(total_length, 2),

        "crack_area": round(total_area, 2),

        "crack_density": round(avg_density, 4),

        "severity_score": score,

        "risk_level": level,

        "recommendation": recommendation(level)

    }