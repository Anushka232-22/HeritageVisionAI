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