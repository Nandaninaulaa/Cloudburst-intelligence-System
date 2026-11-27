
def risk_label(probability):
    """Return the cloudburst risk category."""
    if probability > 85:
        return "SEVERE"
    if probability > 65:
        return "HIGH"
    if probability > 40:
        return "MODERATE"
    return "LOW"
