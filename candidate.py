"""Current experimental message classifier."""

MAINTENANCE_KEYWORDS = [
    "maintenance",
    "planned",
    "outage",
    "scheduled",
    "downtime",
    "change window"
]

INCIDENT_KEYWORDS = [
    "error",
    "errors",
    "outage",
    "down",
    "failed",
    "failure",
    "incident",
]

def predict(text):
    normalized_text = text.lower()

    for keyword in MAINTENANCE_KEYWORDS:
        if keyword in normalized_text:
            return "maintenance"

    for keyword in INCIDENT_KEYWORDS:
        if keyword in normalized_text:
            return "incident"

    return "non_incident"