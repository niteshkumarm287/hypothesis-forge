"""Original hand-written classifier preserved as the project baseline."""

MAINTENANCE_KEYWORDS = (
    "maintenance",
    "planned",
    "scheduled",
    "change window",
)

INCIDENT_KEYWORDS = (
    "error",
    "errors",
    "outage",
    "downtime",
    "down",
    "failed",
    "failure",
    "incident",
    "pagerduty",
    "stuck",
)


def predict(text: str) -> str:
    """Classify one message using hand-written keyword rules."""
    normalized_text = text.lower()

    for keyword in MAINTENANCE_KEYWORDS:
        if keyword in normalized_text:
            return "maintenance"

    for keyword in INCIDENT_KEYWORDS:
        if keyword in normalized_text:
            return "incident"

    return "non_incident"
