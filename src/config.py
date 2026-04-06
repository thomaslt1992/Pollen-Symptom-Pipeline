# HERE ARE THE DEFAULT CONFIGURATIONS FOR THE PROJECT
# THESE CAN BE OVERRIDDEN BY THE USER IN THE CLI OR INSIDE A NOTEBOOK WHEN USED THE LIBRARY

DEFAULT_LAGS = [1, 2, 3, 5, 7]
DEFAULT_WINDOWS = [3, 5, 7]
DEFAULT_SELECTED_POLLEN = ["birch", "poac","ragweed","ambrosia"]
DEFAULT_FORBIDDEN_CURRENT = [
    "symptom_score",
    "standarddeviationwithmedication",
    "averageoverallscorewithoutmedication",
    "standarddeviationwithoutmedication",
    "samples",
]

DAYS = 365 #This is the default number of days to forecast

POLLEN_ALIASES = {
    "poac": "grasses",
    "grass": "grasses",
    "grasses": "grasses",
    "birch": "birch",
}

POLLEN_SEASON_COLORS = {
    "grass": "green",
    "grasses": "green",
    "birch": "blue",
}