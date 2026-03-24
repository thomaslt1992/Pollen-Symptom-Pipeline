# HERE ARE THE DEFAULT CONFIGURATIONS FOR THE PROJECT
# THESE CAN BE OVERRIDDEN BY THE USER IN THE CLI OR INSIDE A NOTEBOOK WHEN USED THE LIBRARY

DEFAULT_LAGS = [1, 2, 3, 5, 7]
DEFAULT_WINDOWS = [3, 5, 7]

DEFAULT_FORBIDDEN_CURRENT = [
    "averageoverallscorewithmedication",
    "standarddeviationwithmedication",
    "averageoverallscorewithoutmedication",
    "standarddeviationwithoutmedication",
    "samples",
]