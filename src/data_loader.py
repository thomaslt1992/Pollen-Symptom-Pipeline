from pathlib import Path
import pandas as pd


def load_and_merge_data(data_dir):
    data_dir = Path(data_dir)

    pollen_files = [
        f for f in data_dir.glob("*.csv")
        if "pollen" in f.name.lower()
    ]
    symptom_files = [
        f for f in data_dir.glob("*.csv")
        if "symptom" in f.name.lower()
    ]

    if len(pollen_files) != 1:
        raise ValueError(
            f"Expected exactly 1 pollen file, found {len(pollen_files)}: "
            f"{[f.name for f in pollen_files]}"
        )

    if len(symptom_files) != 1:
        raise ValueError(
            f"Expected exactly 1 symptom file, found {len(symptom_files)}: "
            f"{[f.name for f in symptom_files]}"
        )

    pollen = pd.read_csv(pollen_files[0])
    symptoms = pd.read_csv(symptom_files[0])

    pollen.columns = pollen.columns.str.strip().str.lower()
    symptoms.columns = symptoms.columns.str.strip().str.lower()

    if "date" not in pollen.columns:
        raise ValueError(f"'date' column not found in pollen file: {pollen_files[0].name}")

    if "date" not in symptoms.columns:
        raise ValueError(f"'date' column not found in symptom file: {symptom_files[0].name}")

    df = pd.merge(pollen, symptoms, on="date", how="outer")
    return df