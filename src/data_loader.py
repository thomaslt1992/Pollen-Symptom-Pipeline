import pandas as pd


def load_and_merge_data(data_dir):
    pollen = pd.read_csv(data_dir / "pollen_berlin_all.csv")
    ps_birch = pd.read_csv(data_dir / "ps_berlin_birch_final.csv")
    ps_poac = pd.read_csv(data_dir / "ps_berlin_poac_final.csv")
    symptoms = pd.read_csv(data_dir / "berlin_symptoms.csv")

    _ = pd.concat([ps_birch, ps_poac], axis=0)

    df = pd.merge(pollen, symptoms, on="date", how="outer")
    return df