from pathlib import Path
import pandas as pd


def load_pollen_seasons(data_dir, selected_pollen=None):
    data_dir = Path(data_dir)

    ps_files = sorted(data_dir.glob("ps_*.csv"))

    if not ps_files:
        raise ValueError(
            f"No pollen season files found with pattern 'ps_*.csv' in {data_dir}"
        )

    ps_dfs = []

    for file in ps_files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()

        required_cols = ["type", "seasons", "st.jd", "en.jd"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{file.name} is missing columns: {missing}")

        df["type"] = df["type"].astype(str).str.strip().str.lower()
        df["seasons"] = pd.to_numeric(df["seasons"], errors="coerce")
        df["st.jd"] = pd.to_numeric(df["st.jd"], errors="coerce")
        df["en.jd"] = pd.to_numeric(df["en.jd"], errors="coerce")

        ps_dfs.append(df[["type", "seasons", "st.jd", "en.jd"]])

    ps_df = pd.concat(ps_dfs, ignore_index=True)
    ps_df = ps_df.dropna(subset=["type", "seasons", "st.jd", "en.jd"]).copy()

    if selected_pollen is not None:
        selected_pollen = [p.strip().lower() for p in selected_pollen]
        ps_df = ps_df[ps_df["type"].isin(selected_pollen)].copy()

    ps_df = ps_df.drop_duplicates(subset=["type", "seasons"]).reset_index(drop=True)

    return ps_df


def add_pollen_season_flags(df, ps_df, selected_pollen):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if "date" not in df.columns:
        raise ValueError("'date' column not found in merged dataframe")

    df["date"] = pd.to_datetime(df["date"])
    df["season_year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear

    selected_pollen = [p.strip().lower() for p in selected_pollen]

    for pollen_type in selected_pollen:
        season_info = ps_df[ps_df["type"] == pollen_type].copy()

        if season_info.empty:
            raise ValueError(f"No pollen season data found for '{pollen_type}'")

        season_map = season_info.rename(
            columns={
                "seasons": "season_year",
                "st.jd": "st_jd",
                "en.jd": "en_jd",
            }
        )[["season_year", "st_jd", "en_jd"]]

        df = df.merge(season_map, on="season_year", how="left")

        df[f"{pollen_type}_in_season"] = (
            (df["day_of_year"] >= df["st_jd"]) &
            (df["day_of_year"] <= df["en_jd"])
        ).astype(int)

        df = df.drop(columns=["st_jd", "en_jd"], errors="ignore")

    df = df.drop(columns=["season_year", "day_of_year"], errors="ignore")

    return df


def load_and_merge_data(data_dir, selected_pollen=None):
    data_dir = Path(data_dir)

    pollen_files = [
        f for f in data_dir.glob("*.csv")
        if "pollen" in f.name.lower() and not f.name.lower().startswith("ps_")
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

    ps_df = load_pollen_seasons(data_dir, selected_pollen=selected_pollen)
    df = add_pollen_season_flags(df, ps_df, selected_pollen=selected_pollen)

    return df