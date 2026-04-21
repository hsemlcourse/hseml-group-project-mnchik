import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Hydrology_Distance_Euclidean"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    df["Mean_Hillshade"] = (
        df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]
    ) / 3

    df["Road_Fire_Distance_Diff"] = (
        df["Horizontal_Distance_To_Roadways"]
        - df["Horizontal_Distance_To_Fire_Points"]
    )

    df["Elevation_Slope"] = df["Elevation"] * df["Slope"]

    df["Hydrology_Road_Ratio"] = (
        df["Horizontal_Distance_To_Hydrology"] + 1
    ) / (df["Horizontal_Distance_To_Roadways"] + 1)

    return df