import dask.dataframe as dd
from pipeline.config import DATA_PATH

def load_and_preprocess():
    df = dd.read_csv(DATA_PATH)
    df = df.dropna()
    df = df[df["feature1"] > 0]
    return df

