from dask_ml.linear_model import LogisticRegression
import joblib
from pipeline.config import MODEL_PATH

def train_model(df):
    # ✅ Ensure the DataFrame contains required columns
    required_cols = ["feature1", "feature2", "feature3", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ✅ Convert selected columns to Dask arrays
    X = df[["feature1", "feature2", "feature3"]].to_dask_array(lengths=True)
    y = df["label"].to_dask_array(lengths=True)

    # ✅ Train model
    model = LogisticRegression()
    model.fit(X, y)

    # ✅ Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    return model

