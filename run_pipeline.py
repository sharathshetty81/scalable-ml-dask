from pipeline.cluster import start_cluster
from pipeline.data_ingestion import load_and_preprocess
from pipeline.train_model import train_model
from pipeline.evaluate import evaluate_model

if __name__ == "__main__":
    client = start_cluster()

    df = load_and_preprocess()
    df = df.persist()

    model = train_model(df)

    X = df[["feature1", "feature2", "feature3"]].to_dask_array(lengths=True)
    y = df["label"].to_dask_array(lengths=True)

    accuracy = evaluate_model(model, X, y)
    print(f"Model Accuracy: {accuracy:.2f}")

