from sklearn.metrics import accuracy_score

def evaluate_model(model, X, y):
    # Ensure data is computed
    X = X.compute()
    y = y.compute()

    print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}")

    # Defensive check to prevent shape mismatch
    if X.shape[1] != model.coef_.shape[0]:
        raise ValueError(
            f"❌ Feature mismatch: Model expects {model.coef_.shape[0]} features, "
            f"but got {X.shape[1]} in input."
        )

    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    print(f"✅ Model Accuracy: {acc:.2f}")
    return acc

