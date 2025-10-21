import pandas as pd
import numpy as np
import autosklearn.regression
import os


def simple_train_test_split(X, y, test_size=0.2, random_state=42):
    """Простая реализация train_test_split без sklearn, так как auto_sklearn требует старую версию scikit-learn"""
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    return X_train, X_test, y_train, y_test


def mean_squared_error_manual(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def label_encode_dataframe(train_df, test_df):
    X_train = train_df.copy()
    X_test = test_df.copy()

    for col in X_train.columns:
        if X_train[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_train[col]):
            uniques = pd.concat([X_train[col], X_test[col]], axis=0).dropna().unique()
            mapping = {val: idx for idx, val in enumerate(uniques)}
            X_train[col] = X_train[col].map(mapping)
            X_test[col] = X_test[col].map(mapping)
            X_train[col] = X_train[col].fillna(-1)
            X_test[col] = X_test[col].fillna(-1)

    return X_train, X_test


if __name__ == '__main__':
    data_path = "airlines_train_regression_10M.csv"
    df = pd.read_csv(data_path)

    y = df["DepDelay"]
    X = df.drop("DepDelay", axis=1)
    X_train, X_test, y_train, y_test = simple_train_test_split(X, y, test_size=0.2)
    X_train, X_test = label_encode_dataframe(X_train, X_test)

    tmp_dir = os.path.join(os.getcwd(), "autosklearn_tmp")

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=3600,
        n_jobs=7,
        metric=autosklearn.metrics.mean_squared_error,
        seed=42,
        tmp_folder=tmp_dir,
        memory_limit=5000
    )

    automl.fit(X_train, y_train)

    predictions = automl.predict(X_test)

    rmse = np.sqrt(mean_squared_error_manual(y_test, predictions))
    print(f"RMSE Auto-sklearn: {rmse:.2f}")
