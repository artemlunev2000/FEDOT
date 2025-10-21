import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


if __name__ == '__main__':
    data_path = "airlines_train_regression_10M.csv"
    df = pd.read_csv(data_path)

    y = df["DepDelay"]
    X = df.drop("DepDelay", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_data = X_train.copy()
    train_data["target"] = y_train

    task = Task("reg")

    automl = TabularAutoML(
        task=task,
        timeout=3600,
        cpu_limit=18,
        memory_limit=40,
        reader_params={"n_jobs": 18, "cv": 6}
    )

    pred = automl.fit_predict(train_data, roles={"target": "target"})
    test_pred = automl.predict(X_test)

    rmse_lama = np.sqrt(mean_squared_error(y_test, test_pred.data[:, 0]))
    print(f"RMSE LightAutoML: {rmse_lama:.2f}")
