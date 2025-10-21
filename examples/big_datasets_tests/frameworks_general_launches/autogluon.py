from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    data_path = "airlines_train_regression_10M.csv"
    df = pd.read_csv(data_path)
    y = df["DepDelay"]
    X = df.drop("DepDelay", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = X_train.copy()
    train_data['target'] = y_train

    predictor = TabularPredictor(
        label='target',
        problem_type='regression',
        eval_metric='root_mean_squared_error'
    ).fit(
        train_data=train_data,
        presets='best_quality',
        time_limit=2000
    )

    predictions = predictor.predict(X_test)

    rmse_autogluon = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE AutoGluon: {rmse_autogluon:.2f}")
