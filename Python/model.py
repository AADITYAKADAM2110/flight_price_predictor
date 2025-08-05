from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from preprocess import preprocess_data

def model_training():
    dataset = preprocess_data()
    X = dataset.drop('price', axis=1)
    y = dataset['price']    

    X_train, X_test, y_train, y_test = train_test_split(dataset.drop('price', axis=1), dataset['price'], test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = {"LinearRegression": LinearRegression(),
             "DecisionTree": DecisionTreeRegressor(),
             "RandomForest": RandomForestRegressor()}
    results = {}

    for name, model_instance in model.items():
        print(f"Training {name} model...")
        model_instance.fit(X_train_scaled, y_train)
        y_pred = model_instance.predict(X_test_scaled)
        print(f"{name} model trained successfully.")
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
    return results


