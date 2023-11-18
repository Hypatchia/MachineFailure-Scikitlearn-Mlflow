import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from scripts.data_preprocessing import preprocess_data
from scripts.data_exploration import explore_data
from scripts.data_preparation import prepare_data
from scripts.train_model import train_model
from scripts.score_model import score_model

if __name__ == "__main__":

    # Read the data
    data = pd.read_csv("Data/MachineFailureData.csv")
    print("Data Read Successfully")
    # Explore the data
    explore_data(data)
    print("Data Explored Successfully")
    # Preprocess the data
    data = preprocess_data(data)
    print("Data Preprocessed Successfully")
    # Prepare the data
    X_train , X_test, y_train, y_test = prepare_data(data)
    print("Data Prepared Successfully")
    # Train the model
    # Train Model 
    parameters = {
        "n_estimators": 100,
        "max_depth": 3,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "bootstrap": True,
        "oob_score": False,
        "random_state": 42,
    }

    rf_model = train_model(X_train , y_train,parameters)
    print("Model Trained Successfully")
    # Get the parameters of the trained model
    params = rf_model.get_params()

    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    print("Model Evaluated Successfully")
    # Score the model
    mae, mse, rmse, r2, accuracy, classifier_report = score_model(rf_model,X_test,y_test,y_pred)
    print("Model Scored Successfully")
    # Set the experiment name
    experiment_name = "RandomForestClassifier"

    # If this is not set, a unique name will be auto-generated for a run
    run_name = "30% Split"

    # Define an artifact path that the model will be saved to.
    artifact_path = "splits"

    # Check if the experiment already exists; if not, create it
    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        print(f"Experiment '{experiment_name}' already exists.")
    # Set the experiment

    mlflow.set_experiment(experiment_name)

    # Start a run within the specified experiment
    with mlflow.start_run(run_name=run_name) as run:
        print("MLflow:")
        # Log the model parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", classifier_report[str(1)]['precision'])
        mlflow.log_metric("recall",classifier_report[str(1)]['recall'] )
        mlflow.log_metric("f1_score",classifier_report[str(1)]['f1-score'] )
        mlflow.log_metric("support",classifier_report[str(1)]['support'])
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Save the model
        mlflow.sklearn.log_model(rf_model, "rf_model")
