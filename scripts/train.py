
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train , y_train):
        
    # Train Model 
    params = {
        "n_estimators": 100,
        "max_depth": 3,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "bootstrap": True,
        "oob_score": False,
        "random_state": 42,
    }

    # Create a Random Forest Classifier
    rf = RandomForestClassifier(**params)

    # Train the model
    model = rf.fit(X_train, y_train)

    return model

