
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train , y_train,params):
        

    # Create a Random Forest Classifier
    rf = RandomForestClassifier(**params)

    # Train the model
    model = rf.fit(X_train, y_train)

    return model

