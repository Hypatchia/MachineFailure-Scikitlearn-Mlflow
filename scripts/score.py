from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import numpy as np


def score_model(model,X_test,y_test,y_pred):

    # Calculate error metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracy)
    # Calculate class probabilities
    y_pred_proba = model.predict_proba(X_test)
    print('Class probabilities: \n', y_pred_proba)
    # Calculate classification metrics
    report = classification_report(y_test, y_pred,output_dict=True)
    print('Classification report: \n', report)

    return mae, mse, rmse, r2, accuracy, report