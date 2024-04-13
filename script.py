
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np  # for np.sqrt
import sklearn
import joblib
import argparse
import os
import pandas as pd


#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error
#import sklearn
#import joblib
#import argparse
#import os
#import pandas as pd
    
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
    
if __name__ == "__main__":

    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args, _ = parser.parse_known_args()
    
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)

    print("[INFO] Reading data")
    print()
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    
    features = list(train_df.columns)
    label = features.pop(-3)  # Assuming the last column is the label for regression
    
    print("Building training and testing datasets")
    print()
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print('Column order: ')
    print(features)
    print()
    
    print("Label column is: ", label)
    print()
    
    print("Data Shape: ")
    print()
    print("---- SHAPE OF TRAINING DATA (85%) ----")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("---- SHAPE OF TESTING DATA (15%) ----")
    print(X_test.shape)
    print(y_test.shape)
    print()
    
    print("Training RandomForest Model.....")
    model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=args.random_state, verbose=3, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")
    print()

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model persisted at " + model_path)
    print()

    # Evaluate the model using regression metrics
    y_pred_test = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print('Test Mean Squared Error:', test_mse)

    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse) # RMSE is just the square root of MSE
    test_r2 = r2_score(y_test, y_pred_test)
    
    print('Evaluation Metrics for Testing Data:')
    print('------------------------------------')
    print(f'Test Mean Squared Error (MSE): {test_mse:.4f}')
    print(f'Test Mean Absolute Error (MAE): {test_mae:.4f}')
    print(f'Test Root Mean Squared Error (RMSE): {test_rmse:.4f}')
    print(f'Test R^2 Score: {test_r2:.4f}')
