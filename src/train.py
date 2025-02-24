import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pickle

def training(param_yaml_path):
    with open(param_yaml_path) as f:
        params_yaml = yaml.safe_load(f)
    
    # take processed train data from processed directory
    processed_data_dir = Path(params_yaml['process']['dir'])
    train_file_path = Path(params_yaml['process']['train_file'])
    train_data_path = processed_data_dir / train_file_path
    
    # take processed test data from processed directory
    test_file_path = Path(params_yaml['process']['test_file'])
    test_data_path = processed_data_dir / test_file_path
    
    # setting random state for reproducibility
    random_state = params_yaml['base']['random_state']

    # defining target column, took value from params yaml file
    target = [params_yaml['base']['target_col']]
    
    # reading data
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    # defining split data for X, y
    y_train = train[target]
    y_test = test[target]

    X_train = train.drop(target, axis = 1)
    X_test = test.drop(target, axis = 1)

    random_state = params_yaml['base']['random_state']

    # number estimators parameters for Random Forest
    n_est = params_yaml['train']['n_est']
    
    # defining and fitting the train data on Random Forest
    rf = RandomForestClassifier(n_estimators = n_est, random_state = random_state)
    rf.fit(X_train, y_train)

    # saving the model to model directory
    model_dir = Path(params_yaml['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = Path('rf_model_1.pkl')
    with open(model_dir / model_file, 'wb') as f:

        # using pickle library for saving the model
        pickle.dump(rf, f)


if __name__ == '__main__':
    param_yaml_path = 'params.yaml'
    training(param_yaml_path = param_yaml_path)