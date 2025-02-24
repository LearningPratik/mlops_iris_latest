import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# transforming the y label
def label_encode(data_path, target_col):
    df = pd.read_csv(data_path)
    le = LabelEncoder()

    df[target_col] = le.fit_transform(df[target_col])
    return df


if __name__ == '__main__':

    # path to my params.yaml file
    param_yaml_path = 'params.yaml'

    # opening it with yaml library
    with open(param_yaml_path) as f:
        params_yaml = yaml.safe_load(f)

    # Using Path module to specify the path of the specific files
    # data directory --> where the split file is 
    data_dir = Path(params_yaml['split']['dir'])
    train_file_path = Path(params_yaml['split']['train_file'])
    train_data_path = data_dir / train_file_path

    # use label_encode function to transform train.csv
    processed_train_data = label_encode(data_path = train_data_path, target_col = params_yaml['base']['target_col'])
    
    # save this transformed file to new dvc/data/processed directory
    processed_data_dir = Path(params_yaml['process']['dir'])
    processed_data_dir.mkdir(parents = True, exist_ok = True)
    processed_train_data_path = processed_data_dir / train_file_path
    processed_train_data.to_csv(processed_train_data_path, index = False)
    
    # same process for test.csv
    test_file_path = Path(params_yaml['split']['test_file'])
    test_data_path = data_dir / test_file_path
    processed_test_data = label_encode(data_path = test_data_path, target_col = params_yaml['base']['target_col'])
    
    processed_test_data_path = processed_data_dir / test_file_path
    processed_test_data.to_csv(processed_test_data_path, index = False)