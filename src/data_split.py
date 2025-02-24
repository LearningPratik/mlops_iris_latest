import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path

# creating a function to split the data into train and test
def data_split(param_yaml_path):

    # opening yaml file with yaml library
    with open(param_yaml_path) as params_file:
        params_yaml = yaml.safe_load(params_file)

    # Path to my original dataset
    # here using params.yaml --> it is a dictionary, I used it's key, refer params.yaml file.
    data = params_yaml['data_source']['data_path']

    # reading the original dataset
    df = pd.read_csv(data)

    # random state for reproducibility, here also using params.yaml
    random_state = params_yaml['base']['random_state']

    # this is test size parameter of train_test_split
    split_ratio = params_yaml['split']['split_ratio']
    
    # splitting the data into train and test
    train, test = train_test_split(df, test_size = split_ratio, random_state = random_state)
    
    # data directory for saving the train.csv, making Path object
    data_dir = Path(params_yaml['split']['dir'])

    # make directory using Path --> dvc/data/split/train.csv
    data_dir.mkdir(parents = True, exist_ok = True)

    train_file_path = Path(params_yaml['split']['train_file'])
    train_data_path = data_dir / train_file_path

    # saving the train file
    train.to_csv(train_data_path, index = False)

    test_data_path = data_dir / Path(params_yaml['split']['test_file'])

    # saving the test file
    test.to_csv(test_data_path, index = False)



if __name__ == '__main__':
    data_split(param_yaml_path = 'params.yaml')