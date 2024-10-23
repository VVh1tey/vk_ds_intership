import pandas as pd
import numpy as np

def preprocess(data: pd.DataFrame, mode: str = 'train') -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        mode (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if mode in ['train', 'predict']:
        data['values'] = data['values'].apply(lambda x: np.array(x))
        
        data['mean'] = data['values'].apply(np.mean)
        
        if mode == 'train':
            features = data[~data['mean'].isnull()]
        elif mode == 'predict':
            features = data
        
        features['std'] = features['values'].apply(np.std)
        features['min'] = features['values'].apply(np.min)
        features['max'] = features['values'].apply(np.max)
        features['sum'] = features['values'].apply(np.sum)
        features['range'] = features['max'] - features['min']
        
        features['diff'] = features['values'].apply(lambda x: x[-1] - x[0])
        features['std_diff'] = features['values'].apply(lambda x: np.std(np.diff(x)))

        features['q25'] = features['values'].apply(lambda x: np.percentile(x, 25))
        features['q75'] = features['values'].apply(lambda x: np.percentile(x, 75))

        features['rolling_mean_3'] = features['values'].apply(lambda x: np.mean(pd.Series(x).rolling(window=3).mean()))
        features['rolling_std_3'] = features['values'].apply(lambda x: np.mean(pd.Series(x).rolling(window=3).std()))
        
        features = features.drop(['dates', 'values'], axis=1, errors='ignore')
        columns_to_drop = []

        if "id" in features.columns:
            columns_to_drop.append('id')

        if "label" in features.columns:
            columns_to_drop.append('label')
        features = features.drop(columns=columns_to_drop, axis=1, errors='ignore')   
        
        return features
    else:
        return None