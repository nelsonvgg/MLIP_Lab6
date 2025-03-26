import pytest
import pandas as pd
import numpy as np
from prediction_demo import data_preparation,data_split,train_model,eval_model

@pytest.fixture
def housing_data_sample():
    return pd.DataFrame(
      data ={
      'price':[13300000,12250000],
      'area':[7420,8960],
    	'bedrooms':[4,4],	
      'bathrooms':[2,4],	
      'stories':[3,4],	
      'mainroad':["yes","yes"],	
      'guestroom':["no","no"],	
      'basement':["no","no"],	
      'hotwaterheating':["no","no"],	
      'airconditioning':["yes","yes"],	
      'parking':[2,3],
      'prefarea':["yes","no"],	
      'furnishingstatus':["furnished","unfurnished"]}
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints has same length
    assert feature_df.shape[0]==len(target_series)

    #Feature only has numerical values
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number,np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split(feature_target_sample):
    return_tuple = data_split(*feature_target_sample)
    # TODO test if the length of return_tuple is 4
    # Check if the return value is a tuple of length 4
    assert isinstance(return_tuple, tuple), "data_split should return a tuple"
    assert len(return_tuple) == 4, "data_split should return a tuple of length 4"
    
    # Unpack the tuple
    X_train, X_test, y_train, y_test = return_tuple
    
    # Check if the splits are pandas DataFrames/Series
    assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame"
    assert isinstance(X_test, pd.DataFrame), "X_test should be a pandas DataFrame"
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series"
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series"
    
    # Check if the lengths of the splits are consistent
    assert len(X_train) == len(y_train), "X_train and y_train should have the same length"
    assert len(X_test) == len(y_test), "X_test and y_test should have the same length"
    
    # Check if the total number of rows matches the original dataset
    total_rows = len(X_train) + len(X_test)
    assert total_rows == feature_target_sample[0].shape[0], "Total rows in splits should match the original dataset"
    #raise NotImplemented