import yaml
from networksecurity.exceptions.exceptions import CustomException
from networksecurity.logging.logger import logging

import os
import sys
import dill
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)

def write_yaml_file(file_path:str, content: object, replace: bool = False):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok= True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_numpy_array_data(file_path:str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(file_path:str, obj :object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils")
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path:str) -> None:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"the file {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_numpy_array_data(file_path:str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(Xtrain, ytrain, Xtest, ytest, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            gs = GridSearchCV(model, para, cv =3)
            gs.fit(Xtrain,ytrain)
            
            model = gs.best_estimator_
            ytrain_pred = model.predict(Xtrain)
            ytest_pred = model.predict(Xtest)

            train_model_score = r2_score(ytrain, ytrain_pred)
            test_model_score = r2_score(ytest, ytest_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
