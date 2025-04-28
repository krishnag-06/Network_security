import os
import sys

import mlflow.sklearn
from networksecurity.exceptions.exceptions import CustomException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact
from networksecurity.entity.artifact_entity import ModelTrainerArtifact
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.utils.main_utils.utils import load_numpy_array_data,load_object,save_object,evaluate_models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

import mlflow

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def track_ml_flow(self, best_model, classificationmetric):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.sklearn.log_model(best_model,"model")
        
    def train_model(self,xtrain, ytrain, xtest, ytest):
        models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boost": GradientBoostingClassifier(),
            "Adaboost": AdaBoostClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN" : KNeighborsClassifier(),
            "SVM": SVC(),
            "Logistic Regression": LogisticRegression()
        }

        params = {
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                'splitter':['best','random'],
                'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                'criterion':['gini', 'entropy', 'log_loss'],
                
                'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boost":{
                'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                'criterion':['squared_error', 'friedman_mse'],
                'max_features':['sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "SVM":{},
            "Adaboost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            },
            "KNN":{"algorithm": ["ball_tree", "kd_tree", "brute"],
                   "n_neighbors": [1,2,3,4,5,6,7,8,9,10]}   
        }

        model_report = evaluate_models(Xtrain= xtrain, ytrain= ytrain, Xtest= xtest, ytest= ytest, models= models, params= params)
        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

        best_model = models[best_model_name]
        best_model.fit(xtrain, ytrain)

        ytrain_pred = best_model.predict(xtrain)

        classification_train_metric = get_classification_score(y_true= ytrain, ypred= ytrain_pred)
        
        self.track_ml_flow(best_model, classification_train_metric)
        ytest_pred = best_model.predict(xtest)
        
        classification_test_metric = get_classification_score(y_true= ytest, ypred= ytest_pred)

        self.track_ml_flow(best_model, classification_test_metric)

        preprocessor = load_object(file_path= self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok= True)

        Network_model = NetworkModel(preprocessor= preprocessor, model= best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj= Network_model)

        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path= self.model_trainer_config.trained_model_file_path,
                                                      train_metric_artifact= classification_train_metric,
                                                      test_metric_artifact= classification_test_metric)
        
        logging.info(f"Model trainer Artifact : {model_trainer_artifact}")
        return model_trainer_artifact
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test,)
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e,sys)

    