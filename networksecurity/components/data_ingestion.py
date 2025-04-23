import os 
import sys
from networksecurity.exceptions.exceptions import CustomException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import pandas as pd
import numpy as np
import pymongo
from typing import List
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class Dataingestion:
    def __init__(self,dataingestionconfig: DataIngestionConfig):
        try:
            self.dataingestionconfig = dataingestionconfig

        except Exception as e:
            raise CustomException(e,sys)
        
    def export_collection_as_dataframe(self):
        try:
            database_name = self.dataingestionconfig.database_name
            collection_name = self.dataingestionconfig.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop("_id", axis= 1)

                df.replace({"na":np.nan},inplace = True)
                return df
            
        except Exception as e:
            CustomException(e,sys)

    def export_data_to_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.dataingestionconfig.feature_store_file_path

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok= True)
            dataframe.to_csv(feature_store_file_path,index= False, header=True)
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
        
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_set, test_set = train_test_split(dataframe, test_size= self.dataingestionconfig.train_test_aplit_ratio)

            logging.info("Performed triantest split on the dataframe")

            dir_path = os.path.dirname(self.dataingestionconfig.training_file_path)
            os.makedirs(dir_path, exist_ok= True)

            logging.info("Exporting train and test file path")

            train_set.to_csv(self.dataingestionconfig.training_file_path,index = False, header = True)
            test_set.to_csv(self.dataingestionconfig.test_file_path,index = False,header = True)
            logging.info("Exported trian and test file path")

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_to_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(train_file_path=self.dataingestionconfig.training_file_path,
                                                          test_file_path=self.dataingestionconfig.test_file_path)
            return dataingestionartifact


        except Exception as e:
            raise CustomException(e,sys)