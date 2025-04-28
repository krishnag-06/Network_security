from networksecurity.exceptions.exceptions import CustomException
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

def get_classification_score(y_true, ypred) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, ypred)
        model_recall_score = recall_score(y_true, ypred)
        model_precision_score = precision_score(y_true, ypred)

        classification_metric = ClassificationMetricArtifact(f1_score= model_f1_score, recall_score= model_recall_score, precision_score= model_precision_score)
        return classification_metric
    except Exception as e:
        raise CustomException(e,sys)

