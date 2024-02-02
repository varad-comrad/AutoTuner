from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import numpy as np

class Report:
    def __init__(self, model, X_test, y_test) -> None:
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)

    def accuracy(self) -> float:
        return accuracy_score(self.y_test, self.y_pred)
    
    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.y_test, self.y_pred)
    
    def report(self) -> dict:
        print(classification_report(self.y_test, self.y_pred))
        return classification_report(self.y_test, self.y_pred, output_dict=True)
    
    def cross_val_score(self, cv=5) -> list:
        return cross_val_score(self.model, self.X_test, self.y_test, cv=cv)
