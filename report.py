class Report:
    def __init__(self, model, X_test, y_test) -> None:
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def accuracy(self) -> float:
        return accuracy_score(self.y_test, self.model.predict(self.X_test))
    
    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.y_test, self.model.predict(self.X_test))
    
    def report(self) -> str:
        return classification_report(self.y_test, self.model.predict(self.X_test))
    
    def cross_val_score(self, cv=5) -> list:
        return cross_val_score(self.model, self.X_test, self.y_test, cv=cv)
