from typing import Self
import sklearn
import pandas as pd
from IPython.display import display


class SKLearnModelSelection:

    selectors = {
        'grid_search': sklearn.model_selection.GridSearchCV,
        'random_search': sklearn.model_selection.RandomizedSearchCV

    }

    def __init__(self, models: list, selector: str | sklearn.model_selection._search.BaseSearchCV='grid_search',
                 random_state: int | list[int|None] | None = None) -> None:
        self.model_classes = models
        if not isinstance(selector, (str, sklearn.model_selection._search.BaseSearchCV)):
            raise TypeError('Selector must be a string or an instance of sklearn.model_selection.BaseSearchCV') 

        if isinstance(selector, str):
            try:
                self.selector = self.selectors[selector]
            except KeyError:
                raise TypeError(
                    f'Desired selector is not within the available selectors. They are:\n{*self.selectors.keys(), }')
        else:
            self.selector = selector

        self.random_state = random_state
        self.best_model = None
        self.best_params = None
    
    def compile(self, params,
                cv: int | None = 5,
                n_jobs: int = -1) -> Self:
        self.params = params
        self.cv = cv
        self.n_jobs = n_jobs

        return self
    
    def __fit_once(self, model, param):
        selector = self.selector(
            model,
            param_grid=param,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )
        selector.fit(self.X_train, self.y_train)
        if self.verbose:
            self.__verbose_msg(model, selector)
            
        if self.keep_all_models:
            self.all_models_.append(
                (type(model).__name__, selector.best_estimator_, selector.best_params_, selector.best_score_))
        if selector.best_score_ > self.best_score:
            self.best_random_state = model.random_state
            self.best_score = selector.best_score_
            self.best_model = selector.best_estimator_
            self.best_params = selector.best_params_
    
    def __fit_loop(self) -> Self:
        models = []
        if isinstance(self.random_state, list):
            for random_state in self.random_state:
                models.extend([model(random_state=random_state) for model in self.model_classes])
            self.params *= len(self.random_state)
        else:
            models = [model(random_state=self.random_state) for model in self.model_classes]
        for model, param in zip(models, self.params):
            self.__fit_once(model, param)
        if self.keep_all_models: 
            self.results = pd.DataFrame(self.all_models_, columns=['Model', 'Best Estimator', 'Best params', 'Best score'])
        return self
    
    def __verbose_msg(self, model, selector):
        display(pd.DataFrame({
            'Model': [type(model).__name__],
            'Best estimator': [selector.best_estimator_],
            'Best params': [selector.best_params_],
            'Best score': [selector.best_score_]
        }),)
        print('\n' + '=' * 150)
        
    def fit(self, X_train, y_train, scoring: str = 'accuracy', keep_all_models: bool=False, verbose=True) -> Self:
        self.X_train = X_train
        self.y_train = y_train

        self.scoring = scoring
        self.best_score = 0
        self.keep_all_models = keep_all_models
        self.verbose = verbose
        if keep_all_models:
            self.all_models_ = []
        return self.__fit_loop()
    
    def build_best_model(self): 
        return type(self.best_model)(random_state=self.best_random_state, **self.best_params)
