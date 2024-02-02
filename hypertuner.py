from typing import Self
import sklearn


class SKLearnModelSelection:

    selectors = {
        'grid_search': sklearn.model_selection.GridSearchCV,
        'random_search': sklearn.model_selection.RandomizedSearchCV

    }

    def __init__(self, models: list, selector: str | sklearn.model_selection.BaseSearchCV='grid_search',
                 random_state: int | None = None) -> None:
        self.model_classes = models
        self.models = [model(random_state=random_state) for model in models]
        if not isinstance(selector, str | sklearn.model_selection.BaseSearchCV):
            raise TypeError('Selector must be a string or an instance of sklearn.model_selection.BaseSearchCV') 

        if isinstance(self.selector, str):
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
    
    def compile(self, X_train,
                y_train, 
                cv: int | None = 5,
                n_jobs: int = -1) -> Self:
        
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.n_jobs = n_jobs

        return self
    
    def __fit_loop(self) -> Self:
        for model, param in zip(self.models, self.params):
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
                    (selector.best_estimator_, selector.best_params_, selector.best_score_))
            else:
                if selector.best_score_ > self.best_score:
                    self.best_score = selector.best_score_
                    self.best_model = selector.best_estimator_
                    self.best_params = selector.best_params_

        return self
    
    def __verbose_msg(self, model, selector):
        print(f'Model: {type(model)}', '=====================', f'Best estimator: {selector.best_estimator_}', f'Best params: {selector.best_params_}',
              f'{selector.best_score_}', sep='\n\n')
        
    def fit(self, params, scoring: str = 'accuracy', keep_all_models: bool=False, verbose=True) -> Self:
        self.params = params
        self.scoring = scoring
        self.best_score = 0
        self.keep_all_models = keep_all_models
        self.verbose = verbose
        if keep_all_models:
            self.all_models_ = []
        return self.__fit_loop()
    
    def build_best_model(self): 
        return type(self.best_model)(random_state=self.random_state, **self.best_params)
