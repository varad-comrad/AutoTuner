from typing import Self


class SKLearnModelSelection:
    def __init__(self, models: list, selector: str='grid_search',
                 random_state: int | None = None) -> None:
        self.model_classes = models
        self.models = [model(random_state=random_state) for model in models]
        self.selector = selector
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
    
    def compile(self, X_train,
                y_train, 
                cv: int | None = 5) -> Self:
        
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        if self.selector == 'grid_search':
            self.selector = GridSearchCV
        elif self.selector == 'random_search':
            self.selector = RandomizedSearchCV
        else:
            raise NotImplementedError('Selector not implemented')
        
        return self
    
    def __fit_grid_search(self, params, scoring, n_jobs, keep_all_models) -> Self:
        for model, param in zip(self.models, self.params):
            selector = self.selector(
                model,
                param_grid=param,
                cv=self.cv,
                scoring=scoring,
                n_jobs=n_jobs
            )
            selector.fit(self.X_train, self.y_train)
            print(f'Model: {type(model)}', '=====================', f'Best estimator: {selector.best_estimator_}', f'Best params: {selector.best_params_}',
                  f'{selector.best_score_}', sep='\n\n')
            if keep_all_models:
                self.all_models_.append(
                    (selector.best_estimator_, selector.best_params_, selector.best_score_))
            else:
                if selector.best_score_ > self.best_score:
                    self.best_score = selector.best_score_
                    self.best_model = selector.best_estimator_
                    self.best_params = selector.best_params_

        return self
    
    def __fit_random_search(self, params, scoring, n_jobs, keep_all_models) -> Self:
        pass
        
    def fit(self, params, scoring: str = 'accuracy', n_jobs: int = -1, keep_all_models: bool=False) -> Self:
        self.params = params
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_score = 0
        if keep_all_models:
            self.all_models_ = []
        if self.selector == 'grid_search':
            return self.__fit_grid_search(params, scoring, n_jobs, keep_all_models)
        elif self.selector == 'random_search':
            return self.__fit_random_search(params, scoring, n_jobs, keep_all_models)    
    
    def build_best_model(self): 
        return type(self.best_model)(random_state=self.random_state, **self.best_params)
