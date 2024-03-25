from modules.training.base import MakeTraining
from dislib.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import clone
import pandas as pd
import pickle


class ModelSelection(MakeTraining):
    def __init__(self, model=None, scoring=None):
        if model is not None:
            if isinstance(model, list):
                self.model = model
            else:
                self.model = [model]
        else:
            self.model = []
        self.fitted = False
        self.results = None
        self.parameters = None
        self.scoring = scoring
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_model = None

    def fit(self, X, y):
        if (isinstance(self.model, list) and len(self.model)
                != len(self.parameters)):
            raise Warning("No fit done. There should be parameters "
                          "specified for all models.")
            return
        if not self.fitted:
            raise ValueError("To train teh best model, the gridsearch should"
                             "be previously fitted.")
        if isinstance(self.model, list):
            best_model = self.best_score_.index(max(self.best_score_))
            self.best_model = clone(self.best_estimator_[best_model]).set_params(**self.best_params_[best_model])
            self.best_model.fit(X.collect(), y.collect())
        else:
            self.best_model = clone(self.best_estimator_).set_params(**self.best_params_)
            self.best_model.fit(X, y)
        return self.best_model

    def execute_grid_search(self, X, y):
        if (isinstance(self.model, list) and len(self.model)
                != len(self.parameters)):
            raise Warning("No fit done. There should be parameters "
                          "specified for all models.")
            return
        if isinstance(self.model, list):
            self.best_estimator_ = []
            self.best_params_ = []
            self.best_score_ = []
            self.results = []
            grid_searchs = []
            for model, parameters in zip(self.model, self.parameters):
                grid_search = GridSearchCV(model, parameters, scoring=self.scoring)
                grid_search.train_candidates(X, y)
                grid_searchs.append(grid_search)
            for grid_search in grid_searchs:
                grid_search.score(X, y)
                self.best_estimator_.append(grid_search.best_estimator_)
                self.best_params_.append(grid_search.best_params_)
                self.best_score_.append(grid_search.best_score_)
                self.results.append(grid_search.cv_results_)
            self.fitted = True
        else:
            grid_search = GridSearchCV(self.model, self.parameters)
            grid_search.fit(X, y)
            self.best_estimator_ = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            self.best_score_ = grid_search.best_score_
            self.results = grid_search.cv_results_
            self.fitted = True

    def execute_random_search(self, X, y):
        if (isinstance(self.model, list) and len(self.model)
                != len(self.parameters)):
            raise Warning("No fit done. There should be parameters "
                          "specified for all models.")
            return
        if isinstance(self.model, list):
            self.best_estimator_ = []
            self.best_params_ = []
            self.best_score_ = []
            self.results = []
            grid_searchs = []
            for model, parameters in zip(self.model, self.parameters):
                grid_search = RandomizedSearchCV(model, parameters, scoring=self.scoring)
                grid_search.train_candidates(X, y)
                grid_searchs.append(grid_search)
            for grid_search in grid_searchs:
                grid_search.score(X, y)
                self.best_estimator_.append(grid_search.best_estimator_)
                self.best_params_.append(grid_search.best_params_)
                self.best_score_.append(grid_search.best_score_)
                self.results.append(grid_search.cv_results_)
            self.fitted = True
        else:
            grid_search = RandomizedSearchCV(self.model, self.parameters)
            grid_search.fit(X, y)
            self.best_estimator_ = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            self.best_score_ = grid_search.best_score_
            self.results = grid_search.cv_results_
            self.fitted = True

    def add_validation_metric(self, scoring):
        self.fitted = False
        self.scoring = scoring

    def set_paramaters_models(self, parameters):
        self.fitted = False
        if isinstance(self.parameters, list):
            self.parameters = parameters

    def set_models(self, model):
        if model is not None:
            self.fitted = False
            if isinstance(model, list):
                if not isinstance(self.model, list) and not isinstance(self.model, dict):
                    if self.model is not None:
                        self.model = [self.model]
                    else:
                        self.model = []
                elif isinstance(self.model, dict):
                    raise ValueError("If the models are specified as a "
                                     "dictionary no list should be added.")
                self.model.extend(model)
            else:
                self.model = [self.model, model]
        else:
            raise ValueError("This function should receive at"
                             " least one model as input.")

    def get_best_model(self, model=None):
        if model is not None:
            if isinstance(self.model, dict) and isinstance(model, str):
                return self.model[model]
            elif isinstance(self.model, list) and isinstance(model, int):
                return self.model[model]
        else:
            return self.model

    def visualize_results(self):
        if self.fitted:
            if isinstance(self.model, list):
                for model, results in zip (self.model, self.results):
                    print("Results with model type: " + str(type(model)))
                    print(pd.DataFrame(results).to_string())
            elif self.model is not None:
                print(pd.DataFrame(self.results))
        else:
            raise Warning("To visualize the data a fit of the models "
                          "should be done.")

    def get_info_best_model(self):
        if self.fitted:
            if not isinstance(self.best_params_, list):
                return self.best_estimator_, self.best_params_
            else:
                best_model = self.best_score_.index(min(self.best_score_))
                return self.best_estimator_[best_model], self.best_params_[best_model]
        else:
            raise ValueError("Before being able to recover the info of the "
                             "best model"
                             "the models should be trained.")

    def get_model(self):
        if self.fitted:
            return self.best_model
        else:
            raise ValueError("Before being able to recover the best model"
                             "the models should be trained.")

    def save_model(self, model=None, save_format="pickle"):
        if self.fitted:
            if save_format == "pickle":
                if model is not None:
                    with open(model, "wb") as f:
                        pickle.dump(self.best_model, f)
                else:
                    with open("./"+str(type(self.best_model).__name__), "wb") as f:
                        pickle.dump(self.best_model, f)
            else:
                pass
        else:
            raise ValueError("Before being able to recover the best model"
                             "the models should be trained.")

    def cross_validation(self):
        # TODO: Future work
        pass
