from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier



class StackingSK:
    def __init__(self, models=[
            RandomForestClassifier(bootstrap=False, criterion='gini', max_features=0.21, min_samples_leaf=9, min_samples_split=3, n_jobs=-1),
            CatBoostClassifier(verbose=0, thread_count=-1),
            XGBClassifier(n_estimators=489, max_depth=15, learning_rate=0.28, subsample=0.9, colsample_bytree=0.99, gamma=0.06, reg_alpha=6 * 1e-5, reg_lambda=0.95, n_jobs=-1)
        ], meta_model=LogisticRegression(n_jobs=-1), cv=4, n_jobs=-1):
        self.models = models
        self.meta_model = meta_model
        self.cv = cv
        self.n_jobs = n_jobs
        self.optimizations = []
        self.alg = StackingClassifier(estimators=[(type(model).__name__, model) for model in self.models], final_estimator=self.meta_model, cv=self.cv, n_jobs=self.n_jobs, verbose=1)
        
    def __str__(self):
        return 'StackingSK'

    def get_model_names(self):
        return {'algorithm': 'StackingSK', 'models': [type(model).__name__ for model in self.models], 'meta_model': type(self.meta_model).__name__}
    
    def get_hyperparams(self):
        return {'cv': self.cv, 'n_jobs': self.n_jobs}
    
    def save_model(self, filename):
        for model in self.models:
            pickle.dump(model, open(filename + '_' + type(model).__name__, 'wb'))
        pickle.dump(self.meta_model, open(filename + '_' + type(self.meta_model).__name__, 'wb'))
    
    def fit(self, X, y):
        self.alg.fit(X, y)

    def predict(self, X):
        return self.alg.predict(X)
    