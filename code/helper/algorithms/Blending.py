from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


class Blending():
    def __init__(self, models=[
            RandomForestClassifier(bootstrap=False, criterion='gini', max_features=0.21, min_samples_leaf=9, min_samples_split=3, n_jobs=-1),
            # MLPClassifier(activation='relu', alpha=0.45, batch_size=350, beta_1=0.85, beta_2=0.93, epsilon=4*1e-7, hidden_layer_sizes=(128, 64), learning_rate='adaptive', learning_rate_init=0.1, solver='adam'), 
            CatBoostClassifier(verbose=0, thread_count=-1),
            XGBClassifier(n_estimators=489, max_depth=15, learning_rate=0.28, subsample=0.9, colsample_bytree=0.99, gamma=0.06, reg_alpha=6 * 1e-5, reg_lambda=0.95, n_jobs=-1)
        ], meta_model=CatBoostClassifier(verbose=0, thread_count=-1), cv=4, n_jobs=-1):
        
        self.models = models
        self.meta_model = meta_model
        self.cv = cv
        self.n_jobs = n_jobs
        self.optimizations = []

    def __str__(self):
        return 'Blending'

    def get_model_names(self):
        return {'algorithm': 'Blending', 'models': [type(model).__name__ for model in self.models], 'meta_model': type(self.meta_model).__name__}
    
    def get_hyperparams(self):
        return {'cv': self.cv, 'n_jobs': self.n_jobs}
    
    def save_model(self, filename):
        for model in self.models:
            pickle.dump(model, open(filename + '_' + type(model).__name__, 'wb'))
        pickle.dump(self.meta_model, open(filename + '_' + type(self.meta_model).__name__, 'wb'))
    
    def data_val_pred(self, estimator, X_train, y_train, X_val):
        estimator.fit(X_train, y_train)
        return estimator.predict(X_val)
    
    def data_test_pred(self, estimator, X_train, y_train, X_test):
        estimator.fit(X_train, y_train)
        return estimator.predict(X_test)
    
    def meta_data(self, X_train, y_train, X_val, y_val, X_test):
        meta_y_train = y_val
        
        trained_models = []
        for model in self.models:
            print(f"=== Обучаем {type(model).__name__}...  ===")
            model.fit(X_train, y_train)
            trained_models.append(model)
            
        data_train_pred = [model.predict(X_val) for model in trained_models]
        data_test_pred = [model.predict(X_test) for model in trained_models]

        meta_X_train = pd.DataFrame(data_train_pred).T
        meta_X_test = pd.DataFrame(data_test_pred).T
        
        return meta_X_train, meta_y_train, meta_X_test

    def fit_predict(self, X_train, y_train, X_val, y_val, X_test):
        meta_X_train, meta_y_train, meta_X_test = self.meta_data(X_train, y_train, X_val, y_val, X_test)
        print("=== Обучаем мета модель... ===")
        self.meta_model.fit(meta_X_train, meta_y_train)
        return self.meta_model.predict(meta_X_test)
    