import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score
from scipy.stats import randint, uniform
from sklearn.utils.class_weight import compute_class_weight
import optuna


class MyModel:
    def __init__(self, model_name):
        assert model_name in ['XGBoost', 'CatBoost', 'LightGBM']
        self.model_name = model_name
        self.optimizations = []
        if model_name == 'XGBoost':
            self.model = XGBClassifier(n_jobs=-1)
            self.param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7, 10, 15]
            }
            
            self.random_param_grid = {
                'n_estimators': randint(50, 500),  
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 15),  
                'subsample': uniform(0.5, 0.5), 
                'colsample_bytree': uniform(0.6, 0.4), 
                'gamma': uniform(0.0, 5.0),  
                'reg_alpha': uniform(0.5, 0.5),  
                'reg_lambda': uniform(0.5, 0.5),  
            }   
            
        elif model_name == 'CatBoost':
            self.model = CatBoostClassifier(verbose=0, thread_count=-1)
            self.param_grid = {
                'iterations': [100, 200],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1]
            }
            self.random_param_grid = {
                'n_estimators': randint(50, 500),  
                'learning_rate': uniform(0.01, 0.3),  
                'depth': randint(3, 15),  
                'l2_leaf_reg': uniform(1, 10),  
                'bagging_temperature': uniform(0.0, 2.0), 
                'border_count': randint(32, 255),  
                'random_strength': uniform(0.5, 2.0),
            }
        else:
            self.model = LGBMClassifier(verbose=-1, n_jobs=-1)
            self.param_grid =  {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [-1, 5, 10]
            }
            self.random_param_grid = {
                'num_leaves': randint(20, 150),  
                'learning_rate': uniform(0.01, 0.2),  
                'n_estimators': randint(50, 500),  
                'max_depth': randint(3, 15),  
                'min_child_samples': randint(10, 100),  
                'subsample': uniform(0.5, 0.5),  
                'colsample_bytree': uniform(0.6, 0.4),  
                'reg_alpha': uniform(0.5, 0.5),  
                'reg_lambda': uniform(0.5, 0.5), 
            }
            

    def __str__(self):
        return self.model_name
    
    def get_hyperparams(self):
        if self.model_name == 'CatBoost':
            return self.model.get_all_params()
        return self.model.get_params()
    
    def save_model(self, filename):
        if self.model_name == 'LightGBM':
            self.model.booster_.save_model(filename)
        else:
            self.model.save_model(filename)
            
    def get_model_names(self):
        return self.model_name
    
    def fit_no_opt(self, X_train, y_train):
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        if self.model_name == 'XGBoost':
            print(f'Обучаем {self.model_name}...')
            self.model.fit(X_train, y_train, sample_weight=class_weights)
            print('Обучение завершено')
            return
        
        if self.model_name == 'CatBoost':
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
            self.model = CatBoostClassifier(class_weights=class_weight_dict, thread_count=-1)
        else: # LightGBM
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
            self.model = LGBMClassifier(class_weight=class_weight_dict, n_jobs=-1)
        
        print(f'Обучаем {self.model_name}...')
        self.model.fit(X_train, y_train)
        print('Обучение завершено')

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X_train, y_train, method='random', epochs=50, cv=4):
        assert method in ['random', 'grid'], "Метод оптимизации должен быть 'random' или 'grid'"
        self.optimizations.append(method)
        
        if method == 'grid':
            search = GridSearchCV(self.model, self.param_grid, cv=cv, scoring='f1')
        else:
            search = RandomizedSearchCV(self.model, param_distributions=self.param_grid, n_iter=epochs, cv=cv,
                                            scoring='f1', random_state=42, n_jobs=-1)

        print(f'=== Оптимизация гиперпараметров для {self.model_name}... ===')
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        print(f'Лучшие параметры для {self.model_name}:\n{search.best_params_}')

    def fit_tpe(self, X_train, y_train, X_val, y_val, n_trials=30):
        self.optimizations.append('TPE')
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        
        def objective(model_name, trial, X_train, X_val, y_train, y_val):
            assert model_name in ['XGBoost', 'CatBoost', 'LightGBM']
            if model_name == 'XGBoost':
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
                }
                model = XGBClassifier(**param, scale_pos_weight=class_weights[1] / class_weights[0], n_jobs=-1)
            elif model_name == 'CatBoost':
                param = {
                    'iterations': trial.suggest_int('cat_iterations', 100, 800),
                    'depth': trial.suggest_int('depth', 4, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True)
                }
                model = CatBoostClassifier(**param, verbose=0, class_weights=class_weight_dict, thread_count=-1)
            else:
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 150),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
                }
                model = LGBMClassifier(**param, verbose=-1, class_weight=class_weight_dict, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            f_score = f1_score(preds, y_val)

            return f_score

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(self.model_name, trial, X_train, X_val, y_train, y_val), n_trials=n_trials)

        print(f'Лучшие гиперпараметры: {study.best_params}')
        print(f'Лучшая оценка: {study.best_value}')
        self.model = self.model.set_params(**study.best_params)
        self.model.fit(X_train, y_train)
        return study.best_params
