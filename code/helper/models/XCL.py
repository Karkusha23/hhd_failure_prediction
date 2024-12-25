import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score
import optuna
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform
from sklearn.utils.class_weight import compute_class_weight


class WeightedXCL:
    def __init__(self, weights=None, thresh=0.5):
        # self.model_xgb = XGBClassifier(tree_method='gpu_hist')
        # self.model_cat = CatBoostClassifier(verbose=0, task_type='GPU')
        # self.model_lgbm = LGBMClassifier(verbose=0, device='gpu')
        self.model_xgb = XGBClassifier(n_jobs=-1)
        self.model_cat = CatBoostClassifier(verbose=0, thread_count=-1)
        self.model_lgbm = LGBMClassifier(verbose=-1, n_jobs=-1)
        self.preds_thresh = thresh
    
        if weights is None:
            self.weights = [1/3, 1/3, 1/3]
        else:
            assert len(weights) == 3, "Количество весов должно быть равно 3"
            self.weights = weights
        
        self.optimizations = []
            
    def fit(self, X_train, y_train):
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        print("Обучаем XGBoost...")
        self.model_xgb = XGBClassifier()
        self.model_xgb.fit(X_train, y_train, scale_pos_weight=class_weights[1] / class_weights[0], n_jobs=-1)

        print("Обучаем CatBoost...")
        self.model_cat = CatBoostClassifier(class_weights=class_weight_dict, thread_count=-1)
        self.model_cat.fit(X_train, y_train)

        print("Обучаем LightGBM...")
        self.model_lgbm = LGBMClassifier(class_weight=class_weight_dict, n_jobs=-1)
        self.model_lgbm.fit(X_train, y_train)
        
        print("Обучение завершено")

    def __str__(self):
        return 'WeightedXCL'
    
    def get_model_names(self):
        return {'main_model': 'WeightedXCL', 'submodels': ['XGBoost', 'CatBoost', 'LightGBM']}
    
    def get_hyperparams(self):
        xgb_params = self.model_xgb.get_params()
        xgb_params['weigth'] = self.weights[0]
        cat_params = self.model_cat.get_all_params()
        cat_params['weigth'] = self.weights[1]
        lgbm_params = self.model_lgbm.get_params()
        lgbm_params['weigth'] = self.weights[2]
        return {'xgboost': xgb_params, 'catboost': cat_params, 'lgbm': lgbm_params}
    
    def save_model(self, filename):
        self.model_xgb.save_model(filename + '_xgb')
        self.model_cat.save_model(filename + '_cat')
        self.model_lgbm.booster_.save_model(filename + '_lgbm')
    
    def predict_proba(self, X):
        proba_xgb = self.model_xgb.predict_proba(X)
        proba_cat = self.model_cat.predict_proba(X)
        proba_lgbm = self.model_lgbm.predict_proba(X)

        # Взвешенное суммирование предсказаний
        weighted_proba = self.weights[0] * proba_xgb + self.weights[1] * proba_cat + self.weights[2] * proba_lgbm
        
        return weighted_proba
    
    def predict(self, X):
        weighted_proba = self.predict_proba(X)
        return weighted_proba[:, 1] >= self.preds_thresh
    
    def set_weights(self, weights):
        assert len(weights) == 3, "Количество весов должно быть равно 3"
        self.weights = weights
    
    def optimize_weights(self, X_val, y_val):
        best_score = 0.0
        best_weights = []
        
        weights_grid = [
            [1/3, 1/3, 1/3],
            [1/2, 1/4, 1/4],
            [1/4, 1/2, 1/4],
            [1/4, 1/4, 1/2],
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4]
        ]
        
        for weights in weights_grid:
            self.set_weights(weights)
            preds = self.predict(X_val)
            score = f1_score(y_val, preds)  # Можно использовать другую метрику

            if score > best_score:
                best_score = score
                best_weights = weights
        
        self.optimizations.append({'optimization': 'optimize_weights', 'score': score, 'data': 'val'})

        return best_weights, best_score
    
    def optimize_models_hyperparameters(self, X_train, y_train, method='random', epochs=5, cv=4):
        assert method in ['random', 'grid'], "Метод оптимизации должен быть 'random' или 'grid'"
        
        if method == 'grid':
            # Grid оптимизация
            param_grid_xgb = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7, 10, 15]
            }
            
            param_grid_cat = {
                'iterations': [100, 200],
                'depth': [4, 6, 8, 10, 15],
                'learning_rate': [0.01, 0.1]
            }

            param_grid_lgbm = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [-1, 5, 10]
            }
            search_xgb = GridSearchCV(self.model_xgb, param_grid_xgb, cv=cv, scoring='f1')
            search_cat = GridSearchCV(self.model_cat, param_grid_cat, cv=cv, scoring='f1')
            search_lgbm = GridSearchCV(self.model_lgbm, param_grid_lgbm, cv=cv, scoring='f1')
        else:
            # Используем RandomizedSearchCV
            random_grid_xgb = {
                'n_estimators': randint(50, 500),  
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 15),  
                'subsample': uniform(0.5, 0.5), 
                'colsample_bytree': uniform(0.6, 0.4), 
                'gamma': uniform(0.0, 5.0),  
                'reg_alpha': uniform(0.5, 0.5),  
                'reg_lambda': uniform(0.5, 0.5),  
            }   
            
            random_grid_cat = {
                'n_estimators': randint(50, 500),  
                'learning_rate': uniform(0.01, 0.3),  
                'depth': randint(3, 15),  
                'l2_leaf_reg': uniform(1, 10),  
                'bagging_temperature': uniform(0.0, 2.0), 
                'border_count': randint(32, 255),  
                'random_strength': uniform(0.5, 2.0),
            }
            
            random_grid_lgbm = {
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
            search_xgb = RandomizedSearchCV(self.model_xgb, param_distributions=random_grid_xgb, n_iter=epochs, cv=cv, scoring='f1', random_state=42, n_jobs=-1)
            search_cat = RandomizedSearchCV(self.model_cat, param_distributions=random_grid_cat, n_iter=epochs, cv=cv, scoring='f1', random_state=42, n_jobs=-1)
            search_lgbm = RandomizedSearchCV(self.model_lgbm, param_distributions=random_grid_lgbm, n_iter=epochs, cv=cv, scoring='f1', random_state=42, n_jobs=-1)
        
        print("=== Оптимизация гиперпараметров для XGBoost... ===")
        search_xgb.fit(X_train, y_train)
        self.model_xgb = search_xgb.best_estimator_
        print("Лучшие параметры для XGBoost:", search_xgb.best_params_)

        # Оптимизация CatBoost
        print("=== Оптимизация гиперпараметров для CatBoost... ===")
        search_cat.fit(X_train, y_train)
        self.model_cat = search_cat.best_estimator_
        print("Лучшие параметры для CatBoost:", search_cat.best_params_)

        # Оптимизация LightGBM
        print("=== Оптимизация гиперпараметров для LightGBM... ===")
        search_lgbm.fit(X_train, y_train)
        self.model_lgbm = search_lgbm.best_estimator_
        print("Лучшие параметры для LightGBM:", search_lgbm.best_params_)

        self.optimizations.append({'optimization': 'simple', 'data': 'train', 'params': {'method': method, 'epochs': epochs, 'cv': cv}})
    
    def optimize_hyperparameters_tpe(self, X_train, y_train, n_trials=30):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        def objective(trial, X_train, X_val, y_train, y_val):
            param_xgb = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 500),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('xgb_gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 1.0, log=True)
            }

            param_cat = {
                'iterations': trial.suggest_int('cat_iterations', 100, 800),
                'depth': trial.suggest_int('cat_depth', 4, 15),
                'learning_rate': trial.suggest_float('cat_learning_rate', 1e-3, 0.3, log=True),
                'bagging_temperature': trial.suggest_float('cat_bagging_temperature', 0.0, 2.0),
                'border_count': trial.suggest_int('cat_border_count', 32, 255),
                'random_strength': trial.suggest_float('cat_random_strength', 0.5, 2.0),
                'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1e-8, 10.0, log=True)
            }

            param_lgbm = {
                'n_estimators': trial.suggest_int('lgbm_n_estimators', 50, 500),
                'max_depth': trial.suggest_int('lgbm_max_depth', 3, 15),
                'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('lgbm_num_leaves', 10, 150),
                'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 5, 100),
                'subsample': trial.suggest_float('lgbm_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('lgbm_reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('lgbm_reg_lambda', 1e-8, 1.0, log=True)
            }

            model_xgb = XGBClassifier(**param_xgb, n_jobs=-1)
            model_cat = CatBoostClassifier(**param_cat, verbose=0, thread_count=-1)
            model_lgbm = LGBMClassifier(**param_lgbm, verbose=-1, n_jobs=-1)
            
            # Обучаем модели
            model_xgb.fit(X_train, y_train)
            model_cat.fit(X_train, y_train)
            model_lgbm.fit(X_train, y_train)

            y_pred_xgb = model_xgb.predict(X_val)
            y_pred_cat = model_cat.predict(X_val)
            y_pred_lgbm = model_lgbm.predict(X_val)

            # Рассчитываем F1-score для каждой модели
            f1_xgb = f1_score(y_val, y_pred_xgb)
            f1_cat = f1_score(y_val, y_pred_cat)
            f1_lgbm = f1_score(y_val, y_pred_lgbm)
    
            return np.mean([f1_xgb, f1_cat, f1_lgbm])
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=n_trials)
        best_params = study.best_params
            
        self.model_xgb = XGBClassifier(
            n_estimators=best_params['xgb_n_estimators'],
            max_depth=best_params['xgb_max_depth'],
            learning_rate=best_params['xgb_learning_rate'],
            subsample=best_params['xgb_subsample'],
            colsample_bytree=best_params['xgb_colsample_bytree'],
            gamma=best_params['xgb_gamma'],
            reg_alpha=best_params['xgb_reg_alpha'],
            reg_lambda=best_params['xgb_reg_lambda'],
            n_jobs=-1
        )

        self.model_cat = CatBoostClassifier(
            iterations=best_params['cat_iterations'],
            depth=best_params['cat_depth'],
            learning_rate=best_params['cat_learning_rate'],
            bagging_temperature=best_params['cat_bagging_temperature'],
            border_count=best_params['cat_border_count'],
            random_strength=best_params['cat_random_strength'],
            l2_leaf_reg=best_params['cat_l2_leaf_reg'],
            verbose=0,
            thread_count=-1
        )

        self.model_lgbm = LGBMClassifier(
            n_estimators=best_params['lgbm_n_estimators'],
            max_depth=best_params['lgbm_max_depth'],
            learning_rate=best_params['lgbm_learning_rate'],
            num_leaves=best_params['lgbm_num_leaves'],
            min_child_samples=best_params['lgbm_min_child_samples'],
            subsample=best_params['lgbm_subsample'],
            colsample_bytree=best_params['lgbm_colsample_bytree'],
            reg_alpha=best_params['lgbm_reg_alpha'],
            reg_lambda=best_params['lgbm_reg_lambda'],
            n_jobs=-1
        )

        self.model_xgb.fit(X_train, y_train)
        self.model_cat.fit(X_train, y_train)
        self.model_lgbm.fit(X_train, y_train)
        
        print(f"Лучшие гиперпараметры: {study.best_params}")
        print(f"Лучшая оценка: {study.best_value}")

        self.optimizations.append({'optimization': 'tpe', 'data': 'train', 'params': {'n_trials': n_trials}})

        return study.best_params
    