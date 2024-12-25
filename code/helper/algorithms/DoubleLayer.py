import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from helper.preprocessing import *
from optuna.samplers import TPESampler
import optuna


class DoubleLayer:
    def __init__(self, n_splits=5, n_opts=10, imp_thresh=0.001, default=True):
        self.n_splits = n_splits
        self.n_opts = n_opts
        self.imp_thresh = imp_thresh
        self.layer1 = None
        self.layer2 = None
        self.important_features = None
        self.layer1_preds = None
        self.default = default  # Использование tuned параметров по умолчанию
        self.layer_params = {
            'n_estimators': 489, 
            'max_depth':15, 
            'learning_rate': 0.28, 
            'subsample': 0.9, 
            'colsample_bytree': 0.99, 
            'gamma': 0.06, 
            'reg_alpha': 6 * 1e-5,
            'reg_lambda':0.95
        }
        
    def __str__(self):
        return 'DoubleLayer'
    
    def get_model_names(self):
        return {'main_model': 'DoubleLayer', 'submodels': [{'layer1': 'XGBoost'}, {'layer2': 'XGBoost'}]}
    
    def get_hyperparams(self):
        return {'n_splits': self.n_splits, 'n_opts': self.n_opts, 'threshold': self.imp_thresh, 'layer1': self.layer1.get_params(), 'layer2': self.layer2.get_params()}
    
    def save_model(self, filename):
        self.layer1.save_model(filename + '_layer1')
        self.layer2.save_model(filename + '_layer2')
    
    def compute_scale_pos_weight(self, y_train):
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = neg_count / pos_count
        return scale_pos_weight

    def optimize_hyperparams_tpe(self, X_train, y_train):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    
        def objective(trial):
            params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
                }
            
            model = XGBClassifier(**params, n_jobs=-1, scale_pos_weight=self.compute_scale_pos_weight(y_train))
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            f_score = f1_score(preds, y_val)
            
            return f_score
            
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_opts)
        
        return study.best_params, study.best_value 
    
    def set_model_1(self, X, y):
        if not self.default:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
            best_params, best_value = self.optimize_hyperparams_tpe(X_train, y_train, X_val, y_val)
        
            print(f"Лучшие гиперпараметры: {best_params}")
            print(f"Лучший результат валидации: {best_value}")
            self.layer_params = best_params
        
        # иначе self.layer_params с базовыми настройками
        self.layer1 = XGBClassifier(**self.layer_params, n_jobs=-1, scale_pos_weight=self.compute_scale_pos_weight(y))
        print("=== Обучаем модель первого слоя ===")
        self.layer1.fit(X, y)
        return self.layer1
    
    def get_important_features(self, model, X):
        feature_importance = model.feature_importances_
        feature_names = X.columns
        feature_importance_normalized = feature_importance / feature_importance.sum()

        # Сохранение признаков по важности
        self.feature2importance = sorted(list(zip(feature_names, feature_importance_normalized)), key=lambda x: x[1], reverse=True)

        # Отбор важных признаков по порогу
        important_features_mask = feature_importance_normalized >= self.imp_thresh
        selected_feature_names = [name for name, is_important in zip(feature_names, important_features_mask) if is_important]
        self.important_features = selected_feature_names
        print(f"=== Выбранные признаки: {selected_feature_names} ===")

        # Возвращаем выбранные важные признаки
        X_selected = X[selected_feature_names]
        X_selected.reset_index(drop=True, inplace=True)
        return X_selected, selected_feature_names
    
    def fit(self, X, y):
        self.set_model_1(X, y)
        X_selected, self.important_features = self.get_important_features(self.layer1, X)
        print("=== Получаем предсказания первой модели ===")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        layer1_preds = np.zeros(len(X))
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            self.layer1.fit(X_train[self.important_features], y_train)
            val_preds = self.layer1.predict(X_val[self.important_features])
            layer1_preds[val_idx] = val_preds
            
        X_selected['layer1_preds'] = layer1_preds

        print("=== Обучаем вторую модель ===")
        self.layer2 = XGBClassifier(**self.layer_params, n_jobs=-1, scale_pos_weight=self.compute_scale_pos_weight(y))
        self.layer2.fit(X_selected, y)
        
    def predict(self, X):
        X_selected = X[self.important_features]
        layer1_preds = self.layer1.predict(X_selected)
        X_selected['layer1_preds'] = layer1_preds
        predictions = self.layer2.predict(X_selected)
        
        return predictions   
    