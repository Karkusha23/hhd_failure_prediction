from dataclass import HDDDataset
import numpy as np
from scipy.special import boxcox1p, inv_boxcox1p
from scipy.stats import boxcox
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import TomekLinks, NearMiss, RandomUnderSampler


class PreprocUtils:
    @classmethod
    def add_time_features(cls, df):
        """
        Добавляет дополнительные признаки: сдвиги и разности, только для raw данных
        """
        shift_periods = 1
        # Создаем 
        shift_features = pd.DataFrame(index=df.index)
        diff_features = pd.DataFrame(index=df.index)
        SMART_FEATURES = ['smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw', 'smart_192_raw', 'smart_197_raw', 'smart_199_raw', 'smart_240_raw', 'smart_241_raw', 'smart_242_raw']
        
        for col in SMART_FEATURES:
            shifted_col = f"shift_{col}_{shift_periods}"
            shift_features[shifted_col] = df[col].shift(shift_periods)
            diff_col = f"diff_{col}_{shift_periods}"
            diff_features[diff_col] = df[col] - df[col].shift(shift_periods)
        
        return pd.concat([df, shift_features, diff_features], axis=1).bfill()
    
    @classmethod
    def drop_unimportant_features(cls, df, columns, drop, cout=False):
        """
        Требует столбцы в упорядоченном по возрастанию важности списке
        """
        counter = 0
        if cout:
            print('    Удаляем не нужные признаки...     ')
        for i, col in enumerate(columns):
            df.drop(columns=[col], inplace=True, errors='ignore')
            if cout:
                print(f'{col} ')
            if i + 1 == drop:
                break
            
        return df
    
    @classmethod
    def normalize_data(cls, df, columns=None, auto=True, method='YJ', inplace=False, lmbd=0.35):
        """
        Выполняет преобразования нормализации
        """

        shift_periods = 1

        METHODS = ['log1p', 'boxcoxlog', 'minmax', 'zscore', 'YJ', 'BC', 'formulae']
        
        if auto:
            assert method in METHODS, f"Выбрано неверное название метода преобразования, доступны {METHODS}"
            SMART_LIST_RAW = [i for i in df.columns if (i.startswith('smart_') and i.endswith('_raw'))]            
            SMART_LIST_NORMALIZED = [i for i in df.columns if (i.startswith('smart_') and i.endswith('_normalized'))]
            if method == 'log1p':
                if inplace:
                    df[SMART_LIST_RAW] = np.log1p(df[SMART_LIST_RAW])
                else:
                    for col in SMART_LIST_RAW:
                        df[[col + '_normalized']] = np.log1p(df[[col]])
            
            elif method == 'boxcoxlog':
                BOXCOX_35_LIST = ['smart_9_raw', 'smart_240_raw', 'smart_241_raw', 'smart_242_raw']
                LOG_LIST = ['smart_5_raw', 'smart_187_raw', 'smart_188_raw', 'smart_192_raw', 'smart_197_raw', 'smart_199_raw']
                LOGLIST = [i for i in df.columns if i in LOG_LIST]
                if inplace:
                    for col in BOXCOX_35_LIST:
                        df[col] = boxcox1p(df[col], 0.35)
                        if f"shift_col_{shift_periods}" in df.columns:
                            df[f"shift_col_{shift_periods}"] = boxcox1p(df[f"shift_col_{shift_periods}"], 0.35)
                    for col in LOG_LIST:
                        df[col] = np.log1p(df[col])
                        if f"shift_col_{shift_periods}" in df.columns:
                            df[f"shift_col_{shift_periods}"] = np.log1p(df[f"shift_col_{shift_periods}"])
                else:
                    for col in BOXCOX_35_LIST:
                        df[col + '_normalized'] = boxcox1p(df[col], 0.35)
                    for col in LOG_LIST:
                        df[col + '_normalized'] = np.log1p(df[col])

            elif method == 'formulae':
                if not inplace:
                    # ! probably deprecated
                    df['smart_5_raw_normalized'] = np.maximum(100 - df['smart_5_raw'], 0)
                    df['smart_9_raw_normalized'] = 100 - np.log1p(df['smart_9_raw'])
                    df['smart_187_raw_normalized'] = 100 - np.log1p(df['smart_187_raw'])
                    df['smart_188_raw_normalized'] = 100 - np.log1p(df['smart_188_raw'])
                    df['smart_192_raw_normalized'] = np.maximum(100 - df['smart_192_raw'], 0)
                    df['smart_197_raw_normalized'] = np.maximum(100 - df['smart_197_raw'], 0)
                    # ! df['smart_198_raw_normalized'] = np.maximum(100 - df['smart_198_raw'], 0)
                    df['smart_199_raw_normalized'] = np.maximum(200 - df['smart_199_raw'], 0)
                    df['smart_240_raw_normalized'] = 100 - np.log1p(df['smart_240_raw'])
                    df['smart_241_raw_normalized'] = 100 - np.log1p(df['smart_241_raw'])
                    df['smart_242_raw_normalized'] = 100 - np.log1p(df['smart_242_raw'])
                
                else:
                    df['smart_5_raw'] = np.maximum(100 - df['smart_5_raw'], 0)
                    df['smart_9_raw'] = 100 - np.log1p(df['smart_9_raw'])
                    df['smart_187_raw'] = 100 - np.log1p(df['smart_187_raw'])
                    df['smart_188_raw'] = 100 - np.log1p(df['smart_188_raw'])
                    df['smart_192_raw'] = np.maximum(100 - df['smart_192_raw'], 0)
                    df['smart_197_raw'] = np.maximum(100 - df['smart_197_raw'], 0)
                    # ! df['smart_198_raw'] = np.maximum(100 - df['smart_198_raw'], 0)
                    df['smart_199_raw'] = np.maximum(200 - df['smart_199_raw'], 0)
                    df['smart_240_raw'] = 100 - np.log1p(df['smart_240_raw'])
                    df['smart_241_raw'] = 100 - np.log1p(df['smart_241_raw'])
                    df['smart_242_raw'] = 100 - np.log1p(df['smart_242_raw'])
            
            else:
                if method == 'minmax':
                    scaler = MinMaxScaler()
                elif method == 'zscore':
                    scaler = StandardScaler()
                elif method == 'YJ':
                    scaler = PowerTransformer(method='yeo-johnson')
                elif method == 'BC':
                    scaler = PowerTransformer(method='box-cox')

                if inplace:
                    df[SMART_LIST_RAW] = scaler.fit_transform(df[SMART_LIST_RAW])
                else:
                    for col in SMART_LIST_RAW:
                        df[col + '_normalized'] = scaler.fit_transform(df[[col]])
                        
        else:
            assert columns is not None
            assert set(columns) <= set(df.columns)  # Проверяем, что выбранные столбцы есть в датасете    
            assert method in ['log', 'log1p', 'boxcox']
            
            if method == 'log':
                if inplace:
                    df[columns] = df[columns].apply(lambda x: np.log(x))
                else:
                    for col in columns:
                        df[col + '_normalized'] = df[col].apply(lambda x: np.log(x))
            elif method == 'log1p':
                if inplace:
                    df[columns] = df[columns].apply(lambda x: np.log1p(x))
                else:
                    for col in columns:
                        df[col + '_normalized'] = df[col].apply(lambda x: np.log1p(x))
            elif method == 'boxcox':
                if inplace:
                    df[columns] = boxcox1p(df[columns], lmbd)     
                else:
                    for col in columns:
                        df[col + '_normalized'] = boxcox1p(df[col], lmbd)

        return df


class Preprocessing():
    def __init__(self, data: HDDDataset):
        self.data = data
        self.df = data.df
        self.lmb = None
        self.train_df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.shift_periods = 1
        self.operations = []
        
    def clear_unused_data(self):
        """
        Удаляет лишние признаки
        """
        print('Clearing unused columns...')
        self.df = self.df.drop(columns=['model', 'capacity_bytes', 'smart_198_raw'], errors='ignore')
        self.operations.append({'operation': 'clear_unused_data'})
    
    def add_target_column(self, target_column='failure30'):
        print('Adding target column...')
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(by=['serial_number', 'date'])
        self.df[target_column] = 0
        # Группировка по серийным номерам
        self.df[target_column] = (
            self.df.groupby('serial_number')['failure']
            .transform(lambda x: x[::-1].rolling(window=30, min_periods=1).max()[::-1])
        )
        self.df[target_column] = self.df[target_column].astype('int8')
        self.operations.append({'operation': 'add_taget_column'})
    
    def add_time_features(self):
        if self.X_train is not None:
            print('Adding time features to splitted dataset...')
            self.X_train = PreprocUtils.add_time_features(self.X_train)
            if self.X_val is not None:
                self.X_val = PreprocUtils.add_time_features(self.X_val)
            self.X_test = PreprocUtils.add_time_features(self.X_test)
        elif self.train_df is not None:
            print('Adding time features to unsplitted dataset...')
            self.train_df = PreprocUtils.add_time_features(self.train_df)
        else:
            raise RuntimeError('Need to prepare data for this operation!')
        self.operations.append({'operation': 'add_time_features', 'params': {'splitted': (self.X_train is not None)}})

    def normalize_data(self, columns=None, auto=True, method='YJ', inplace=False, lmbd=0.35):
        if self.X_train is not None:
            print(f'Normalizing splitted dataset with {method}...')
            self.X_train = PreprocUtils.normalize_data(self.X_train, columns, auto, method, inplace, lmbd)
            if self.X_val is not None:
                self.X_val = PreprocUtils.normalize_data(self.X_val, columns, auto, method, inplace, lmbd)
            self.X_test = PreprocUtils.normalize_data(self.X_test, columns, auto, method, inplace, lmbd)
        elif self.train_df is not None:
            print(f'Normalizing unsplitted dataset with {method}...')
            self.train_df = PreprocUtils.normalize_data(self.train_df, columns, auto, method, inplace, lmbd)
        else:
            raise RuntimeError('Need to prepare data for this operation!')
        self.operations.append({'operation': 'normalize_data', 'params': {'method': method, 'auto': auto, 'inplace': inplace, 'splitted': (self.X_train is not None)}})
                        
    def get_train_df(self, pair=False):
        if not pair:
            return self.train_df
        return self.train_df.drop(columns=['failure30'], errors='ignore'), self.train_df['failure30']
    
    def prepare_train_df(self):
        assert self.operations[0]['operation'] == 'clear_unused_data' and self.operations[1]['operation'] == 'add_taget_column'
        print('Prepairing train dataset...')
        self.train_df = self.df.drop(columns=['model', 'date', 'serial_number', 'capacity_bytes', 'smart_198_raw'], errors='ignore')
        self.operations.append({'operation': 'prepare_train_df'})
        return self.train_df
    
    def train_test_val_split(self, train_size=0.8, val_size=0.1, test_size=0.1, sampling_strat=-1, oversampling=None, undersampling=None):
        assert self.train_df is not None
        assert train_size + val_size + test_size == 1
        assert sampling_strat == -1 or (sampling_strat > 0 and sampling_strat < 1), 'Некорректное значение sampling_strat'
        assert oversampling in ['Default', 'Enn', 'Borderline', None, 'Tomek', 'Adasyn'], 'Некорректное значение oversampling'
        assert undersampling in ['Tomek', 'NearMiss', None, 'Random'], 'Некорректное значение undersampling' 
        
        print('Splitting train dataset...')
        X, y_true = self.get_train_df(True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_true, test_size=test_size, random_state=42, stratify=y_true)
        
        if sampling_strat > 0:
            if oversampling == None or oversampling == 'Default':
                print('Overampling with SMOTE')
                self.X_train, self.y_train = SMOTE(sampling_strategy=sampling_strat, n_jobs=-1).fit_resample(self.X_train, self.y_train)
            elif oversampling == 'enn':
                print('Overampling with ENN...')
                self.X_train, self.y_train = SMOTEENN(sampling_strategy=sampling_strat, n_jobs=-1).fit_resample(self.X_train, self.y_train)
            elif oversampling == 'Borderline':
                print('Overampling with BorderlineSMOTE...')
                self.X_train, self.y_train = BorderlineSMOTE(sampling_strategy=sampling_strat, n_jobs=-1).fit_resample(self.X_train, self.y_train)
            elif oversampling == 'Adasyn':
                print('Overampling with ADASYN...')
                self.X_train, self.y_train = ADASYN(sampling_strategy=sampling_strat, n_jobs=-1).fit_resample(self.X_train, self.y_train)
            else:
                print('Overampling with TomekLinks...')
                self.X_train, self.y_train = SMOTETomek(sampling_strategy=sampling_strat, n_jobs=-1).fit_resample(self.X_train, self.y_train)
        if undersampling is not None:
            if undersampling == 'Tomek':
                print('Undersampling with TomekLinks...')
                self.X_train, self.y_train = TomekLinks(n_jobs=-1).fit_resample(self.X_train, self.y_train)
            elif undersampling == 'Random':
                print('Undersampling with Random...')
                self.X_train, self.y_train = RandomUnderSampler(random_state=42, replacement=True).fit_resample(self.X_train, self.y_train)
            else:
                print('Undersampling with NearMiss...')
                self.X_train, self.y_train = NearMiss(version=2, n_jobs=-1).fit_resample(self.X_train, self.y_train)
            
        if val_size > 0:
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=test_size / (test_size + val_size), random_state=42, stratify=self.y_test)
        self.operations.append({'operation': 'train_test_val_split', 'params': {'train_size': train_size, 'val_size': val_size, 'test_size': test_size, 'sampling_strat': sampling_strat, 'oversampling': oversampling, 'undersampling': undersampling}})
    
    def rescale_types(self):
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.y_train = self.y_train.astype('int8')
        self.y_test = self.y_test.astype('int8')
        if self.X_val is not None:
            self.X_val = self.X_val.astype('float32')
            self.y_val = self.y_val.astype('int8')
        self.operations.append({'operation': 'rescale_types'})
        
    def drop_unimportant_features(self, columns=None, drop=None):
        formulae_normalized = False
        
        for op in self.operations:
            if op['operation'] == 'normalize_data' and op['params']['method'] == 'formulae':
                formulae_normalized = True
                break
                
        if columns is None:
            if formulae_normalized:
                columns = [
                    'failure',
                    'diff_smart_5_raw_1',
                    'diff_smart_242_raw_1',
                    'diff_smart_240_raw_1',
                    'diff_smart_199_raw_1',
                    'diff_smart_9_raw_1',
                    'diff_smart_241_raw_1',
                    # --- 
                    'shift_smart_9_raw_1',
                    'shift_smart_242_raw_1',
                    'shift_smart_241_raw_1',
                    'shift_smart_240_raw_1',
                    'smart_241_raw',
                    'smart_9_raw',
                    'diff_smart_192_raw_1',
                    'smart_240_raw',
                    # --- 
                    'shift_smart_199_raw_1',
                    'shift_smart_192_raw_1',
                    'smart_242_raw'
                    'shift_smart_187_raw_1',
                    'diff_smart_188_raw_1',
                    # ---
                    'shift_smart_5_raw_1',
                    'smart_188_raw',
                    'smart_192_raw',
                    'shift_smart_197_raw_1',
                    'shift_smart_188_raw_1',
                    'smart_199_raw',
                    # ---
                    'diff_smart_197_raw_1',
                    'smart_5_raw',
                    'smart_187_raw',
                    'smart_197_raw',                    
                    'diff_smart_187_raw_1',
                ]
                drop = 13 if drop is None else drop
            else:
                columns = [
                    'failure',
                    'diff_smart_5_raw_1',
                    'diff_smart_199_raw_1',
                    'smart_197_raw_normalized',
                    'diff_smart_9_raw_1',
                    'diff_smart_240_raw_1',
                    'diff_smart_188_raw_1',
                    'smart_199_raw_normalized',
                    'diff_smart_242_raw_1',
                    'diff_smart_241_raw_1',
                    'smart_9_raw_normalized',
                    'diff_smart_192_raw_1',
                    # --- 
                    'smart_242_raw_normalized',
                    'smart_188_raw_normalized',
                    'smart_241_raw_normalized',
                    'shift_smart_241_raw_1',
                    'shift_smart_242_raw_1',
                    'shift_smart_5_raw_1',                    
                    'shift_smart_240_raw_1',
                    'shift_smart_199_raw_1',
                    # --- 
                    'shift_smart_9_raw_1',
                    'shift_smart_192_raw_1',
                    'smart_240_raw_normalized',
                    'smart_188_raw',
                    'smart_192_raw_normalized',
                    'shift_smart_187_raw_1',
                    'smart_241_raw'
                    'shift_smart_197_raw_1',
                    'smart_240_raw',
                    'smart_9_raw',
                    'smart_242_raw',
                    'smart_199_raw',
                    'smart_192_raw',
                    'diff_smart_197_raw_1',
                    # --- 
                    'smart_5_raw',
                    'smart_5_raw_normalized',
                    'smart_187_raw',
                    'shift_smart_188_raw_1',                    
                    'diff_smart_187_raw_1',                  
                    'smart_187_raw_normalized',
                    'smart_197_raw',
                ] 
                drop = 10 if drop is None else drop
                
        if self.X_train is not None:
            self.X_train = PreprocUtils.drop_unimportant_features(self.X_train, columns, drop, cout=True)
            if self.X_val is not None:
                self.X_val = PreprocUtils.drop_unimportant_features(self.X_val, columns, drop)
            self.X_test = PreprocUtils.drop_unimportant_features(self.X_test, columns, drop)
            
        elif self.train_df is not None:
            self.train_df = PreprocUtils.drop_unimportant_features(self.train_df, columns, drop, cout=True)
        else:
            raise RuntimeError('Need to prepare data for this operation!')
        
        self.operations.append({'operation': 'drop_unimportant_features', 'columns': columns, 'drop': drop})
        
    def denormalize_data(self, columns, method: str, lmb=None):
        assert set(columns) <= set(self.df.columns)  # Проверяем, что выбранные столбцы есть в датасете
        assert method in ['log', 'log1p', 'boxcox']
        assert lmb is None or method == 'boxcox'
        
        if method == 'log':
            self.df[columns] = self.df[columns].apply(lambda x: np.exp(x))
        elif method == 'log1p':
            self.df[columns] = self.df[columns].apply(lambda x: np.expm1(x))
        elif method == 'boxcox':
            self.df[columns] = self.df[columns].apply(lambda x: inv_boxcox1p(x, lmb))
        self.operations.append({'operation': 'denormalize_data', 'params': {'method': method}})
    
    @staticmethod
    def split_data_ln(X):
        X_normalized = pd.DataFrame()

        X_normalized['smart_5_normalized'] = np.maximum(100 - X['smart_5_raw'], 0)
        X_normalized['smart_9_normalized'] = 100 - np.log1p(X['smart_9_raw'])
        X_normalized['smart_187_normalized'] = 100 - np.log1p(X['smart_187_raw'])
        X_normalized['smart_188_normalized'] = 100 - np.log1p(X['smart_188_raw'])
        X_normalized['smart_192_normalized'] = np.maximum(100 - X['smart_192_raw'], 0)
        X_normalized['smart_197_normalized'] = np.maximum(100 - X['smart_197_raw'], 0)
        # ! X_normalized['smart_198_normalized'] = np.maximum(100 - X['smart_198_raw'], 0)
        X_normalized['smart_199_normalized'] = np.maximum(200 - X['smart_199_raw'], 0)
        X_normalized['smart_240_normalized'] = 100 - np.log1p(X['smart_240_raw'])
        X_normalized['smart_241_normalized'] = 100 - np.log1p(X['smart_241_raw'])
        X_normalized['smart_242_normalized'] = 100 - np.log1p(X['smart_242_raw'])
        
        scaler = PowerTransformer(method='yeo-johnson')
        X_log = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        return X_normalized, X_log   
        