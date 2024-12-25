import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

datasets_dir = os.path.dirname(os.path.realpath(__file__)) + '\\datasets'

# Dataset class

class HDDDataset:
    def __init__(self, dataframe, to_copy=True, name='UnnamedDataset'):
        assert isinstance(dataframe, pd.DataFrame) and isinstance(name, str)
        self.df = dataframe.copy() if to_copy else dataframe
        self.name = name

    @classmethod
    def read_csv(cls, filename, name=None):
        return cls(pd.read_csv(datasets_dir + '\\' + filename), False, name if name else filename[:-4])

    def to_csv(self, to_overwrite=False, filename=None):
        path = datasets_dir + '\\' + (filename if filename else self.name + '.csv')
        if os.path.isfile(path) and not to_overwrite:
            raise RuntimeError(f'to_csv error - {filename if filename else self.name + ".csv"} already exists')
        self.df.to_csv(path, index=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.df.iloc[item]
        return HDDDataset(self.df.iloc[item])

    def __str__(self):
        return self.name + ':\n' + str(self.df)

    def __copy__(self):
        return HDDDataset(self.df, True, self.name)

    def copy(self):
        return self.__copy__()

    # Список столбцов
    def column_list(self):
        return self.df.columns.values.tolist()

    # Возвращает датасет с переназначенными индексами. Новые индексы: [start_index, start_index + 1, ...]
    def reindex(self, start_index=0):
        assert isinstance(start_index, int)
        return HDDDataset(self.df.set_index(pd.Series(range(start_index, start_index + len(self.df)))), False)

    # Реляционная операция проекции. Возвращает датасет со столбцами, переданными в аргумент
    def projection(self, columns):
        if isinstance(columns, str):
            return HDDDataset(pd.DataFrame({columns: self.df[columns]}), False)
        if hasattr(columns, '__len__'):
            if len(columns) == 1:
                return HDDDataset(pd.DataFrame({columns[0]: self.df[columns[0]]}), False)
            return HDDDataset(self.df[columns], False)
        raise RuntimeError('Projection argument type error')

    # Отраженная операция проекции. Возвращает новый датасет, в котором удалены столбцы, переданные в аргумент
    def exclude_projection(self, exclude_columns):
        columns = self.column_list()
        if isinstance(exclude_columns, str):
            columns.remove(exclude_columns)
        elif hasattr(exclude_columns, '__len__'):
            for column in exclude_columns:
                columns.remove(column) if column else None
        else:
            raise RuntimeError('Exclude projection argument type error')
        if len(columns) == 1:
            return HDDDataset(pd.dataFrame({columns[0]: self.df[columns[0]]}), False)
        return HDDDataset(self.df[columns], False)

    # Операция слияния датасетов (вертикальное, то есть к строчкам первого добавляем строки второго). Возвращает датасет
    # left и right работают аналогично соединению в РА
    # Если left=True, right=True, то итоговый датасет будет содержать только общие столбцы исходных датасетов
    # Если left=True, right=False, то итоговый датасет будет содержать общие столбцы исходных датасетов и оставшиеся столбцы первого датасета
    # Если left=False, right=True, то аналогично предыдущему случаю
    # Если left=False, right=False, то итоговый датачет будет содержать объединение столбцов исходных датасетов
    def merge(self, other, left=False, right=False):
        assert isinstance(other, HDDDataset) or isinstance(other, pd.DataFrame)
        if isinstance(other, pd.DataFrame):
            other = HDDDataset(other, False)
        first = self.reindex(0)
        second = other.reindex(len(self))
        if left:
            second = second.exclude_projection([column if column not in first.column_list() else None for column in second.column_list()])
        if right:
            first = first.exclude_projection([column if column not in second.column_list() else None for column in first.column_list()])
        return HDDDataset(pd.concat([first.df, second.df]), False)

    # Возвращает поддатасет для определенного серийного номера
    def get_data_by_serial_number(self, serial_number):
        assert 'serial_number' in self.column_list()
        return HDDDataset(self.df[self.df['serial_number'] == serial_number], False).exclude_projection(['serial_number', 'model'])

    # Является ли значение в столбце одинаковым для всех строчек датасета
    def is_attribute_constant(self, attribute):
        assert attribute in self.column_list()
        return len(self.df[attribute].unique()) == 1

    # График зависимости одного столбца от другого
    def draw_dependancy(self, firstCol, secondCol):
        assert firstCol in self.column_list() and secondCol in self.column_list()
        x = self.df[firstCol]
        y = self.df[secondCol]
        plt.xlabel(firstCol)
        plt.ylabel(secondCol)
        plt.plot(x, y, 'o')
        plt.show()

    # Матрица корреляции
    def draw_correlation(self):
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
        plt.show()

    # Временной ряд для определенного серийного номера для определенного столбца
    def draw_time_series(self, serial_number, attribute):
        assert attribute in self.column_list()
        toDraw = self.get_data_by_serial_number(serial_number)
        plt.plot(toDraw.df['date'], toDraw.df[attribute])
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        plt.show()
