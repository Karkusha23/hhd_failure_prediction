## Постановка задачи
Учитывая данные мониторинга состояния диска [S.M.A.R.T]([https://ru.wikipedia.org/wiki/S.M.A.R.T](https://ru.wikipedia.org/wiki/S.M.A.R.T).) и данных о неисправностей, необходимо придумать собственное решение и определить, выйдет ли из строя каждый диск в течение следующих 30 дней.
## Описание датасета
Датасет представлен в виде `csv` файла, содержащего записи SMART данных различных дисков в течение периодов времени. 
Атрибуты:
- `date` - дата записи
- `failure` - отказ диска
- `model` - название модели
- `serial_number` - серийный номер диска
- `smart 5` - число обнаруженных ошибок чтения\записи
- `smart 9` - время работы
- `smart 187` - число ошибок, которое накопитель сообщил хосту (интерфейсу компьютера) при любых операциях
- `smart 188` - число прерванных операций в связи с таймаутом HDD
- `smart 192` - число циклов выключений или аварийных отказов
- `smart 197` - число секторов, являющихся кандидатами на замену.
- `smart 198` - число некорректируемых (средствами диска) секторов. (критические дефекты)
- `smart 199` - число ошибок, возникающих при передаче данных по интерфейсу
- `smart 240` - общее время нахождения блока головок в рабочем положении в часах
- `smart 241` - полное число записанных секторов.
- `smart 242` - полное число прочитанных секторов.
## Обработка данных
### Очистка данных
В результате предварительного анализа данных было выяснено следующее:
- признак `model` может быть удалён, по причине того, что модель всего одна
- признак `capacity_bytes` может быть удалён по той же причине
- признак `smart_raw_198` равен признаку `smart_raw_197`, поэтому он также может быть удалён (это было замечено по матрице корреляций, а также это указано в [документации seagate](https://t1.daumcdn.net/brunch/service/user/axm/file/zRYOdwPu3OMoKYmBOby1fEEQEbU.pdf#page=7.10)
- отсутствуют `nan` значения
### Нормализация
Были протестированы следующие методы нормализации данных:
- логарифмизация
- преобразования Бокса-Кокса
- смешанные преобразования логарифмизации и Бокса-Кокса (выборочно для различных столбцов)
- **преобразования Yeo-Johnson** 
- нормализация по формулам из [документации seagate](https://t1.daumcdn.net/brunch/service/user/axm/file/zRYOdwPu3OMoKYmBOby1fEEQEbU.pdf#page=7.10)
### Получение новых признаков
По причине наличия временной составляющей в задаче, было принято решение добавить следующие признаки:
- сдвинутые признаки (значение того же самого признака, но за предыдущий день) (`shift_smart_n_raw_`)
- разность признаков (разность между текущим значением признака и сдвинутым) (`diff_smart_n_raw_`)
## Методы решения
Были использованы следующие методы решения
- Модели градиентного бустинга (XGBoost, LightGBM, CatBoost)
- Двухслойный XGBoost (подробнее в документации)
- Ансамбль
- Блендинг
- Стекинг
## Полученные результаты
Лучшие результаты экспериментов

**Для класса 1**

| Название метрики / <br>название модели   | accuracy  | precision     | recall        | f1            | ROC-AUC       |
| ---------------------------------------- | --------- | ------------- | ------------- | ------------- | ------------- |
| CatBoost                                 | **0.999** | 0.966         | 0.554         | 0.704         | 0.777         |
| LightGBM                                 | **0.999** | 0.791         | 0.490         | 0.605         | 0.745         |
| XGBoost                                  | **0.999** | 0.940         | 0.677         | **0.787** II  | 0.838         |
| Двухслойный XGBoost                      | **0.999** | 0.808         | **0.696** III | 0.748         | **0.848** III |
| Ансамбль                                 | **0.999** | **0.969** III | 0.624         | 0.761         | 0.813         |
| Блендинг                                 | **0.999** | **0.986** I   | 0.564         | 0.717         | 0.782         |
| Блендинг (SMOTE)                         | **0.999** | 0.915         | **0.733** II  | **0.814** I   | **0.866** II  |
| Стекинг (meta_model: CatBoost)           | **0.999** | 0.519         | **0.789** I   | 0.627         | **0.894** I   |
| Стекинг (meta_model: LogisticRegression) | **0.999** | **0.979** II  | 0.642         | **0.775** III | 0.821         |

**Для класса 0** метрики precision, recall, f1 равны 1

_Подробности реализации в соответствующем md файле._
