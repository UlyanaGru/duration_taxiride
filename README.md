## Прогнозирование длительности поездок такси

Данный [проект](https://github.com/UlyanaGru/duration_taxiride/blob/master/duration_taxiride.ipynb) направлен на предсказание длительности поездок такси в Нью-Йорке с использованием различных алгоритмов машинного обучения.

### Данные

Данные включают информацию о поездках такси:
- Временные характеристики (время, день недели, месяц)
- Географические координаты (широта/долгота начала и окончания)
- Расстояние поездки
- Погодные условия
- Праздничные дни
- OSRM-метрики (расчетное время и расстояние)

### Целевая переменная

**trip_duration** - длительность поездки в секундах (предсказывается в логарифмическом масштабе)

### Использованные алгоритмы

#### 1. Linear Regression
- **Тип**: Линейная модель
- **Особенности**: Базовая модель для установления бейзлайна
- **Преимущества**: Быстрая обучение, интерпретируемость
- **Недостатки**: Предполагает линейную зависимость, чувствительна к выбросам

#### 2. Decision Tree
- **Тип**: Дерево решений
- **Особенности**: Склонен к переобучению без регуляризации
- **Гиперпараметры**: `max_depth` от 7 до 20
- **Результат**: Оптимальная глубина найдена через перебор

#### 3. Random Forest
- **Тип**: Ансамбль деревьев
- **Гиперпараметры**:
  - `n_estimators=200`
  - `max_depth=12`
  - `min_samples_split=20`
  - `random_state=42`
- **Преимущества**: Уменьшает переобучение, robust к выбросам

#### 4. Gradient Boosting
- **Тип**: Последовательный ансамбль
- **Гиперпараметры**:
  - `learning_rate=0.5`
  - `n_estimators=100`
  - `max_depth=6`
  - `min_samples_split=30`
- **Преимущества**: Высокая точность, хорошая обобщающая способность

#### 5. XGBoost (Extreme Gradient Boosting)
- **Тип**: Оптимизированный градиентный бустинг
- **Гиперпараметры**:
  - `eta=0.1`
  - `max_depth=6`
  - `subsample=0.9`
  - `colsample_bytree=0.9`
  - `early_stopping_rounds=20`
- **Особенности**: Регуляризация, обработка пропущенных значений
- **Преимущества**: Высокая производительность, скорость обучения

### Метрики оценки

### Основные метрики:
- **RMSLE** (Root Mean Squared Logarithmic Error) - основная метрика
- **MeAE** (Median Absolute Error) - медианная абсолютная ошибка в минутах
- **Feature Importance** - важность признаков для интерпретации модели

### Результаты моделей:
| Модель | Train RMSLE | Valid RMSLE |
|--------|-------------|-------------|
| Linear Regression | ~0.42 | ~0.43 |
| Decision Tree | ~0.00 | ~0.45 |
| Random Forest | ~0.32 | ~0.35 |
| Gradient Boosting | ~0.30 | ~0.33 |
| XGBoost | Лучший результат | на тесте |

![1](https://github.com/UlyanaGru/duration_taxiride/blob/master/pic/duration_hour.png)
![2](https://github.com/UlyanaGru/duration_taxiride/blob/master/pic/geographical_endbeginnig.png)
![3](https://github.com/UlyanaGru/duration_taxiride/blob/master/pic/rmse_rmsle.png)

### Использование

1. Предобработка данных
2. Feature engineering
3. Обучение моделей
4. Предсказание на тестовых данных
5. Создание submission-файла

```bash
# Пример создания submission-файла
submission = pd.DataFrame({'id': test_id, 'trip_duration': predictions})
submission.to_csv('submission.csv', index=False)