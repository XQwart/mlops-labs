# MLOps Lab 5: DVC Pipeline

Воспроизводимый pipeline обучения модели машинного обучения с использованием DVC (Data Version Control). Работа выполнена в рамках курса "Приложения машинного обучения".

## Задача

Регрессия на датасете Wine Quality (UCI Machine Learning Repository): предсказание органолептической оценки красного вина (quality) на основе физико-химических характеристик.

## Структура pipeline

Pipeline состоит из четырёх последовательных этапов, описанных в `dvc.yaml`:

1. **clean** - загрузка сырых данных с UCI, удаление дубликатов и пропусков, обрезка выбросов по квантилям 1% и 99%.
2. **featurize** - генерация новых признаков (total_acidity, free_to_total_sulfur, sugar_alcohol_ratio, acid_ph_ratio).
3. **train** - обучение RandomForestRegressor с подбором гиперпараметров через GridSearchCV (4-fold CV) на 80% выборки.
4. **evaluate** - оценка качества на отложенной выборке (20%) по метрикам RMSE, MAE, R².
