## Структура Бейзлайна (Итоговая версия)
В качестве бейзлайн-модели для рекоммендации треков была выбрана модель SVD библиотеки scikit-surprise

- Были подобраны оптимальные гиперпараметры для модели.
- Была реализована функция которая выдает топ пар user_id/song_id для каждого пользователя с соответствуующими вероятностями.
- Была посчитана кастомная метрика с использованием предсказаний о недостающих треках с помощью модели Catboost


Важно отметить, что метрика на нашей задаче достаточно дизбалансированна, т.к нам гораздо важнее не рекоммендовать пользователям не релевантный трек, чем отбросить тот трек, который ему бы понравился.
## Метрики
- MAE:  0.3935
- RMSE: 0.4508
- Custom-metric(accuracy): 0.55
- Custom-metric(precision) 60% пользователей с precision > 0.8.

Перед нами предстоит задача научиться лучше обрабатывать "холодных" пользователей, это таких, про которых мы еще почти ничего не знаем. Например новые пользователи.

## UPD
Неожиданно в дополнение была обучена модель ALS из библиотеки implicit, которая дала качество еще лучше

## Метрики
- AUC:  0.7454
- precision:  0.8439
- recall:  0.2054
