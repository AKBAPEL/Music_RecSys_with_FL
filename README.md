# Music RecSys Model with Federated Learning

## 1. Описание проекта

Цель проекта ― разработать музыкальную рекомендательную систему с возможностью дообучения с помощью федеративного обучения:

* **Рекомендательная система**: базовая модель Matrix Factorization (ALS)
* **Федеративное обучение**: объединение моделей от клиентов (Clients → Master) для приватного дообучения
* **Веб-интерфейс**: Streamlit-приложение для загрузки данных, EDA и управления моделями
* **API-слой**: FastAPI для обучения/предсказания и обмена параметрами моделей

---

## 2. Участники команды

* **Белый Андрей**
* **Хамаганов Ильдар**
* **Купряков Дмитрий**

---



## 3. Структура репозитория

```text
Music_RecSys_with_FL/
├── .dockerignore
├── .gitignore
├── Dockerfile.fastapi
├── Dockerfile.streamlit
├── README.md
├── baselines/               ← Базовые решения и эксперименты
│   ├── baseline.ipynb
│   ├── baseline.md
│   ├── experiment_als.ipynb
│   ├── experiment_ranking_by_logreg.ipynb
│   ├── experiments.md
│   ├── metric-catboost-baseline.ipynb
│   └── new_baseline.ipynb
├── dataset.md               ← Описание датасета
├── docker-compose.yml       ← Конфигурация Docker
├── dockerfile               ← Альтернативный Dockerfile
├── logs/                    ← Директория для логов
├── models/                  ← Обученные модели
├── notebooks/               ← Аналитические ноутбуки
│   ├── improving baseline
│   ├── members_eda.ipynb
│   ├── songs_EDA.ipynb
│   └── train_EDA.ipynb
├── requirements.txt         ← Зависимости Python
└── web_service/             ← Веб-сервис (backend + frontend)
    ├── Dockerfile.fastapi
    ├── Dockerfile.streamlit
    ├── __init__.py
    ├── app.py               ← Streamlit frontend
    ├── data/                ← Данные для демонстрации
    │   └── train_truncated.csv
    ├── dockerfile
    ├── main.py              ← FastAPI backend
    ├── ml/                  ← Модуль машинного обучения
    │   ├── __init__.py
    │   ├── federated.py     ← Реализация федеративного обучения
    │   ├── inference.py     ← Логика предсказаний
    │   ├── preprocess.py    ← Предобработка данных
    │   └── train.py         ← Обучение моделей
    ├── patterns.py          ← Pydantic схемы
    ├── prepared_models/     ← Предобученные модели
    ├── store/               ← Хранилище моделей
    │   ├── __init__.py
    │   └── model_store.py
    └── users_models/        ← Пользовательские модели
```

---

## 4. Описание функционала

### 4.1. Backend (FastAPI)

* \`**/fit**\`

  * Параметры:

    ```json
    {
      "model_name": "string",
      "model_type": "ALS",
      "factors": 50,
      "iterations": 10,
      "regularization": 0.01,
      "alpha": 0.01
    }
    ```
  * Описание: запускает обучение новой ALS-модели с указанными параметрами. Сохраняет веса в папку `models/`.
  * Ответ:

    ```json
    { "status": "training_started", "model_id": "als_20230531_1530" }
    ```

* \`**/models**\`

  * Описание: возвращает список всех сохранённых моделей в `models/`.
  * Пример вывода:

    ```json
    [
      { "model_id": "als_20230531_1530", "created_at": "2025-05-31T15:30:00Z" },
      { "model_id": "als_20230530_1210", "created_at": "2025-05-30T12:10:00Z" }
    ]
    ```

* \`**/set**\`

  * Параметры (query-param): `model_id` (string)
  * Описание: помечает указанную модель активной (для последующих запросов `/predict`).
  * Ответ:

    ```json
    { "status": "active_model_updated", "model_id": "als_20230531_1530" }
    ```

* \`/**predict**\`

  * Параметры (query-param):

    * `user_id` (int) ― идентификатор пользователя
    * `n` (int) ― количество рекомендаций
  * Описание: возвращает топ-N рекомендованных треков для заданного `user_id` на основе текущей активной модели.
  * Пример ответа:

    ```json
    {
      "recommendations": [672, 1045, 2330, 19, 756]
    }
    ```

<!-- * **Федеративное обучение (Clients → Master)**

  * Клиенты (непоказано) обучают локальные копии моделей на собственных данных и отсылают градиенты/веса на endpoint `/federate` (опционально).
  * Master объединяет веса и пересчитывает глобальную модель.
  * В этой версии хранилище моделей локальное (папка `models/`), но легко адаптируется к облачным хранилищам. -->

### 4.2. Frontend (Streamlit)

* **Разведочный анализ (EDA)**

  * Загрузка своего CSV или использование дефолтного `data/train_truncated.csv`.
  * Вывод:

    * Head DataFrame (первые 50 записей сериализованных в JSON или raw, если загружено вручную).
    * Пропущенные значения (по каждому столбцу).
    * Статистика по нечисловым полям (`describe(exclude="int")`).
    * Распределение целевой переменной (`value_counts(normalize=True)`).
  * Графики:

    1. Гистограмма `source_system_tab` с разбивкой по `target`.
    2. Сетка из subplots по парам уникальных `source_system_tab`.
    3. Сетка из subplots по парам уникальных `source_type`.

* **Управление моделями**

  * Ввод имени модели и выбор типа (ALS).
  * Настройка гиперпараметров:

    * `factors` (10–100, шаг 1)
    * `iterations` (1–50, шаг 1)
    * `regularization` (0.0–1.0, шаг 0.01)
    * `alpha` (0.0–1.0, шаг 0.01)
  * Кнопка **«Обучить модель»** отправляет `POST /fit`.
  * Секция **«Available Models»**: при загрузке страницы делает `GET /models`, заполняет `selectbox` доступными `model_id`.

    * Кнопка **«Set Active Model»** отправляет `POST /set?model_id=<выбранный>` и показывает `st.success`.
  * Секция **«Получить треки»**: ввод `user_id` и `n`, кнопка **«Recommend»** отправляет `POST /predict?user_id=<>&n=<>`, выводит на экран топ-N рекомендаций.

---

## 5. Инфраструктура и технологии

* Язык разработки: **Python 3.11**
* Виртуальное окружение: **venv**
* Система контроля версий: **Git**
* Линтер: **flake8**
* Форматтер: **black**
* СУБД для продакшен-решений: **PostgreSQL** (через `SQLAlchemy`/`psycopg2`; в этой версии используется CSV-датасет)
* Фреймворк API: **FastAPI + Uvicorn**
* Веб-сервис (UI): **Streamlit**
* ML-библиотека: **implicit** (ALS на основе OpenMP, требует `libgomp1`)
* Контейнеризация: **Docker + Docker Compose**
* Сетевой драйвер: **bridge** (сервисам `backend` и `frontend` разрешён доступ друг к другу по DNS-именам)

---

## 6. Инструкция по использованию

1. **Клонировать репозиторий**

   ```bash
   git clone https://github.com/<username>/Music_RecSys_with_FL.git
   cd Music_RecSys_with_FL
   ```
2. **Запустить контейнеры**

   ```bash
   docker-compose up --build
   ```
4. **Проверить работу**

   * FastAPI Swagger: `http://localhost:8000/docs`
   * Streamlit UI: `http://localhost:8501`

---

https://docs.google.com/presentation/d/1ONfOWTdt6rwxtPwURvoFkH-_6j8t940mqe6YsjxwtEM/edit?slide=id.p#slide=id.p[Презентация]
