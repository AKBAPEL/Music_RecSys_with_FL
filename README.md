# Music RecSys model with a Federated Learning

## Описание проекта
Создание музыкальной рекомендательной системы с реализацией федеративного обучения

## Участники команды
- Белый Андрей
- Хамаганов Ильдар
- Купряков Дмитрий

## Описание функциональностей проекта
- Рекомендательная система 
- Дообучение для конкретного  пользователя 
- Стягивание разосланных моделей (Clients->Master)
- Приложение 
- Веб-интерфейс -> app.py

## Описание инфраструктуры
- Язык разработки: Python 3.11
- Менеджмент версий python: venv
- Система контроля версий кода: git
- Линтер: flake8
- Автоформаттер: black
- Хранилище данных: PostgreSQL
- Контейнеризация: Docker
- API: FastAPI
- Web-service: Streamlit
- 

## Запуск API
`uvicorn fast_api:app & npx localtunnel --port 8000 --subdomain fastapi & wget -q -O - https://loca.lt/mytunnelpassword`

