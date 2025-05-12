import streamlit as st
import requests
import logging
from logging.handlers import RotatingFileHandler
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from patterns import FitRequest

st.set_page_config(layout="wide")
st.title("Music Recommendation System")

API_URL = " http://127.0.0.1:8000"

def eda_demonstration(train: pd.DataFrame):
    st.header("Разведочный Анализ Данных")
    st.subheader("Train DataFrame", divider=True)
    st.dataframe(data=train.head())

    st.badge("Пропущенные значения", color="green")
    missing_values = train.isnull().sum()
    st.dataframe(pd.DataFrame(missing_values, columns=['count']))

    st.badge("Характеристики столбцов")
    st.dataframe(pd.DataFrame(train.describe(exclude='int')), use_container_width=False)

    st.badge("Распределение целевой переменной", color="green")
    target_dist = train['target'].value_counts(normalize=True)
    st.dataframe(target_dist)

    # Первый график
    st.badge("Распределение Source System Tab")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=train,
        x="source_system_tab",
        hue="target",
        multiple="dodge",
        shrink=.9,
        ax=ax
    )
    st.pyplot(fig)

    # Второй набор графиков
    st.badge("Аналитика Source System Tabs")
    unique_tabs = train['source_system_tab'].unique()
    n_cols = 4
    n_rows = (len(unique_tabs) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.ravel()

    for i in range(0, len(unique_tabs), 2):
        tabs_subset = unique_tabs[i:i + 2]
        sns.histplot(
            data=train[train['source_system_tab'].isin(tabs_subset)],
            x='source_system_tab',
            hue='target',
            multiple="dodge",
            shrink=.9,
            ax=axes[i // 2]
        )

    # Удаление пустых графиков
    for j in range(i // 2 + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

    # Третий
    st.badge("Распределение Source Type")
    unique_tabs = train['source_type'].unique()

    n = len(unique_tabs)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.ravel()

    for i in range(0, len(unique_tabs), 2):
        sns.histplot(data=train[train['source_type'].isin(unique_tabs[i:i + 2])], x='source_type', hue='target',
                        multiple="dodge", shrink=.9, ax=axes[i // 2])

    for j in range(i // 2 + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)



train = None

# Загрузка датасета
with st.expander("Загрузить датасет"):
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    if uploaded_file:
        try:
            train = pd.read_csv(uploaded_file)
            train['source_type'] = train['source_type'].fillna("isnan")
            train['source_screen_name'] = train['source_screen_name'].fillna("isnan")
            train['source_system_tab'] = train['source_system_tab'].fillna("isnan")
            train['msno'] = train['msno'].astype(str)
            st.success("Dataset uploaded successfully")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

if train is None:
    train = pd.read_csv("../train.csv")
    train['msno'] = train['msno'].astype(str)

# EDA
with st.expander("Получить разведку"):
    eda_demonstration(train)



# Создание моделей
with st.expander("Управление моделями"):
    st.subheader("Создать свою модель")
    model_name = st.text_input("Model Name", key='My awesome model')
    model_type = st.selectbox("Model Type", ["ALS"]) # добавить может быть Matrix Factorization
    params = {
        "factors": st.number_input("Factors", 10, 100, 50),
        "iterations": st.number_input("Iterations", 1, 50, 10),
        "regularization": st.number_input("Regularization", 0.0, 1.0, 0.01),
        "alpha": st.number_input("Alpha", 0.0, 1.0, 0.01)
    }
    if st.button("Обучить модель"):
        response = requests.post(
            f"{API_URL}/fit",
            json={
                "model_name": model_name,
                "model_type": model_type,
                **params
            }
        )
        st.write(response.json())

    st.subheader("Available Models")
    models = requests.get(f"{API_URL}/models").json()
    selected_model = st.selectbox("Active Model", [m["model_id"] for m in models])
    if st.button("Set Active Model"):
        requests.post(f"{API_URL}/set", params={"model_id": selected_model})
        st.success("Model updated")


    st.subheader("Получить треки")
    user_id = st.number_input("User ID", min_value=0, value=0)
    n = st.number_input("Number of Recommendations", 5, 20, 10)
    if st.button("Recommend"):
        response = requests.post(
            f"{API_URL}/predict",
            params={"user_id": user_id, "n": n}
        )
        if response.status_code == 200:
            recommendations = response.json()["recommendations"]
            st.write("Top Recommendations:")
            for i, song in enumerate(recommendations, 1):
                st.write(f"{i}. Song ID: {song}")
        else:
            st.error("Error getting recommendations")
