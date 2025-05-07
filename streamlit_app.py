import streamlit as st
import requests
import logging
from logging.handlers import RotatingFileHandler
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Логи
logger = logging.getLogger("streamlit")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    "logs/streamlit.log",
    maxBytes=1024 * 1024,
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

BASE_URL = "http://backend:8000"


def eda_section(train: pd.DataFrame):
    st.subheader("Exploratory Data Analysis")

    try:
        # Первый график
        st.write("Распределение Source System Tab")
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
        st.write("Аналитика Source System Tabs")
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
        st.write("Распределение Source Type")
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

    except Exception as e:
        st.error(f"Error generating EDA: {str(e)}")
        logger.error(f"EDA error: {str(e)}")


def main():
    st.title("Music Recommendation System")

    train = None
    # Загрузка данных
    with st.expander("Upload Dataset"):
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            try:
                train = pd.read_csv(uploaded_file)
                train['source_type'] = train['source_type'].fillna("isnan")
                train['source_screen_name'] = train['source_screen_name'].fillna("isnan")
                train['source_system_tab'] = train['source_system_tab'].fillna("isnan")
                st.success("Dataset uploaded successfully")
                logger.info("New dataset uploaded")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                logger.error(f"File load error: {str(e)}")

    if train is None:
        try:
            train = pd.read_csv("train.csv") 
        except Exception as e:
            st.error("Default dataset not found")
            logger.error(f"Default dataset load error: {str(e)}")
            return

    # EDA
    with st.expander("Exploratory Data Analysis"):
        if train is not None:
            eda_section(train)
        else:
            st.warning("Upload data to see EDA")


    


if __name__ == "__main__":
    main()
    
