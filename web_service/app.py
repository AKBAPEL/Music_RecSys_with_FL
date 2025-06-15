import logging
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

logger = logging.getLogger("frontend")
logger.setLevel(logging.INFO)

log_dir = "/app/logs"
if not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Created log directory: {log_dir}")
    except Exception as e:
        logger.error(f"Failed to create log directory: {str(e)}")
        log_dir = None

if log_dir:
    log_file = os.path.join(log_dir, "frontend.log")
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("File handler initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize file handler: {str(e)}")
        log_dir = None

if not log_dir:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.warning("Using stream handler instead of file handler")

st.set_page_config(layout="wide")
st.title("Music Recommendation System")

API_URL = "http://backend:8000"


@st.cache_data
def load_data(path: str = "data/train_truncated.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["msno"] = df["msno"].astype(str)
    df["source_type"] = df["source_type"].fillna("isnan")
    df["source_screen_name"] = df["source_screen_name"].fillna("isnan")
    df["source_system_tab"] = df["source_system_tab"].fillna("isnan")
    return df


train_df = load_data()

st.sidebar.header("Model & Federated Learning")
model_name = st.sidebar.text_input("Model Name", "my_model")
model_type = st.sidebar.selectbox("Model Type", ["ALS"])
factors = st.sidebar.slider("Factors", 10, 200, 50)
iterations = st.sidebar.slider("Iterations", 1, 50, 10)
regularization = st.sidebar.number_input("Regularization", 0.0, 1.0, 0.01)
alpha = st.sidebar.number_input("Alpha", 0.0, 2.0, 1.0)

if st.sidebar.button("Train Model"):
    payload = {
        "model_name": model_name,
        "model_type": model_type,
        "factors": factors,
        "iterations": iterations,
        "regularization": regularization,
        "alpha": alpha,
    }
    logger.info("Fit request payload: %s", payload)
    try:
        r = requests.post(f"{API_URL}/fit", json=payload, timeout=10)
        if r.ok:
            st.sidebar.success("Training started")
            st.sidebar.json(r.json())
        else:
            st.sidebar.error(f"Training error: {r.text}")
    except Exception as e:
        logger.error("Training request failed: %s", str(e))
        st.sidebar.error(f"Request failed: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.subheader("Active Model")
try:
    models: List[Dict[str, Any]] = requests.get(f"{API_URL}/models", timeout=5).json()
    ids = [m["model_id"] for m in models]
    active = st.sidebar.selectbox("Select Model", ids)
    if st.sidebar.button("Set Active Model"):
        logger.info("Set active model: %s", active)
        try:
            s = requests.post(f"{API_URL}/set", params={"model_id": active}, timeout=5)
            if s.ok:
                st.sidebar.success(f"Active: {active}")
            else:
                st.sidebar.error(f"Set model failed: {s.text}")
        except Exception as e:
            logger.error("Set model request failed: %s", str(e))
            st.sidebar.error(f"Request failed: {str(e)}")
except Exception as e:
    logger.error("Models fetch error: %s", str(e))
    st.sidebar.error(f"Cannot load models: {str(e)}")

if st.sidebar.button("Run Federated Training"):
    logger.info("Starting federated learning")
    try:
        f = requests.post(f"{API_URL}/federate", timeout=30)
        if f.ok:
            st.sidebar.success("Federated model ready")
            st.sidebar.json(f.json())
        else:
            err = f.json().get("detail", f.text)
            logger.error("Federation error: %s", err)
            st.sidebar.error(err)
    except Exception as e:
        logger.error("Federation request failed: %s", str(e))
        st.sidebar.error(f"Request failed: {str(e)}")

with st.expander("Exploratory Data Analysis", expanded=True):
    st.header("Exploratory Data Analysis")
    st.dataframe(train_df.head())
    st.subheader("Missing Values")
    miss = train_df.isnull().sum().to_frame("count")
    st.dataframe(miss)
    st.subheader("Target Distribution")
    st.bar_chart(train_df["target"].value_counts(normalize=True))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=train_df,
        x="source_system_tab",
        hue="target",
        multiple="dodge",
        shrink=0.9,
        ax=ax,
    )
    st.pyplot(fig)

st.header("Recommendations")
user_id = st.number_input("User ID", min_value=0, value=0)
n = st.slider("Recommendations count", 1, 20, 10)
if st.button("Recommend"):
    logger.info("Predict request: user=%d, n=%d", user_id, n)
    try:
        resp = requests.post(
            f"{API_URL}/predict", params={"user_id": user_id, "n": n}, timeout=10
        )
        if resp.ok:
            recs = resp.json().get("recommendations", [])
            st.write("### Top Recommendations")
            for i, song in enumerate(recs, 1):
                st.write(f"{i}. {song}")
        else:
            err = resp.text
            logger.error("Predict error: %s", err)
            st.error(err)
    except Exception as e:
        logger.error("Predict request failed: %s", str(e))
        st.error(f"Request failed: {str(e)}")
