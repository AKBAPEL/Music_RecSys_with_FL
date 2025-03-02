import streamlit as st
import os
from pygame import mixer

st.title('Music for you')
st.sidebar.header("Управление")
mixer.init()
user_id = st.sidebar.text_input("Введите ваш айди/имя", key="current_user")


tracks = [f for f in os.listdir("music") if f.endswith(('.mp3', '.wav'))]
selected_track = st.sidebar.selectbox("Выберите трек", tracks)

music_name = st.text_input("Поиск музыки", key="music_name")

dislike, prev_track, pause_continue, next_track, like = st.columns(5, gap='large')


if 'paused' not in st.session_state:
    st.session_state.paused = False
if "current_track" not in st.session_state:
    st.session_state.current_track = None


user_events = []

with st.container():
    if dislike.button('💔', use_container_width=True):
        user_events.append([user_id, music_name])
    if like.button('🩷', use_container_width=True):
        st.success("oke")

    if prev_track.button('⏪', use_container_width=True):
        st.success("Музыка добавлена")
    if next_track.button('⏩', use_container_width=True):
        st.success("Дообучено!")


    if pause_continue.button(['⏸️','▶️'][st.session_state.paused*1], use_container_width=True):
        if st.session_state.current_track == selected_track and st.session_state.paused:
            mixer.music.unpause()
        elif not mixer.music.get_busy() or st.session_state.paused:
            mixer.music.load(os.path.join("music", selected_track))
            st.session_state.current_track = selected_track
            mixer.music.play()
        if mixer.music.get_busy() and not st.session_state.paused:
            mixer.music.pause()
        st.session_state.paused = not st.session_state.paused
        st.rerun()
st.subheader(f"Сейчас играет: 🎧 {selected_track}")

st.session_state
