import streamlit as st
import tensorflow as tf
import tensorflow_text

st.set_page_config(page_title="Sigma Ai | Aplikasi Penerjemah Bahasa Geser", page_icon="🤖")
st.title("Demo NMT Indonesia|Geser")

@st.cache_resource
def load_model(): 
    loaded = tf.saved_model.load('model', options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'))
    translate_func = loaded.signatures['serving_default']
    return loaded

model = load_model()

    
# Form to add your items
with st.form("my_form"):
    user_input = st.text_area("Masukan Kata...", max_chars=200)
    result = tf.constant([user_input])
    hasil = model.translate(result)
    translation = hasil[0].numpy().decode()

    submitted = st.form_submit_button("Terjemahkan")

    if submitted:
        st.write("Hasil Terjemahan")
        st.info(translation)
