import numpy
import pandas as pd
import streamlit as st
from tensorflow import keras

st.write("""
         # Real Estate App
         
         Identify up and comming Montreal neighborhoods and undervalued condos!
         
         """)

# Load keras model
model = keras.models.load_model('tf_linear_model')
