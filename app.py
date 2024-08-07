import streamlit as st
from multiapp import MultiApp
import N2A
import N3B
import N2B
import N3A
import Forecasts

app = MultiApp()

# Add all your applications here
app.add_app("N2A Prediction", N2A.app)
app.add_app("N3B Prediction", N3B.app)
app.add_app("N2B Prediction", N2B.app)
app.add_app("N3A Predictions", N3A.app)
app.add_app("Forecasts", Forecasts.app)
# The main app
app.run()
