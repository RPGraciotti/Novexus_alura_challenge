# Aplicação no streamlit

# Importando bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Importando scripts de preprocessamento
from utils import leitura_dados
from utils import limpeza_e_tratamento

# Importando modelo
model = joblib.load(r'./modelo/modelo_NB.sav')

st.title('NOVEXUS')

st.image('./logos/Logo (2).png')
st.write('# Identificação de churn de clientes')

