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

st.info('Selecione os dados')

# Framework:
# Caixas de seleção para gerar dados simulados
# Discretas
gender = st.selectbox('Gender:', ('Female','Male'))
seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
partner = st.selectbox('Partner:', ('Yes', 'No'))
dependents = st.selectbox('Dependents:', ('Yes', 'No'))
phone = st.selectbox('Phone Service:', ('Yes', 'No'))
lines = st.selectbox('Multiple Lines:', ('Yes', 'No', 'No phone service'))
internet = st.selectbox('Internet Service:', ('DSL', 'Fiber optic', 'No'))
security = st.selectbox('Online Security:', ('Yes', 'No', 'No internet service'))
backup = st.selectbox('Online Backup:', ('Yes', 'No', 'No internet service'))
protection = st.selectbox('Device Protection:', ('Yes', 'No', 'No internet service'))
techsupport = st.selectbox('Tech Support:', ('Yes', 'No', 'No internet service'))
streamingtv = st.selectbox('Streaming TV:', ('Yes', 'No', 'No internet service'))
streamingmovies = st.selectbox('Streaming Movies:', ('Yes', 'No', 'No internet service'))
contract = st.selectbox('Type of Contract:', ('One year', 'Month-to-month', 'Two year'))
billing = st.selectbox('Paperless Billing:', ('Yes', 'No'))
payment = st.selectbox('Payment Method:', ('Mailed check', 'Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)'))
# Continuas
tenure = st.slider('Tenure', min_value = 0, max_value = 72, value = 0)
monthlycharges = st.number_input('Monthly Charges:', min_value = 0, max_value = 150, value = 0)
totalcharges = st.number_input('Total Charges:', min_value = 0, max_value = 10000, value = 0)

# Criar um dicionario e um dataframe a partir dos dados simulados

data = {
    'Gender': gender,
    'SeniorCitizen': seniorcitizen,
    'Partner': partner,
    'Dependents': dependents,
    'PhoneService': phone,
    'MultipleLines': lines,
    'InternetService': internet,
    'OnlineSecurity': security,
    'OnlineBackup': backup,
    'OnlineProtection': protection,
    'TechSupport': techsupport,
    'StreamingTV': streamingtv,
    'StreamingMovies': streamingmovies,
    'Contract': contract,
    'PaperlessBilling': billing,
    'PaymentMethod':payment,
    'Tenure': tenure,
    'MonthlyCharges': monthlycharges,
    'TotalCharges': totalcharges
}

features = pd.DataFrame.from_dict([data])
st.dataframe(features)


# Aplicar nossa ferramenta de preprocessamento

# TESTE

# Aplicar model.predcit nesses dados depois do processamento

#prediction = model.predict(features)

# if predict == 1: mensagem de aviso
# else: mensagem de aviso (ai pode variar o tipo, usando warning ou success deve dar cores diferentes etc e tal)

