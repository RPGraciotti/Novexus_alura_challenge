### Arquivo de preparação que será convertido em .py

import numpy as np
import pandas as pd
import json


def leitura_dados(path):

    with open(file = path) as f:
        file = json.load(f)

    df = pd.json_normalize(file)

    return(df)


def limpeza_e_tratamento(df):
    
    df.loc[df[df['account.Charges.Total'] == " "].index, 'account.Charges.Total'] = df.loc[df[df['account.Charges.Total'] == " "].index, 'account.Charges.Monthly']

    df['account.Charges.Total'] = df['account.Charges.Total'].astype(float)

    df_sem_vazio = df[df['Churn'] != ''].copy()
    df_sem_vazio.reset_index(drop = True, inplace = True)

    colunas_binarias = ['Churn', 'customer.gender', 'customer.SeniorCitizen', 'customer.Partner', 'customer.Dependents', 'phone.PhoneService', 'account.PaperlessBilling']

    bin = {
        'No': 0,
        'Yes' : 1,
        'Female' : 0,
        'Male' : 1
    }

    df_sem_vazio[colunas_binarias] = df_sem_vazio[colunas_binarias].replace(bin)

    df_limpo = df_sem_vazio.copy()
    
    df_limpo.drop(columns = ['customer.gender', 'phone.PhoneService'], inplace = True)

    df_dummies = pd.get_dummies(df_limpo, columns = ['phone.MultipleLines','internet.InternetService','internet.OnlineSecurity','internet.OnlineBackup','internet.DeviceProtection','internet.TechSupport','internet.StreamingTV','internet.StreamingMovies','account.Contract','account.PaymentMethod'])

    cols_to_remove = df_dummies.filter(like = 'No internet service').columns
    df_dummies.drop(columns = cols_to_remove, inplace = True)

    df_final = df_dummies.copy()

    return(df_final)




