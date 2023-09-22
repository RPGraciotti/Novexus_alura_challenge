# Preparação dos dados e fit do modelo

import numpy as np
import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
import warnings


with open(file = './data/Telco-Customer-Churn.json') as f:
    file = json.load(f)

df = pd.json_normalize(file)

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

df_sample = df_final.sample(frac = 1, random_state = 42).reset_index(drop = True)
y_or = df_sample['Churn']
y_or = y_or.rename('target')
X_or = df_sample.drop(['Churn','customerID'], axis = 1)
split = train_test_split(X_or, y_or, stratify = y_or, test_size = 0.2, random_state = 42)

smote_enn = SMOTEENN(random_state = 42)
X_resampled, y_resampled = smote_enn.fit_resample(split[0], split[2])

cv = RepeatedStratifiedKFold(random_state = 42)

scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

warnings.filterwarnings('ignore')
# Definição do espaço de parâmetros
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

scoring = {'Acurácia': 'accuracy', 'Precisão': 'precision','Recall': 'recall','F1': 'f1', 'ROC_AUC':'roc_auc'}

m_NB = GaussianNB()

# Busca
gs_NB = RandomizedSearchCV(estimator = m_NB, 
                 param_distributions = params_NB, 
                 cv = cv,
                 verbose = 1, 
                 scoring = scoring,
                 n_iter = 100,
                 random_state = 42,
                 refit = 'F1') 
gs_NB.fit(X_or, y_or)

# Salvando objeto de resultados
params = list(gs_NB.best_params_.values())[0]

# Modelo fitado com variáveis de treino e parâmetros otimizados
m_NB = GaussianNB(var_smoothing = params)
m_NB.fit(X = X_resampled, y = y_resampled)

joblib.dump(m_NB, filename = './modelo/modelo_NB.sav')