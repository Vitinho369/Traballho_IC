import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from consts import MLP_CONFIGURER, DESCION_TREE_CONFIGURER, KNEIGHBORS_CONFIGURER, LINEAR_REGRESSION_CONFIGURER, CSV_FILE
import os

# Função para converter de one-hot encoding para valores nominais
def convert_one_hot_to_nominal(df, prefix):
    cols = [col for col in df.columns if col.startswith(prefix)]
    df[prefix] = df[cols].idxmax(axis=1).str.replace(prefix + "_", "")
    df.drop(cols, axis=1, inplace=True)

# Carregar a base de dados Titanic
df = pd.read_csv(CSV_FILE)

# Selecionar colunas relevantes para regressão
df = df[['tempreature', 'humidity', 'water_level', 'N', 'P', 'K', 'Fan_actuator_OFF', 'Fan_actuator_ON', 'Watering_plant_pump_OFF', 'Watering_plant_pump_ON', 'Water_pump_actuator_OFF', 'Water_pump_actuator_ON']]

# Definir recursos (features) e alvo (target)
features = df.drop(['water_level'], axis=1)
targets = df[['water_level']]

# Identificar colunas numéricas e categóricas
numeric_features = ['tempreature', 'humidity', 'N', 'P', 'K']
categorical_features = ['Fan_actuator_OFF', 'Fan_actuator_ON', 'Watering_plant_pump_OFF', 'Watering_plant_pump_ON', 'Water_pump_actuator_OFF', 'Water_pump_actuator_ON']

# Pré-processamento
# Criar transformers para colunas numéricas e categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Criar o pré-processador usando ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Achatar os vetores coluna em arrays unidimensionais
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
targets = targets.values.ravel()

# Definindo modelos
def evaluate_model(model_name, pipeline, random_state):
    cv = KFold(n_splits=10, random_state=random_state, shuffle=True)
    scores = cross_val_score(pipeline, features, targets, cv=cv, scoring='r2')
    
    path_name = f'./resultadosMetricas/{model_name[:-1]}'
    os.makedirs(path_name, exist_ok=True)
    
    with open(f'{path_name}/{model_name[:-1]}_metrics.txt', 'a') as file:
        file.write(f"{model_name}\n")
        file.write(f"{model_name} r2 Mean: {scores.mean()}\n")
        file.write(f"{model_name} r2 Std: {scores.std()}\n")
        file.write(f"{model_name} r2 Scores: {scores}\n\n")

        print(f"{model_name} r2 Scores: {scores}")
        print(f"{model_name} r2 Mean: {scores.mean()}")
        print(f"{model_name} r2 Std: {scores.std()}")

def graphicDispersion(model_name,pipeline, X_train, y_train,X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
   
    # Criar e exibir o gráfico de dispersão
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title(f'{model_name[:-2]}: Valores Reais vs. Preditos')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)  # Linha de referência
    plt.grid(True)
    plt.savefig(f'./resultadosMetricas/{model_name[:-1]}/{model_name[:-1]}.png')
    plt.close()

index = 0
for config in LINEAR_REGRESSION_CONFIGURER:
    # Definir o pipeline para Regressão Linear
     for i in range(1,10):
        linear_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression(**config))
        ])

        print("Linear Regression")
        model_name = f"LinearRegression_{index}"
        evaluate_model(model_name, linear_pipeline, i)
        graphicDispersion(model_name, linear_pipeline, X_train, y_train, X_test, y_test)
        index += 1

index = 0
for config in KNEIGHBORS_CONFIGURER:
    # K-Nearest Neighbors Regressor
     for i in range(1,10):
        knn_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(**config))  # Número de vizinhos
        ])

        print("K-Nearest Neighbors")
        model_name = f"KNeighborsRegressor_{index}"
        evaluate_model(model_name, knn_pipeline, i)
        graphicDispersion(model_name, knn_pipeline, X_train, y_train, X_test, y_test)
        index += 1

index = 0
for config in DESCION_TREE_CONFIGURER:
    # Árvore de Decisão Regressora
    
    for i in range(1,10):
        config['random_state'] = i
        tree_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(**config))
        ])
        print("Decision Tree") 
        model_name = f"DecisionTree_{index}"
        evaluate_model(model_name, tree_pipeline, i)
        graphicDispersion(model_name, tree_pipeline, X_train, y_train, X_test, y_test)
        index += 1

index = 0
for config in MLP_CONFIGURER:
    # MLP Regressor
    
    for i in range(1,10):
        config['random_state'] = i
        mlp_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', MLPRegressor(**config))
        ])
        print("MLP Regression")
        model_name = f"MLPRegressor_{index}"
        evaluate_model(model_name, mlp_pipeline, i)
        graphicDispersion(model_name, mlp_pipeline, X_train, y_train, X_test, y_test)
        index += 1