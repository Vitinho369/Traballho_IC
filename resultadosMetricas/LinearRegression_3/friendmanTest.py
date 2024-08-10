import numpy as np
from scipy.stats import friedmanchisquare

# Resultados do R² para cada configuração do LinearRegression
results = np.array([
    [0.89340743, 0.89255708, 0.89423465, 0.88031887, 0.89031031, 
     0.89926654, 0.88691332, 0.88756558, 0.8927544, 0.89080426],  # LinearRegression_30
    [0.89531251, 0.8931737, 0.89418702, 0.89111897, 0.89246252, 
     0.89846482, 0.88701655, 0.87044002, 0.88779779, 0.88707344],  # LinearRegression_31
    [0.90110244, 0.89472945, 0.88437969, 0.88421359, 0.88727031, 
     0.89084441, 0.90035461, 0.89313914, 0.88843149, 0.88141356],  # LinearRegression_32
    [0.89420341, 0.87866437, 0.88907205, 0.89586583, 0.89098876, 
     0.89031365, 0.89957066, 0.88909783, 0.88782524, 0.89101476],  # LinearRegression_33
    [0.89047654, 0.8980145, 0.89172817, 0.87972387, 0.88267661, 
     0.89492985, 0.89214354, 0.88587871, 0.88978121, 0.90086497],  # LinearRegression_34
    [0.88334827, 0.88123311, 0.89164712, 0.88445431, 0.88847197, 
     0.90643069, 0.89001696, 0.89181354, 0.90128561, 0.88813723],  # LinearRegression_35
    [0.89745511, 0.88901818, 0.88842134, 0.8861167, 0.8881251, 
     0.89740106, 0.89558459, 0.89430781, 0.88746001, 0.88247021],  # LinearRegression_36
    [0.89416304, 0.8941719, 0.9006004, 0.88958277, 0.89947397, 
     0.88185689, 0.883855, 0.88507051, 0.88741752, 0.88977046],  # LinearRegression_37
    [0.89664846, 0.89390979, 0.88394773, 0.89445457, 0.88742121, 
     0.89244891, 0.88605666, 0.88729528, 0.88748869, 0.89727219],  # LinearRegression_38
    [0.89340743, 0.89255708, 0.89423465, 0.88031887, 0.89031031, 
     0.89926654, 0.88691332, 0.88756558, 0.8927544, 0.89080426],  # LinearRegression_39
])

# Realizar o teste de Friedman
statistic, p_value = friedmanchisquare(*results)

# Gerar a saída
output_text = f"Estatística de Friedman: {statistic}\nValor-p: {p_value}\n"

# Salvar os resultados em um arquivo de texto
with open("LinearRegressionTest_3.txt", "w") as file:
    file.write(output_text)

print("Resultados salvos em 'LinearRegressionTest_3.txt'")
