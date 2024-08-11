import numpy as np
from scipy.stats import friedmanchisquare

# Resultados do R² para cada configuração do KNeighborsRegressor
results = np.array([
    [0.95161987, 0.9519232,  0.96141972, 0.94772453, 0.95697277, 0.95012926,
     0.94820935, 0.94354469, 0.95218254, 0.94726068],  # KNeighborsRegressor_40
    [0.95243125, 0.95019362, 0.94778977, 0.95113588, 0.9538942,  0.9476762,
     0.95093788, 0.95715382, 0.95127121, 0.94778589],  # KNeighborsRegressor_41
    [0.95608801, 0.95112689, 0.95730445, 0.94719602, 0.94997499, 0.94623042,
     0.94505603, 0.95698058, 0.95223322, 0.94914124],  # KNeighborsRegressor_42
    [0.94160648, 0.95989229, 0.95284206, 0.95163033, 0.95171382, 0.95224674,
     0.94409437, 0.94422151, 0.95692574, 0.95399332],  # KNeighborsRegressor_43
    [0.9555801,  0.94827391, 0.9483821,  0.949941,   0.94461782, 0.95833984,
     0.9455853,  0.95507031, 0.95688996, 0.94428823],   # KNeighborsRegressor_44
     [0.95243125, 0.95019362, 0.94778977, 0.95113588, 0.9538942 , 0.9476762],# KNeighborsRegressor_45
     [0.95608801, 0.95112689, 0.95730445, 0.94719602, 0.94997499, 0.94623042,
 0.94505603, 0.95698058, 0.95223322, 0.94914124],  # KNeighborsRegressor_46
    [0.94160648, 0.95989229, 0.95284206, 0.95163033, 0.95171382, 0.95224674,
 0.94409437, 0.94422151, 0.95692574, 0.95399332],  # KNeighborsRegressor_47
 [0.9555801,  0.94827391, 0.9483821,  0.949941,   0.94461782, 0.95833984,
 0.9455853,  0.95507031, 0.95688996, 0.94428823],  # KNeighborsRegressor_48
 [0.93985318, 0.95054656, 0.95340685, 0.95180149, 0.94802761, 0.95223516,
 0.95577713, 0.95033393, 0.95133657, 0.95482465] # KNeighborsRegressor_49

])

# Realizar o teste de Friedman
statistic, p_value = friedmanchisquare(*results)

# Gerar a saída
output_text = f"Estatística de Friedman: {statistic}\nValor-p: {p_value}\n"

# Salvar os resultados em um arquivo de texto
with open("KnnRegressorTest_4.txt", "w") as file:
    file.write(output_text)

print("Resultados salvos em 'KnnRegressorTest_4.txt'")
