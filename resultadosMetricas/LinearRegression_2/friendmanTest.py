import numpy as np
from scipy.stats import friedmanchisquare

results = np.array([
    [0.8906360114285595, 0.8905962461496195, 0.8906943493687581, 0.890813244124437,
    0.8897047341867417, 0.8905878684313304, 0.8906616559816566, 0.8906217976258095, 0.8906838796393162,
    0.8904910432198946]
])

statistic, p_value = friedmanchisquare(*results)

output_text = f"Estatística de Friedman: {statistic}\nValor-p: {p_value}\n"

with open("./LinearRegression_2/LinearRegressionTest_2.txt", "w") as file:
    file.write(output_text)

print("Resultados salvos em 'LinearRegressionTest_2.txt'")
