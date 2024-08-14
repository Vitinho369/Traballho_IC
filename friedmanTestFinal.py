import os
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

path_name = './friedmanTestFinal'
os.makedirs(path_name, exist_ok=True)

def wilcoxon_test(data, col1, col2):
    try:
        stat, p_value = wilcoxon(data[col1], data[col2], zero_method='wilcox', correction=False)
        return f'{col1} & {col2} : p-valor = {p_value}\n'
    except Exception as e:
        return f'Erro ao realizar o teste de Wilcoxon entre {col1} e {col2}: {e}\n'

resultsRegression = np.array([
    [0.9467741904064807, 0.9464109307593411, 0.9466233097087805, 0.9458459345559425,
    0.9462158906386312, 0.9469100833097566, 0.947128608011243, 0.9468486812516896, 
    0.9463595595288993, 0.9464822885721615], #Knn 0

    [0.8906360114285594, 0.8905962461496195, 0.890694349368758, 0.8908132441244367,
    0.8897047341867413, 0.8905878684313306,  0.8906616559816565, 0.8906217976258095,
    0.8906838796393164, 0.8904910432198945], #Linear 3

    [0.9277299574543466, 0.9285621195713711, 0.9294077443065587, 0.9303062768118341,
     0.9293394784236328, 0.9252435724978463, 0.9278079446649382, 0.9217704512901843,
     0.9295173079089094, 0.9259404071533434], #Mlp 0
    
    [0.927794476454898, 0.9229465022199064, 0.941340817515776,0.5532464121439713, 
     0.9309701092271112, 0.5382724612876769, 0.9190142367862231,
      -0.15763012642860466,  0.7376378076687894, 0.9199154553841099], #DecisionTree 2

])

nameFile = ["KnnRegressorTest", "DecisionTreeRegressorTest", "LinearRegressionTest", "MlpRegressorTest"]

statistic, p_value = friedmanchisquare(*resultsRegression)

output_text = ""
output_text = f"Estatística de Friedman: {statistic}\nValor-p: {p_value}\n"

with open(f"./friedmanTestFinal/resultsRegression.txt", "w") as file:
    file.write(output_text)

print(f"Resultados salvos em './friedmanTestFinal/resultsRegression.txt'")

if p_value < 0.05:
    path_name = './friedmanTestFinal/wilcoxonTest'
    os.makedirs(path_name, exist_ok=True)

    output_text = ""
    # Gerar os pares de colunas (modelos) para comparar
    for i in range(len(resultsRegression)):
        for j in range(i + 1, len(resultsRegression)):
            output_text += wilcoxon_test(resultsRegression, i, j)

    # Salvar os resultados em um arquivo de texto
    with open(f"./friedmanTestFinal/wilcoxonTest/resultsRegression.txt", "w") as file:
        file.write(output_text)

    print(f"Resultados salvos em ./friedmanTestFinal/wilcoxonTest/resultsRegression.txt")
