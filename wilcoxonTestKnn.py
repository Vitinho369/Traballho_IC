import numpy as np
from scipy.stats import wilcoxon

def wilcoxon_test(data, col1, col2):
    try:
        stat, p_value = wilcoxon(data[col1], data[col2], zero_method='wilcox', correction=False)
        return f'{col1} & {col2} : p-valor = {p_value}\n'
    except Exception as e:
        return f'Erro ao realizar o teste de Wilcoxon entre {col1} e {col2}: {e}\n'
    
results = np.array([
    0.464895523268825, #Valor p de KnnRegressor_0
    0.9755303035128957, #Valor p de KnnRegressor_1
    5.4125671148260496e-05, #Valor p de KnnRegressor_2
    4.656698437092106e-09, #Valor p de KnnRegressor_3
    0.9036712329146966 #Valor p de KnnRegressor_4
])

for pValor in results:

    if pValor < 0.05:
          
        output_text = ""
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                output_text += wilcoxon_test(results, i, j)

        # Salvar os resultados em um arquivo de texto
        with open("./wilcoxonTest/KnnRegressorWilcoxonTest.txt", "a") as file:
            file.write(output_text)

        print("Resultados salvos em './wilcoxonTest/KnnRegressorWilcoxonTest.txt'")
