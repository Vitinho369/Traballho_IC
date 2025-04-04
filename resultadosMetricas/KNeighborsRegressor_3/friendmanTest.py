import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
def wilcoxon_test(data, col1, col2):
    try:
        stat, p_value = wilcoxon(data[col1], data[col2], zero_method='wilcox', correction=False)
        return f'{col1} & {col2} : p-valor = {p_value}\n'
    except Exception as e:
        return f'Erro ao realizar o teste de Wilcoxon entre {col1} e {col2}: {e}\n'
    

results = np.array([
    [0.95604178, 0.9610005,  0.96081281, 0.95403545, 0.95895166, 0.95689057,
     0.95734191, 0.95632765, 0.96470646, 0.95489019],  # KNeighborsRegressor_30
    [0.9578851,  0.95905307, 0.96414723, 0.95909205, 0.96491598, 0.95632027,
     0.95550496, 0.95515802, 0.96114323, 0.95478354],  # KNeighborsRegressor_31
    [0.95892365, 0.95676372, 0.95537772, 0.95784693, 0.96384383, 0.95482091,
     0.96124583, 0.96338676, 0.96083065, 0.95551705],  # KNeighborsRegressor_32
    [0.96041786, 0.9556615,  0.96544024, 0.95617343, 0.95654032, 0.95615465,
     0.95446967, 0.96464428, 0.9613493,  0.95881489],  # KNeighborsRegressor_33
    [0.95307931, 0.96224807, 0.96287509, 0.95939395, 0.96220557, 0.96092732,
     0.9524269,  0.95512244, 0.96266823, 0.95976692],  # KNeighborsRegressor_34
    [0.96077588, 0.95516967, 0.95807348, 0.95685596, 0.95396397, 0.96573763,
     0.95740689, 0.96152978, 0.96172362, 0.95423938],  # KNeighborsRegressor_35
    [0.95178137, 0.94739511, 0.9492909,  0.95380621, 0.94811421, 0.95041041,
     0.95054715, 0.95249252, 0.95010793, 0.95312014],  # KNeighborsRegressor_36
    [0.94218378, 0.95426396, 0.95497797, 0.95205936, 0.95511814, 0.95144127,
     0.94455972, 0.94635266, 0.94572786, 0.96153414],  # KNeighborsRegressor_37
    [0.95167343, 0.94783782, 0.95326656, 0.95547594, 0.95466109, 0.9523722,
     0.95072881, 0.94825489, 0.94390014, 0.95379532],  # KNeighborsRegressor_38
    [0.95198095, 0.95501114, 0.95285264, 0.94433274, 0.95117744, 0.9502844,
     0.94813864, 0.94717962, 0.95879571, 0.94806517]   # KNeighborsRegressor_39
])

statistic, p_value = friedmanchisquare(*results)

output_text = f"Estatística de Friedman: {statistic}\nValor-p: {p_value}\n"

with open("KnnRegressorTest_3.txt", "w") as file:
    file.write(output_text)

print("Resultados salvos em 'KnnRegressorTest_3.txt'")

if p_value < 0.05:
   
    output_text = ""
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            output_text += wilcoxon_test(results, i, j)

    # Salvar os resultados em um arquivo de texto
    with open("./wilcoxonTest/KnnRegressorTest_3.txt", "w") as file:
        file.write(output_text)

    print("Resultados salvos em './wilcoxonTest/KnnRegressorTest_3.txt'")
