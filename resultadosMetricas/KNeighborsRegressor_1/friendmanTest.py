import numpy as np
from scipy.stats import friedmanchisquare

# Resultados do R2 para cada configuração do KNeighborsRegressor
results = np.array([
    [0.94201328, 0.95418097, 0.95428658, 0.95457952, 0.95761916, 0.95023606,
     0.9454062,  0.94749599, 0.94095573, 0.96168994],  # KNeighborsRegressor_10
    [0.9533667,  0.94928961, 0.95514719, 0.95386999, 0.95199489, 0.95365798,
     0.94902226, 0.94645706, 0.94799266, 0.95349094],  # KNeighborsRegressor_11
    [0.9476199,  0.95348362, 0.95251274, 0.94452837, 0.95063016, 0.94822855,
     0.9504407,  0.95013832, 0.95692318, 0.9473132],   # KNeighborsRegressor_12
    [0.94851235, 0.95302976, 0.95897296, 0.95087839, 0.95576205, 0.94916142,
     0.94741786, 0.94748094, 0.95465311, 0.94829717],  # KNeighborsRegressor_13
    [0.95323657, 0.94985617, 0.94625758, 0.95061038, 0.95301278, 0.94588332,
     0.95198223, 0.95516383, 0.95510533, 0.94776354],  # KNeighborsRegressor_14
    [0.95312557, 0.94917887, 0.95606401, 0.94485389, 0.94745842, 0.94631284,
     0.94978377, 0.95960573, 0.95199072, 0.95280255],  # KNeighborsRegressor_15
    [0.94381452, 0.95684313, 0.95774356, 0.95168315, 0.95207115, 0.95272373,
     0.9441798,  0.94746738, 0.95562484, 0.95225541],  # KNeighborsRegressor_16
    [0.95070695, 0.9498387,  0.94896042, 0.94828626, 0.94447517, 0.95956709,
     0.94734056, 0.95346037, 0.95648828, 0.94723371],  # KNeighborsRegressor_17
    [0.9511849,  0.94547991, 0.95327388, 0.95083497, 0.9494922,  0.95150189,
     0.95473309, 0.95307515, 0.95659943, 0.95600838],  # KNeighborsRegressor_18
    [0.94164376, 0.95154325, 0.951949,   0.9588674,  0.95494336, 0.95429067,
     0.94437711, 0.94581927, 0.94727189, 0.95508962]   # KNeighborsRegressor_19
])

# Realizar o teste de Friedman
statistic, p_value = friedmanchisquare(*results)
output_text = f"Estatística de Friedman: {statistic}\nValor-p: {p_value}"

# Salvar os resultados em um arquivo de texto
with open("KnnRegressorTest_1.txt", "w") as file:
    file.write(output_text)
