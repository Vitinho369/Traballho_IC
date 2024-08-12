import numpy as np
from scipy.stats import friedmanchisquare

results = np.array([
    [-1.70311951e-05,  8.83915380e-01,  8.73408438e-01, -8.02860138e-05,
      8.75891152e-01,  8.95862777e-01, -9.07545734e-05, -4.82174684e-07,
     -8.32761219e-06,  8.76506930e-01],  # MLPRegressor_20
    [ 8.55445976e-01,  8.88224018e-01, -3.73576157e-04,  8.61400634e-01,
     -1.27927183e-04, -7.26258456e-06, -2.96994625e-04,  8.67755749e-01,
      8.74556158e-01, -4.62849874e-04],  # MLPRegressor_21
    [-3.44673308e-04,  8.89309819e-01, -2.77219962e-06, -6.38374311e-05,
     -1.78344357e-04, -1.31962492e-04, -6.52046917e-04, -6.53715489e-04,
     -8.57637386e-04, -1.23838180e-04],  # MLPRegressor_22
    [-1.66927597e-03,  9.02306604e-01, -9.76779594e-05, -2.27861527e-04,
     -5.15117885e-04, -1.45257944e-04, -4.67311767e-06,  8.32694819e-01,
     -8.91745479e-04, -1.08068883e-05],  # MLPRegressor_23
    [0.90817237, 0.90900855, 0.86351865, 0.88547807, 0.92156763, 
     0.90809336, 0.91090827, 0.86922365, 0.92493478, 0.87834018],  # MLPRegressor_24
    [-1.69370554e-04, -1.60127146e-03,  8.77562224e-01, -3.37177566e-04,
      8.78572467e-01,  8.83149492e-01,  8.86581280e-01,  8.67721287e-01,
      8.38218648e-01, -3.22835996e-04],  # MLPRegressor_25
    [-1.51712338e-05, -5.19575968e-06, -5.39405760e-05, -2.72999961e-04,
     -2.09455754e-04, -5.21384842e-04, -2.68785533e-04, -5.64854309e-05,
     -5.09986652e-05, -4.10230205e-05],  # MLPRegressor_26
    [0.59824678, 0.86724245, 0.70913377, 0.88083697, 0.87668993, 
     0.86772921, 0.8617944, 0.83591786, 0.8732016, 0.84161136],  # MLPRegressor_27
    [-3.02046777e-04, -2.03310826e-04, -5.16324129e-04, -7.26490748e-04,
     -4.42757543e-05, -5.26730714e-07,  7.16466283e-01, -3.21159505e-06,
     -8.23851913e-05, -2.01521810e-04],  # MLPRegressor_28
    [-2.17042043e-05, -5.31396530e-04, -1.53069484e-04, -7.20134183e-05,
     -1.05816578e-04, -6.58433795e-06, -1.06992753e-04, -1.39456991e-05,
     -4.21541223e-05, -3.66112280e-04]   # MLPRegressor_29
])

statistic, p_value = friedmanchisquare(*results)

output_text = f"Estatística de Friedman: {statistic}\nValor-p: {p_value}\n"

with open("MLPRegressorTest_2.txt", "w") as file:
    file.write(output_text)

print("Resultados salvos em 'MLPRegressorTest_2.txt'")
