import numpy as np
from scipy.stats import friedmanchisquare

results = np.array([
    [0.94306637, 0.93300611, 0.92481903, 0.93796619, 0.92547061, 0.93193466,
     0.92112719, 0.93127405, 0.90818345, 0.92045191],  # MLPRegressor_0
    [0.92943421, 0.93903189, 0.92628476, 0.91885514, 0.93770442, 0.92631329,
     0.92637922, 0.92865007, 0.92663232, 0.92633587],  # MLPRegressor_1
    [0.93290733, 0.93371484, 0.91637725, 0.93826414, 0.92802554, 0.91872692,
     0.93470726, 0.93191743, 0.93525445, 0.92418228],  # MLPRegressor_2
    [0.93110835, 0.9355707,  0.92938674, 0.92688219, 0.93629388, 0.93672778,
     0.93372023, 0.92386508, 0.91537828, 0.93412953],  # MLPRegressor_3
    [0.93109236, 0.93546615, 0.93519393, 0.93552375, 0.93420263, 0.93620157,
     0.91231873, 0.9177656,  0.9334389,  0.92219115],  # MLPRegressor_4
    [0.93018792, 0.92599631, 0.91826482, 0.92670308, 0.92355453, 0.90402174,
     0.93563477, 0.93093536, 0.93366804, 0.92346916],  # MLPRegressor_5
    [0.92980871, 0.92485217, 0.9166611,  0.93416568, 0.93170477, 0.93331232,
     0.93178975, 0.93454087, 0.93331333, 0.90793077],  # MLPRegressor_6
    [0.92949451, 0.90904535, 0.92357679, 0.92239303, 0.89747664, 0.93624253,
     0.92714111, 0.93260394, 0.92033319, 0.91939742],  # MLPRegressor_7
    [0.91667806, 0.92527018, 0.93244158, 0.92269793, 0.93298457, 0.93681077,
     0.93540903, 0.92773184, 0.93641282, 0.92873629],  # MLPRegressor_8
    [0.93161115, 0.92128267, 0.91512389, 0.92293578, 0.91377472, 0.92593632,
     0.91886521, 0.90728035, 0.90790888, 0.90615941]   # MLPRegressor_9
])

statistic, p_value = friedmanchisquare(*results)

output_text = f"Estatística de Friedman: {statistic}\nValor-p: {p_value}\n"

with open("MLPRegressorTest_0.txt", "w") as file:
    file.write(output_text)

print("Resultados salvos em 'MLPRegressorTest_0_9_Results.txt'")
