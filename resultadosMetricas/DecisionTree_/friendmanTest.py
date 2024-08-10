from scipy.stats import friedmanchisquare
grupo1 = [0.9561366, 0.94946702, 0.95015403, 0.95655412 ,0.93977886, 0.95649091, 0.94746526, 0.94323236, 0.95368805, 0.94601549]

statistic, p_value = friedmanchisquare(grupo1, grupo2, grupo3)

print("Estatística de Friedman:", statistic)
print("Valor-p:", p_value)