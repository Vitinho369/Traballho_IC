from scipy.stats import friedmanchisquare
import numpy as np

results = np.array([
    [0.94559684, 0.93601559, 0.93385124, 0.94382852, 0.92109932, 0.95224774, 0.93977557, 0.9451045, 0.94204069, 0.95384817],
    [0.6652777, 0.67034448, 0.68092417, 0.91839706, 0.65927041, 0.75204182, -0.30215062, 0.85676481, 0.93097676, -0.29938248],
    [0.94168305, 0.93908034, 0.94948333, 0.93974182, 0.92500947, 0.93631986, 0.89083713, 0.91311573, 0.93723172, 0.93719864],
    [0.04357404, 0.75026579, 0.72665411, -0.00767341, 0.70999747, 0.73270981, 0.76991903, 0.86224877, 0.74694364, 0.04808538],
    [0.92168548, 0.9126632, 0.93911103, 0.91857789, 0.91583352, 0.9052742, 0.91537997, 0.9242578, 0.92898991, 0.90836935],
    [0.22777432, -0.25244068, -0.28171975, 0.05104098, -0.21838063, 0.31508225, -0.1954117, -0.29277473, -0.24996086, -0.22396182],
    [0.65656249, 0.87469739, 0.89906113, 0.70046144, 0.68211072, 0.91990491, 0.8654559, 0.8881019, 0.04234802, 0.84767418],
    [0.96925021, 0.96340092, 0.96821166, 0.9624002, 0.96404561, 0.96710285, 0.96916054, 0.96287118, 0.96768512, 0.96637055],
    [0.96426551, 0.96708878, 0.96675888, 0.96124588, 0.95867513, 0.96157596, 0.95615666, 0.96344059, 0.96360591, 0.97063911],
    [0.96843874, 0.96338437, 0.95807677, 0.96909257, 0.96534894, 0.96825384, 0.96945378, 0.96629208, 0.96470523, 0.96920522]
])

stat, p = friedmanchisquare(*results)

print(f'Estatística de Friedman: {stat}')
print(f'Valor-p: {p}')
output_text = f'Estatística de Friedman: {stat}\nValor-p: {p}'


with open("DecisionTree0_test.txt", "w") as file:
    file.write(output_text)