# Gerar a saída
output_text = f"Estatística de Friedman: {statistic}\nValor-p: {p_value}\n"

# Salvar os resultados em um arquivo de texto
with open("MLPRegressorTest_3.txt", "w") as file:
    file.write(output_text)