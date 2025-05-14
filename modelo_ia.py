import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Dados simulados
data = {
    'intensidade_sinal': [2, 3, 1, 4, 3],
    'uso_cpu': [20, 35, 15, 40, 30],
    'uso_rede': [100, 300, 80, 500, 250],
    'consumo_bateria': [10, 20, 8, 25, 18]
}

df = pd.DataFrame(data)

X = df[['intensidade_sinal', 'uso_cpu', 'uso_rede']]
y = df['consumo_bateria']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Erro quadrático médio:", mean_squared_error(y_test, y_pred))

plt.plot(y_test.values, label='Real', marker='o')
plt.plot(y_pred, label='Previsto', marker='x')
plt.xlabel('Amostras')
plt.ylabel('Consumo de Bateria')
plt.title('Previsão de Consumo de Bateria')
plt.legend()
plt.grid(True)

# Garante que a pasta 'imagens' exista antes de salvar
import os
os.makedirs('imagens', exist_ok=True)

# Salva o gráfico com segurança
plt.savefig('imagens/grafico_previsao.png')
plt.show()

