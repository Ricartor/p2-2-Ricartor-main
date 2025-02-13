"""    def predict(self, X: np.ndarray) -> np.ndarray:
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        lista_min_distancia = []
        lista_idx_min_distancia = []
        for i in range(0,self.k):
            lista_min_distancia.append(np.inf)
            lista_idx_min_distancia.append(np.inf)
        for i in range(0,len(self.x_train)):
            distancia_x_al_vector = minkowski_distance(X,self.x_train[i],self.p)
            if distancia_x_al_vector > lista_min_distancia[-1]:
                lista_min_distancia[-1] = distancia_x_al_vector
                lista_idx_min_distancia[-1] = self.x_train[-1]
                ordenar_edades = list(zip(lista_idx_min_distancia, lista_min_distancia))
                ordenar_edades.sort(key=lambda x: x[1])  

                # Separar listas nuevamente
                lista_idx_min_distancia, lista_min_distancia = zip(*ordenar_edades)
        dict_valores = {}
        for i in range(0,self.k):
            idx = lista_idx_min_distancia[i]
            dict_valores.get(self.y_train[idx],1)
        print(max(dict_valores))"""
"""import numpy as np
nombres = np.array(["Ana", "Luis", "Carlos"])
edades = np.array([25, 30, 20])

indices_ordenados = np.argsort(edades)
print(indices_ordenados[-1:-4:-1])
print(indices_ordenados[0:2])
"""
"""datos = {2: 1, 3: 2, 6: 5}

# Encontrar la clave con el máximo valor
maximo_valor = 0
for clase,num_repeticiones in datos.items():
    if num_repeticiones > maximo_valor:
        maximo_valor = num_repeticiones
        clase_maxima = clase
print(clase_maxima)"""

import matplotlib.pyplot as plt

plt.hist([5,2,1,5,2,3,7],bins=2,range=(0,1))
plt.show()