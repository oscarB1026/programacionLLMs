import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

def generar_caso_de_uso_intervalo_confianza_bootstrap():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función intervalo_confianza_bootstrap.
    """

    # 1. Configuración aleatoria
    n_train = random.randint(50, 150)
    n_test = random.randint(10, 30)
    n_features = random.randint(2, 4)
    n_bootstrap = random.choice([50, 100, 150])
    confianza = random.choice([0.90, 0.95, 0.99])

    # 2. Generar datos de entrenamiento con relación lineal
    X_train = np.random.uniform(0, 10, size=(n_train, n_features))
    coeficientes = np.random.uniform(1, 5, size=n_features)
    ruido = np.random.normal(0, 1, size=n_train)
    y_train = X_train @ coeficientes + ruido

    # 3. Generar datos de prueba
    X_test = np.random.uniform(0, 10, size=(n_test, n_features))

    # ---------------------------------------------------------
    # 4. Construir el INPUT
    # ---------------------------------------------------------
    input_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'n_bootstrap': n_bootstrap,
        'confianza': confianza
    }

    # ---------------------------------------------------------
    # 5. Calcular el OUTPUT esperado (Ground Truth)
    #    Replicamos la lógica que debería tener
    #    intervalo_confianza_bootstrap
    # ---------------------------------------------------------

    # A. Realizar bootstrap
    predicciones = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Muestra con reemplazo
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_muestra = X_train[indices]
        y_muestra = y_train[indices]

        # Entrenar modelo y predecir
        modelo = LinearRegression()
        modelo.fit(X_muestra, y_muestra)
        predicciones.append(modelo.predict(X_test))

    # B. Calcular intervalos de confianza
    predicciones = np.array(predicciones)
    percentil_inferior = ((1 - confianza) / 2) * 100
    percentil_superior = (1 - (1 - confianza) / 2) * 100

    limite_inferior = np.percentile(predicciones, percentil_inferior, axis=0)
    limite_superior = np.percentile(predicciones, percentil_superior, axis=0)

    # C. Construir DataFrame de salida
    output_data = pd.DataFrame({
        'limite_inferior': np.round(limite_inferior, 4),
        'limite_superior': np.round(limite_superior, 4)
    })

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    i, o = generar_caso_de_uso_intervalo_confianza_bootstrap()

    print('---- inputs ----')
    for k, v in i.items():
        print('\n', k, ':\n', v)

    print('\n---- expected output ----\n', o)
