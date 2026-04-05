import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_perfil_vendedor():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_perfil_vendedor.
    """

    # 1. Configuración aleatoria
    n_filas = random.randint(20, 60)

    vendedores_posibles = ['Ana', 'Carlos', 'Luis', 'María', 'Pedro', 'Sofía', 'Jorge', 'Laura']
    n_vendedores = random.randint(3, 5)
    vendedores = random.sample(vendedores_posibles, n_vendedores)

    vendedor_col = 'vendedor'
    monto_col = 'monto'
    fecha_col = 'fecha'

    # 2. Generar fechas aleatorias dentro de un año
    fechas = pd.date_range(start='2023-01-01', end='2023-12-31', periods=n_filas)
    fechas = pd.to_datetime(np.random.choice(fechas, size=n_filas, replace=True))

    # 3. Generar el DataFrame aleatorio
    df = pd.DataFrame({
        vendedor_col: [random.choice(vendedores) for _ in range(n_filas)],
        monto_col: np.round(np.random.uniform(100, 5000, size=n_filas), 2),
        fecha_col: fechas
    })

    # ---------------------------------------------------------
    # 4. Construir el INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'vendedor_col': vendedor_col,
        'monto_col': monto_col,
        'fecha_col': fecha_col
    }

    # ---------------------------------------------------------
    # 5. Calcular el OUTPUT esperado (Ground Truth)
    #    Replicamos la lógica que debería tener calcular_perfil_vendedor
    # ---------------------------------------------------------

    # A. Convertir fecha a datetime
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # B. Calcular mes con mayor facturación por vendedor
    df['mes'] = df[fecha_col].dt.month
    mes_mayor = (
        df.groupby([vendedor_col, 'mes'])[monto_col]
        .sum()
        .reset_index()
        .sort_values(monto_col, ascending=False)
        .groupby(vendedor_col)
        .first()['mes']
        .rename('mes_mayor_facturacion')
    )

    # C. Calcular métricas por vendedor
    resumen = df.groupby(vendedor_col, as_index=False).agg(
        total_ventas=(monto_col, 'sum'),
        promedio_venta=(monto_col, 'mean'),
        num_transacciones=(monto_col, 'count')
    )

    resumen['promedio_venta'] = resumen['promedio_venta'].round(2)
    resumen = resumen.rename(columns={vendedor_col: 'vendedor'})

    # D. Agregar mes con mayor facturación
    resumen = resumen.merge(mes_mayor, left_on='vendedor', right_index=True)

    # E. Ordenar y reiniciar índice
    resumen = resumen.sort_values('total_ventas', ascending=False).reset_index(drop=True)

    output_data = resumen

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    i, o = generar_caso_de_uso_calcular_perfil_vendedor()

    print('---- inputs ----')
    for k, v in i.items():
        print('\n', k, ':\n', v)

    print('\n---- expected output ----\n', o)
