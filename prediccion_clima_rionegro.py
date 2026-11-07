import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class HMM:
    """
    Clase para representar, simular y predecir un Modelo Oculto de Markov (HMM)
    aplicado al clima de Rionegro, Antioquia, Colombia.
    """
    def __init__(self, states, observations, start_prob, trans_prob, emit_prob):
        """
        Inicializa los parámetros del modelo HMM.
        
        Parámetros:
        - states: Lista de estados ocultos ['Soleado', 'Nublado', 'Lluvioso']
        - observations: Lista de observaciones ['Humedad Baja', 'Humedad Media', 'Humedad Alta']
        - start_prob: Distribución de probabilidad inicial
        - trans_prob: Matriz de transición entre estados ocultos
        - emit_prob: Matriz de emisión de observaciones dado cada estado
        """
        self.states = states
        self.observations = observations
        self.n_states = len(states)
        self.n_obs = len(observations)
        
        self.start_prob = np.array(start_prob, dtype=float)
        self.trans_prob = np.array(trans_prob, dtype=float)
        self.emit_prob = np.array(emit_prob, dtype=float)
        
        self._validate_probabilities()

    def _validate_probabilities(self):
        assert np.isclose(self.start_prob.sum(), 1.0), "La distribución inicial debe sumar 1."
        assert np.allclose(self.trans_prob.sum(axis=1), 1.0), "Las filas de transición deben sumar 1."
        assert np.allclose(self.emit_prob.sum(axis=1), 1.0), "Las filas de emisión deben sumar 1."

    def simulate(self, n_days=30, obs_from_api=None, seed=42):
        """
        Simula o infiere una secuencia de `n_days` días.
        - Si se proporcionan observaciones reales de la API (`obs_from_api`), muestrea
          la secuencia de estados ocultos condicionada a las observaciones reales y
          a las probabilidades de transición/emisión del HMM (filtrado Bayesiano secuencial).
        - Si no se proporcionan observaciones externas, realiza una simulación generativa
          pura de estados y observaciones.
        """
        if seed is not None:
            np.random.seed(seed)
            
        state_indices = np.zeros(n_days, dtype=int)
        obs_indices = np.zeros(n_days, dtype=int)
        
        if obs_from_api is not None:
            obs_indices = np.array(obs_from_api[:n_days], dtype=int)
            
            # Día 1: Probabilidad posterior P(S_0 | O_0) ∝ P(S_0) * P(O_0 | S_0)
            post_0 = self.start_prob * self.emit_prob[:, obs_indices[0]]
            post_0 /= post_0.sum()
            state_indices[0] = np.random.choice(self.n_states, p=post_0)
            
            # Días 2..N: Probabilidad posterior P(S_t | S_{t-1}, O_t) ∝ P(S_t | S_{t-1}) * P(O_t | S_t)
            for t in range(1, n_days):
                prev_state = state_indices[t - 1]
                post_t = self.trans_prob[prev_state] * self.emit_prob[:, obs_indices[t]]
                post_t /= post_t.sum()
                state_indices[t] = np.random.choice(self.n_states, p=post_t)
        else:
            # Simulación generativa estándar
            state_indices[0] = np.random.choice(self.n_states, p=self.start_prob)
            obs_indices[0] = np.random.choice(self.n_obs, p=self.emit_prob[state_indices[0]])
            
            for t in range(1, n_days):
                prev_state = state_indices[t - 1]
                state_indices[t] = np.random.choice(self.n_states, p=self.trans_prob[prev_state])
                obs_indices[t] = np.random.choice(self.n_obs, p=self.emit_prob[state_indices[t]])
                
        return state_indices, obs_indices


# ==============================================================================
# 1. CONSUMO DE LA API OPEN-METEO (RIONEGRO, ANTIOQUIA, COLOMBIA)
# ==============================================================================
def obtener_datos_rionegro(n_days=30):
    """
    Consulta la API de Open-Meteo para Rionegro, Antioquia (lat=6.1551, lon=-75.3737)
    usando el endpoint de temperature_2m y deriva 30 períodos diarios de observación.
    """
    # URL base proporcionada para Rionegro, Antioquia
    url_base = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=6.1551&longitude=-75.3737&hourly=temperature_2m"
    )
    # Agregamos past_days=14 y forecast_days=16 para cubrir 30 días completos
    url_30_dias = f"{url_base}&past_days=14&forecast_days=16"
    
    print("Consultando API de Open-Meteo para Rionegro, Antioquia...")
    print(f"Endpoint: {url_base}")
    
    try:
        response = requests.get(url_30_dias, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df_hourly = pd.DataFrame({
            'time': pd.to_datetime(data['hourly']['time']),
            'temperature_2m': data['hourly']['temperature_2m']
        }).dropna()
        
        # Agrupamos por día para obtener la temperatura media diaria de los 30 días
        df_hourly['date'] = df_hourly['time'].dt.date
        df_daily = df_hourly.groupby('date', as_index=False)['temperature_2m'].mean()
        df_daily = df_daily.head(n_days)
        
        temperaturas = df_daily['temperature_2m'].to_numpy()
        fechas = df_daily['date'].astype(str).to_numpy()
        
        # Relación física inversa entre temperatura y humedad relativa en Rionegro:
        # - Temperaturas más altas -> Humedad Baja (0)
        # - Temperaturas medias    -> Humedad Media (1)
        # - Temperaturas más bajas -> Humedad Alta (2)
        p33, p66 = np.percentile(temperaturas, [33.3, 66.6])
        obs_api = []
        for temp in temperaturas:
            if temp >= p66:
                obs_api.append(0)  # Humedad Baja
            elif temp >= p33:
                obs_api.append(1)  # Humedad Media
            else:
                obs_api.append(2)  # Humedad Alta
                
        print(f"Datos cargados exitosamente: {len(temperaturas)} días registrados en Rionegro.\n")
        return np.array(obs_api, dtype=int), temperaturas, fechas

    except Exception as e:
        print(f"Aviso: No se pudo conectar a la API ({e}). Usando simulación local de respaldo.\n")
        return None, None, None


# ==============================================================================
# 2. CONFIGURACIÓN DEL MODELO HMM Y SIMULACIÓN DE 30 DÍAS
# ==============================================================================
estados = ['Soleado', 'Nublado', 'Lluvioso']
observaciones = ['Humedad Baja', 'Humedad Media', 'Humedad Alta']

# Distribución inicial: [Soleado: 0.6, Nublado: 0.3, Lluvioso: 0.1]
prob_inicial = [0.6, 0.3, 0.1]

# Matriz de transición:
# - Soleado  -> [Soleado: 0.7, Nublado: 0.2, Lluvioso: 0.1]
# - Nublado  -> [Soleado: 0.3, Nublado: 0.4, Lluvioso: 0.3]
# - Lluvioso -> [Soleado: 0.2, Nublado: 0.3, Lluvioso: 0.5]
matriz_transicion = [
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
]

# Matriz de emisión:
# - Soleado  -> [Baja: 0.7, Media: 0.2, Alta: 0.1]
# - Nublado  -> [Baja: 0.2, Media: 0.5, Alta: 0.3]
# - Lluvioso -> [Baja: 0.1, Media: 0.3, Alta: 0.6]
matriz_emision = [
    [0.7, 0.2, 0.1],
    [0.2, 0.5, 0.3],
    [0.1, 0.3, 0.6]
]

dias_simulacion = 30
obs_rionegro, temp_rionegro, fechas_rionegro = obtener_datos_rionegro(n_days=dias_simulacion)

# Instanciar el modelo HMM y simular los 30 días usando los datos de Rionegro
modelo_hmm = HMM(estados, observaciones, prob_inicial, matriz_transicion, matriz_emision)
secuencia_estados, secuencia_obs = modelo_hmm.simulate(
    n_days=dias_simulacion,
    obs_from_api=obs_rionegro,
    seed=42
)


# ==============================================================================
# 3. IMPRESIÓN DE LAS PRIMERAS 10 OBSERVACIONES
# ==============================================================================
print("=" * 65)
print("PRIMERAS 10 OBSERVACIONES - RIONEGRO, ANTIOQUIA")
print("=" * 65)
for dia in range(10):
    estado_str = estados[secuencia_estados[dia]]
    obs_str = observaciones[secuencia_obs[dia]]
    print(f"Día {dia + 1}: {estado_str} → {obs_str}")


# ==============================================================================
# 4. ESTADÍSTICAS DEL MODELO (30 DÍAS)
# ==============================================================================
print("\n" + "=" * 65)
print("ESTADÍSTICAS DE LA SIMULACIÓN EN RIONEGRO (30 DÍAS)")
print("=" * 65)

# 1. Número de días en cada estado climático
print("\n1. Número de días en cada estado climático:")
conteo_estados = np.bincount(secuencia_estados, minlength=len(estados))
for i, estado in enumerate(estados):
    porcentaje_e = (conteo_estados[i] / dias_simulacion) * 100
    print(f"   - {estado}: {conteo_estados[i]} días ({porcentaje_e:.1f}%)")

# 2. Porcentaje de cada tipo de humedad observada
print("\n2. Porcentaje de cada tipo de humedad observada:")
conteo_obs = np.bincount(secuencia_obs, minlength=len(observaciones))
for j, obs in enumerate(observaciones):
    porcentaje_o = (conteo_obs[j] / dias_simulacion) * 100
    print(f"   - {obs}: {porcentaje_o:.1f}% ({conteo_obs[j]} días)")

# 3. Matriz de confusión entre estados y observaciones
matriz_confusion = np.zeros((len(estados), len(observaciones)), dtype=int)
for s, o in zip(secuencia_estados, secuencia_obs):
    matriz_confusion[s, o] += 1

df_confusion = pd.DataFrame(
    matriz_confusion,
    index=[f"Estado: {s}" for s in estados],
    columns=[f"Obs: {o}" for o in observaciones]
)
print("\n3. Matriz de confusión entre Estados y Observaciones:")
print(df_confusion)

if temp_rionegro is not None:
    print(f"\nTemperatura media en Rionegro durante los 30 días: {temp_rionegro.mean():.2f} °C "
          f"(Mín: {temp_rionegro.min():.1f} °C, Máx: {temp_rionegro.max():.1f} °C)")


# ==============================================================================
# 5. VISUALIZACIÓN CON MATPLOTLIB (GRÁFICOS ALINEADOS TEMPORALMENTE)
# ==============================================================================
# Colores distintivos solicitados:
# Soleado (amarillo), Nublado (azul claro), Lluvioso (azul oscuro)
colores_estados = {
    0: '#FFD700',  # Amarillo (Soleado)
    1: '#87CEEB',  # Azul claro (Nublado)
    2: '#1E3F66'   # Azul oscuro (Lluvioso)
}

# Colores para las barras de humedad
colores_humedad = {
    0: '#F4A261',  # Humedad Baja
    1: '#2A9D8F',  # Humedad Media
    2: '#264653'   # Humedad Alta
}

dias = np.arange(1, dias_simulacion + 1)
alturas_estados = secuencia_estados + 1
alturas_obs = secuencia_obs + 1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(
    'Modelo Oculto de Markov (HMM) - Clima en Rionegro, Antioquia (30 Días - Open-Meteo API)',
    fontsize=14, fontweight='bold'
)

# --- Gráfico 1: Barras de la secuencia de Estados Ocultos ---
bar_colors_est = [colores_estados[s] for s in secuencia_estados]
ax1.bar(dias, alturas_estados, color=bar_colors_est, edgecolor='black', width=0.75)
ax1.set_yticks([1, 2, 3])
ax1.set_yticklabels(estados, fontsize=10, fontweight='bold')
ax1.set_ylabel('Estado Climático', fontsize=11)
ax1.set_title('Secuencia de Estados Ocultos Predichos/Simulados', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_ylim(0, 3.7)

leyenda_estados = [
    Patch(facecolor=colores_estados[i], edgecolor='black', label=estados[i])
    for i in range(len(estados))
]
ax1.legend(handles=leyenda_estados, loc='upper right', title='Estados Climáticos')

# --- Gráfico 2: Barras de las Observaciones de Humedad (+ Temperatura API) ---
bar_colors_obs = [colores_humedad[o] for o in secuencia_obs]
ax2.bar(dias, alturas_obs, color=bar_colors_obs, edgecolor='black', width=0.75)
ax2.set_yticks([1, 2, 3])
ax2.set_yticklabels(observaciones, fontsize=10, fontweight='bold')
ax2.set_ylabel('Nivel de Humedad', fontsize=11)
ax2.set_xlabel('Día (Rionegro, Antioquia)', fontsize=11)
ax2.set_title('Observaciones de Humedad Derivadas de Open-Meteo (temperature_2m)', fontsize=12)
ax2.set_xticks(dias)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(0, 3.7)

# Si tenemos la serie de temperatura de la API, la superponemos en un eje secundario
if temp_rionegro is not None:
    ax2_temp = ax2.twinx()
    ax2_temp.plot(dias, temp_rionegro, color='#D62828', marker='o', linewidth=1.8,
                   markersize=4, label='Temp. Media (°C)')
    ax2_temp.set_ylabel('Temperatura 2m (°C)', color='#D62828', fontsize=10, fontweight='bold')
    ax2_temp.tick_params(axis='y', labelcolor='#D62828')

leyenda_obs = [
    Patch(facecolor=colores_humedad[i], edgecolor='black', label=observaciones[i])
    for i in range(len(observaciones))
]
ax2.legend(handles=leyenda_obs, loc='upper left', title='Observaciones')

plt.tight_layout()
plt.show()