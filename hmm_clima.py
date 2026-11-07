import os
import sys
import platform
import requests
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Garantizar salida UTF-8 en consolas de Windows y Notebooks
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configuración global de semilla aleatoria para reproducibilidad estricta
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Estilo visual de calidad publicable
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'figure.dpi': 150
})


# ==============================================================================
# 1. DOCUMENTACIÓN DEL ENTORNO Y REPRODUCIBILIDAD
# ==============================================================================
def documentar_entorno():
    """Imprime las versiones exactas del lenguaje, sistema y librerías."""
    try:
        import hmmlearn
        hmmlearn_ver = hmmlearn.__version__
    except ImportError:
        hmmlearn_ver = "No requerido (implementación nativa NumPy)"

    info = {
        "Python": sys.version.split()[0],
        "Sistema Operativo": f"{platform.system()} {platform.release()}",
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
        "Matplotlib": matplotlib.__version__,
        "Requests": requests.__version__,
        "hmmlearn": hmmlearn_ver,
        "Semilla Global (Seed)": str(GLOBAL_SEED)
    }
    print("=" * 75)
    print("1. REPRODUCIBILIDAD Y DOCUMENTACIÓN DEL ENTORNO")
    print("=" * 75)
    for k, v in info.items():
        print(f"  - {k:<24}: {v}")
    print("=" * 75 + "\n")
    return info


# ==============================================================================
# 2. CLASE DEL MODELO OCULTO DE MARKOV (HMM)
# ==============================================================================
class HMM:
    """
    Modelo Oculto de Markov (HMM) discreto con simulación Monte Carlo
    y decodificación óptima mediante el algoritmo de Viterbi.
    """
    def __init__(self, states, observations, start_prob, trans_prob, emit_prob, name="Base"):
        self.name = name
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

    def simulate(self, n_days=30, obs_from_api=None, rng=None):
        """
        Simula una trayectoria de estados ocultos y observaciones de longitud `n_days`.
        Usa un generador `np.random.Generator` explícito para garantizar reproducibilidad.
        """
        if rng is None:
            rng = np.random.default_rng(GLOBAL_SEED)

        state_indices = np.zeros(n_days, dtype=int)
        obs_indices = np.zeros(n_days, dtype=int)

        if obs_from_api is not None and len(obs_from_api) >= n_days:
            obs_indices = np.array(obs_from_api[:n_days], dtype=int)
            # Filtrado Bayesiano secuencial condicionado a las observaciones de la API
            post_0 = self.start_prob * self.emit_prob[:, obs_indices[0]]
            post_0 /= post_0.sum()
            state_indices[0] = rng.choice(self.n_states, p=post_0)

            for t in range(1, n_days):
                prev_state = state_indices[t - 1]
                post_t = self.trans_prob[prev_state] * self.emit_prob[:, obs_indices[t]]
                post_t /= post_t.sum()
                state_indices[t] = rng.choice(self.n_states, p=post_t)
        else:
            # Simulación generativa pura de cadenas de Markov ocultas
            state_indices[0] = rng.choice(self.n_states, p=self.start_prob)
            obs_indices[0] = rng.choice(self.n_obs, p=self.emit_prob[state_indices[0]])

            for t in range(1, n_days):
                prev_state = state_indices[t - 1]
                state_indices[t] = rng.choice(self.n_states, p=self.trans_prob[prev_state])
                obs_indices[t] = rng.choice(self.n_obs, p=self.emit_prob[state_indices[t]])

        return state_indices, obs_indices

    def decode_viterbi(self, obs_indices):
        """
        Algoritmo de Viterbi para inferir la secuencia más probable de estados ocultos
        a partir de las observaciones de humedad.
        """
        T = len(obs_indices)
        eps = 1e-12
        log_start = np.log(self.start_prob + eps)
        log_trans = np.log(self.trans_prob + eps)
        log_emit = np.log(self.emit_prob + eps)

        viterbi = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)

        viterbi[0] = log_start + log_emit[:, obs_indices[0]]
        for t in range(1, T):
            for s in range(self.n_states):
                scores = viterbi[t - 1] + log_trans[:, s]
                backpointer[t, s] = np.argmax(scores)
                viterbi[t, s] = np.max(scores) + log_emit[s, obs_indices[t]]

        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(viterbi[-1])
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path


# ==============================================================================
# 3. DATOS METEOROLÓGICOS DE RIONEGRO, ANTIOQUIA (API OPEN-METEO)
# ==============================================================================
def obtener_datos_rionegro(n_days=90):
    """
    Obtiene datos horarios de temperature_2m desde la API de Open-Meteo para
    Rionegro, Antioquia (lat=6.1551, lon=-75.3737). Si no hay conexión de red,
    genera una serie sintética calibrada con el perfil térmico real de Rionegro (17°C ± 2.2°C).
    """
    url_base = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=6.1551&longitude=-75.3737&hourly=temperature_2m"
    )
    url_extendida = f"{url_base}&past_days=74&forecast_days=16"

    try:
        response = requests.get(url_extendida, timeout=3)
        response.raise_for_status()
        data = response.json()
        df_hourly = pd.DataFrame({
            'time': pd.to_datetime(data['hourly']['time']),
            'temperature_2m': data['hourly']['temperature_2m']
        }).dropna()
        df_hourly['date'] = df_hourly['time'].dt.date
        df_daily = df_hourly.groupby('date', as_index=False)['temperature_2m'].mean().head(n_days)
        temperaturas = df_daily['temperature_2m'].to_numpy()
        fuente = "API Open-Meteo en vivo (Rionegro, Antioquia)"
    except Exception:
        # Respaldo determinista calibrado con la climatología de Rionegro (altitud 2,125 msnm)
        rng_temp = np.random.default_rng(GLOBAL_SEED)
        t_axis = np.arange(n_days)
        temperaturas = 17.2 + 1.4 * np.sin(2 * np.pi * t_axis / 15.0) + rng_temp.normal(0, 0.9, size=n_days)
        fuente = "Perfil climatológico calibrado de Rionegro (Modo Offline Reproducible)"

    p33, p66 = np.percentile(temperaturas, [33.3, 66.6])
    obs_api = np.where(temperaturas >= p66, 0, np.where(temperaturas >= p33, 1, 2))
    return obs_api, temperaturas, fuente


# ==============================================================================
# 4. DEFINICIÓN Y EVALUACIÓN DE LOS 3 ESCENARIOS EXPERIMENTALES
# ==============================================================================
estados = ['Soleado', 'Nublado', 'Lluvioso']
observaciones = ['Humedad Baja', 'Humedad Media', 'Humedad Alta']

# Escenario 1: Modelo Base (Especificación original, 30 días)
escenario_1 = {
    "id": "Escenario 1: Base (Clima Templado)",
    "short": "E1: Base (30d)",
    "n_days": 30,
    "n_runs": 500,
    "start_prob": [0.6, 0.3, 0.1],
    "trans_prob": [
        [0.7, 0.2, 0.1],  # Soleado
        [0.3, 0.4, 0.3],  # Nublado
        [0.2, 0.3, 0.5]   # Lluvioso
    ],
    "emit_prob": [
        [0.7, 0.2, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.3, 0.6]
    ]
}

# Escenario 2: Temporada de Lluvias / La Niña en Rionegro (30 días, matrices modificadas)
escenario_2 = {
    "id": "Escenario 2: Temporada Lluviosa (La Niña)",
    "short": "E2: Lluvioso (30d)",
    "n_days": 30,
    "n_runs": 500,
    "start_prob": [0.2, 0.3, 0.5],
    "trans_prob": [
        [0.40, 0.40, 0.20],  # Menor persistencia de días soleados
        [0.15, 0.45, 0.40],  # Mayor transición de nublado a lluvioso
        [0.10, 0.25, 0.65]   # Alta persistencia de lluvia
    ],
    "emit_prob": [
        [0.60, 0.30, 0.10],
        [0.15, 0.50, 0.35],
        [0.05, 0.25, 0.70]   # Mayor emisión de Humedad Alta en lluvia
    ]
}

# Escenario 3: Temporada Seca con Horizonte Extendido (90 días, mayor persistencia solar)
escenario_3 = {
    "id": "Escenario 3: Temporada Seca (Horizonte 90 días)",
    "short": "E3: Seco (90d)",
    "n_days": 90,
    "n_runs": 500,
    "start_prob": [0.75, 0.20, 0.05],
    "trans_prob": [
        [0.80, 0.15, 0.05],  # Alta persistencia de días soleados
        [0.45, 0.40, 0.15],  # Rápida recuperación hacia días soleados
        [0.35, 0.40, 0.25]   # Baja persistencia de lluvia
    ],
    "emit_prob": [
        [0.75, 0.20, 0.05],
        [0.25, 0.55, 0.20],
        [0.10, 0.35, 0.55]
    ]
}

escenarios = [escenario_1, escenario_2, escenario_3]


def calcular_estadisticos(valores):
    """
    Calcula Media, Desviación Estándar, Mínimo, Máximo e Intervalo de Confianza al 95% (IC 95%).
    """
    arr = np.asarray(valores, dtype=float)
    n = len(arr)
    media = np.mean(arr)
    std = np.std(arr, ddof=1)
    minimo = np.min(arr)
    maximo = np.max(arr)
    # Valor crítico z = 1.96 para IC del 95% en M=500 réplicas
    margen_error = 1.96 * (std / np.sqrt(n))
    ic_low = media - margen_error
    ic_high = media + margen_error
    return media, std, minimo, maximo, ic_low, ic_high


def evaluar_escenarios(lista_escenarios, seed=GLOBAL_SEED):
    """
    Ejecuta M simulaciones Monte Carlo independientes por cada escenario
    para obtener distribuciones robustas de las métricas principales.
    """
    registros_tabla = []
    resultados_detallados = {}

    for idx_esc, cfg in enumerate(lista_escenarios):
        modelo = HMM(
            estados, observaciones,
            cfg["start_prob"], cfg["trans_prob"], cfg["emit_prob"],
            name=cfg["id"]
        )
        rng = np.random.default_rng(seed + idx_esc * 100)
        n_days = cfg["n_days"]
        n_runs = cfg["n_runs"]

        metricas = {
            "% Días Soleado": [],
            "% Días Nublado": [],
            "% Días Lluvioso": [],
            "% Humedad Baja": [],
            "% Humedad Media": [],
            "% Humedad Alta": [],
            "Exactitud Viterbi (%)": []
        }

        matriz_conf_acum = np.zeros((3, 3), dtype=float)
        primera_trayectoria = None

        for r in range(n_runs):
            s_seq, o_seq = modelo.simulate(n_days=n_days, rng=rng)
            pred_seq = modelo.decode_viterbi(o_seq)

            if r == 0:
                primera_trayectoria = (s_seq, o_seq, pred_seq)

            # Porcentajes de estados
            c_est = np.bincount(s_seq, minlength=3) / n_days * 100.0
            metricas["% Días Soleado"].append(c_est[0])
            metricas["% Días Nublado"].append(c_est[1])
            metricas["% Días Lluvioso"].append(c_est[2])

            # Porcentajes de observaciones
            c_obs = np.bincount(o_seq, minlength=3) / n_days * 100.0
            metricas["% Humedad Baja"].append(c_obs[0])
            metricas["% Humedad Media"].append(c_obs[1])
            metricas["% Humedad Alta"].append(c_obs[2])

            # Exactitud de predicción de Viterbi
            acc = np.mean(s_seq == pred_seq) * 100.0
            metricas["Exactitud Viterbi (%)"].append(acc)

            # Acumular matriz de confusión
            for s, o in zip(s_seq, o_seq):
                matriz_conf_acum[s, o] += 1

        # Calcular estadísticos descriptivos e IC 95% para cada métrica
        for nombre_metrica, vals in metricas.items():
            media, std, vmin, vmax, ic_l, ic_h = calcular_estadisticos(vals)
            registros_tabla.append({
                "Escenario": cfg["short"],
                "Días (N)": n_days,
                "Réplicas (M)": n_runs,
                "Métrica": nombre_metrica,
                "Media": round(media, 2),
                "Desv. Est. (SD)": round(std, 2),
                "Mínimo": round(vmin, 2),
                "Máximo": round(vmax, 2),
                "IC 95% Inf": round(ic_l, 2),
                "IC 95% Sup": round(ic_h, 2),
                "IC 95% Formato": f"[{ic_l:.2f}, {ic_h:.2f}]"
            })

        resultados_detallados[cfg["short"]] = {
            "config": cfg,
            "metricas_raw": metricas,
            "matriz_confusion_media": matriz_conf_acum / n_runs,
            "primera_trayectoria": primera_trayectoria
        }

    df_resultados = pd.DataFrame(registros_tabla)
    return df_resultados, resultados_detallados


# ==============================================================================
# 5. EJECUCIÓN PRINCIPAL Y REPORTE DE RESULTADOS
# ==============================================================================
if __name__ == "__main__":
    # 1. Documentar entorno
    entorno_info = documentar_entorno()

    # 2. Consultar datos de Rionegro (Open-Meteo)
    obs_rionegro, temp_rionegro, fuente_datos = obtener_datos_rionegro(n_days=30)
    print(f"Fuente de datos meteorológicos: {fuente_datos}")
    print(f"Temperatura media Rionegro (30 días): {temp_rionegro.mean():.2f} °C "
          f"(SD: {temp_rionegro.std(ddof=1):.2f} °C, Mín: {temp_rionegro.min():.2f} °C, Máx: {temp_rionegro.max():.2f} °C)\n")

    # 3. Evaluar los 3 escenarios con 500 réplicas Monte Carlo cada uno
    df_stats, detalles = evaluar_escenarios(escenarios, seed=GLOBAL_SEED)

    # Mostrar las primeras 10 observaciones de la corrida principal del Escenario 1
    s_seq_e1, o_seq_e1, _ = detalles["E1: Base (30d)"]["primera_trayectoria"]
    print("=" * 75)
    print("2. PRIMERAS 10 OBSERVACIONES (ESCENARIO 1 - SEMILLA 42)")
    print("=" * 75)
    for d in range(10):
        print(f"Día {d + 1:2d}: {estados[s_seq_e1[d]]:<9} → {observaciones[o_seq_e1[d]]}")

    # Imprimir Tabla Comparativa Completa
    print("\n" + "=" * 75)
    print("3. TABLA COMPARATIVA DE LOS 3 ESCENARIOS (MEDIA, SD, MÍN, MÁX, IC 95%)")
    print("=" * 75)
    columnas_mostrar = [
        "Escenario", "Días (N)", "Métrica", "Media",
        "Desv. Est. (SD)", "Mínimo", "Máximo", "IC 95% Formato"
    ]
    print(df_stats[columnas_mostrar].to_string(index=False))

    # Guardar tabla en CSV para referencia
    df_stats.to_csv("resultados_escenarios_hmm.csv", index=False, encoding="utf-8-sig")

    # ==========================================================================
    # 6. GENERACIÓN DE 2 FIGURAS DE CALIDAD PUBLICABLE
    # ==========================================================================
    colores_estados = ['#FFD700', '#87CEEB', '#1E3F66']  # Amarillo, Azul claro, Azul oscuro
    colores_humedad = ['#F4A261', '#2A9D8F', '#264653']

    # --------------------------------------------------------------------------
    # FIGURA 1: Serie Temporal Alineada (Estados Ocultos vs. Observaciones y Temperatura en Rionegro)
    # --------------------------------------------------------------------------
    modelo_api = HMM(
        estados, observaciones,
        escenario_1["start_prob"], escenario_1["trans_prob"], escenario_1["emit_prob"]
    )
    rng_api = np.random.default_rng(GLOBAL_SEED)
    s_api, o_api = modelo_api.simulate(n_days=30, obs_from_api=obs_rionegro, rng=rng_api)
    dias_30 = np.arange(1, 31)

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    fig1.suptitle(
        "Figura 1: Dinámica Temporal de Estados Climáticos Ocultos y Humedad en Rionegro, Antioquia (N = 30 días)",
        y=0.98
    )

    # Panel A: Estados Ocultos
    ax1.bar(
        dias_30, s_api + 1,
        color=[colores_estados[s] for s in s_api],
        edgecolor='black', linewidth=0.7, width=0.75
    )
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(estados, fontweight='bold')
    ax1.set_ylabel("Estado Climático Oculto")
    ax1.set_title("A) Secuencia de Estados Climáticos Inferidos (Modelo HMM Base - Semilla = 42)")
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_ylim(0, 4.1)
    leyenda_est = [
        Patch(facecolor=colores_estados[i], edgecolor='black', label=estados[i])
        for i in range(3)
    ]
    ax1.legend(handles=leyenda_est, loc='upper right', ncol=3, title="Estado Climático", frameon=True)

    # Panel B: Observaciones de Humedad + Temperatura 2m
    ax2.bar(
        dias_30, o_api + 1,
        color=[colores_humedad[o] for o in o_api],
        edgecolor='black', linewidth=0.7, width=0.75
    )
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(observaciones, fontweight='bold')
    ax2.set_ylabel("Categoría de Humedad")
    ax2.set_xlabel("Tiempo (Día del Mes en Rionegro, Antioquia)")
    ax2.set_title("B) Observaciones de Humedad y Serie de Temperatura Media Diaria (Open-Meteo API)")
    ax2.set_xticks(dias_30)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_ylim(0, 4.2)

    ax2_t = ax2.twinx()
    ax2_t.plot(
        dias_30, temp_rionegro, color='#D62828', marker='o',
        linewidth=1.8, markersize=4.5, label='Temperatura 2m (°C)'
    )
    ax2_t.set_ylabel("Temperatura Media 2m (°C)", color='#D62828')
    ax2_t.tick_params(axis='y', labelcolor='#D62828')
    ax2_t.set_ylim(temp_rionegro.min() - 0.5, temp_rionegro.max() + 1.2)

    leyenda_obs = [
        Patch(facecolor=colores_humedad[i], edgecolor='black', label=observaciones[i])
        for i in range(3)
    ]
    ax2.legend(handles=leyenda_obs, loc='upper left', ncol=3, title="Humedad Observada", frameon=True)
    ax2_t.legend(loc='upper right', frameon=True)

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig("figura_1_serie_temporal_rionegro.png", dpi=200)

    # --------------------------------------------------------------------------
    # FIGURA 2: Comparación Multi-Escenario (Intervalos de Confianza 95%, Distribuciones y Exactitud)
    # --------------------------------------------------------------------------
    fig2, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15.5, 6.0))
    fig2.suptitle(
        "Figura 2: Análisis Comparativo de los 3 Escenarios Climáticos HMM (M = 500 Réplicas Monte Carlo)",
        y=0.99
    )

    nombres_cortos = [e["short"] for e in escenarios]
    x_pos = np.arange(len(nombres_cortos))
    ancho = 0.25

    # Subplot A: Distribución de Estados Ocultos con barras de error (Desviación Estándar / IC 95%)
    metricas_est = ["% Días Soleado", "% Días Nublado", "% Días Lluvioso"]
    for i, (m_est, col, nombre_est) in enumerate(zip(metricas_est, colores_estados, estados)):
        sub = df_stats[df_stats["Métrica"] == m_est]
        medias = sub["Media"].to_numpy()
        sds = sub["Desv. Est. (SD)"].to_numpy()
        bars = ax_a.bar(
            x_pos + (i - 1) * ancho, medias, width=ancho,
            color=col, edgecolor='black', yerr=sds, capsize=4,
            label=nombre_est, error_kw={'elinewidth': 1.2}
        )
        for bar, sd_val in zip(bars, sds):
            h = bar.get_height()
            ax_a.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h + sd_val + 1.2),
                ha='center', va='bottom', fontsize=7.5, fontweight='bold'
            )

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(nombres_cortos)
    ax_a.set_ylabel("Proporción de Días (%) ± 1 SD")
    ax_a.set_title("A) Distribución de Estados Ocultos")
    ax_a.set_ylim(0, 98)
    ax_a.grid(axis='y', linestyle='--', alpha=0.4)
    ax_a.legend(title="Estado Climático", loc="upper left", ncol=3, fontsize=8)

    # Subplot B: Distribución de Observaciones de Humedad
    metricas_obs = ["% Humedad Baja", "% Humedad Media", "% Humedad Alta"]
    for i, (m_obs, col, nombre_obs) in enumerate(zip(metricas_obs, colores_humedad, observaciones)):
        sub = df_stats[df_stats["Métrica"] == m_obs]
        medias = sub["Media"].to_numpy()
        sds = sub["Desv. Est. (SD)"].to_numpy()
        bars = ax_b.bar(
            x_pos + (i - 1) * ancho, medias, width=ancho,
            color=col, edgecolor='black', yerr=sds, capsize=4,
            label=nombre_obs, error_kw={'elinewidth': 1.2}
        )
        for bar, sd_val in zip(bars, sds):
            h = bar.get_height()
            ax_b.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h + sd_val + 1.2),
                ha='center', va='bottom', fontsize=7.5, fontweight='bold'
            )

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(nombres_cortos)
    ax_b.set_ylabel("Proporción de Observaciones (%) ± 1 SD")
    ax_b.set_title("B) Humedad Emitida por Escenario")
    ax_b.set_ylim(0, 98)
    ax_b.grid(axis='y', linestyle='--', alpha=0.4)
    ax_b.legend(title="Nivel de Humedad", loc="upper left", ncol=3, fontsize=8)

    # Subplot C: Boxplot de Exactitud de Predicción Viterbi (%) en los 3 Escenarios
    datos_viterbi = [
        detalles[sc]["metricas_raw"]["Exactitud Viterbi (%)"]
        for sc in nombres_cortos
    ]
    bp = ax_c.boxplot(
        datos_viterbi, tick_labels=nombres_cortos, patch_artist=True,
        showmeans=True,
        meanprops={'marker': 'D', 'markerfacecolor': '#D62828', 'markeredgecolor': 'black'}
    )
    colores_cajas = ['#FFD700', '#1E3F66', '#F4A261']
    for patch, color in zip(bp['boxes'], colores_cajas):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax_c.set_ylabel("Exactitud de Decodificación Viterbi (%)")
    ax_c.set_title("C) Exactitud de Predicción (Viterbi)")
    ax_c.grid(axis='y', linestyle='--', alpha=0.4)

    # Leyenda explicativa de la media en el boxplot
    ax_c.plot([], [], 'D', markerfacecolor='#D62828', markeredgecolor='black', label='Media Muestral')
    ax_c.legend(loc='lower right')

    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig("figura_2_comparacion_escenarios.png", dpi=200)
    print("\nFiguras guardadas exitosamente:")
    print("  - figura_1_serie_temporal_rionegro.png")
    print("  - figura_2_comparacion_escenarios.png")
    plt.show()
