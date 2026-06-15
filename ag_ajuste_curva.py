"""
Algoritmo Genético Clásico — Ajuste de Curva
=============================================
Cuasi-polinomio objetivo (tipo Fourier + término lineal):

    f(x) = a·sin(x) + b·sin(2x) + c·sin(3x)
          + d·cos(x) + e·cos(2x)
          + f·x + g

Parámetros reales: a=3.0, b=1.5, c=1.0, d=4.0, e=2.0, f=1.0, g=2.5
Dominio: x ∈ [0, 2π], 200 puntos

Representación: 7 genes de 8 bits (uint8, valores 0–255)
Decodificación: parámetro_k = gen_k / PESO  (PESO = 50)

Operadores implementados:
  • Selección por torneo (5 participantes, 5 % de 100)
  • Cruzamiento de un punto a nivel de bit (máscaras AND / OR)
  • Mutación: compuerta NOT sobre bits aleatorios (tasa 3 %)
  • Elitismo: mezcla padres + hijos, conserva los 100 mejores

"""

import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════
USAR_MUTACION  = True    # True / False
USAR_ELITISMO  = True    # True / False

N_POBLACION    = 100
N_GENERACIONES = 200
TORNEO_K       = 5       # participantes por torneo (5 % de 100)
TASA_MUTACION  = 0.03    # fracción de bits a mutar por generación

SEMILLA        = 42      # None para resultados variables entre ejecuciones

# ═══════════════════════════════════════════════════════════════════
#  CUASI-POLINOMIO Y PARÁMETROS OBJETIVO
# ═══════════════════════════════════════════════════════════════════
#   f(x) = a·sin(x) + b·sin(2x) + c·sin(3x)
#         + d·cos(x) + e·cos(2x)
#         + f·x + g
#
# Parámetro de mayor valor: d = 4.0
# PESO = 50  →  rango representable: 0–255/50 = 5.1  (tolerancia ≈ 27 %)
# Resolución: 1/50 = 0.02 por parámetro
#
# Genes exactos correspondientes a los parámetros reales:
#   a=150, b=75, c=50, d=200, e=100, f=50, g=125
# ═══════════════════════════════════════════════════════════════════
PARAMS_REALES = np.array([3.0, 1.5, 1.0, 4.0, 2.0, 1.0, 2.5])
NOMBRES       = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
PESO          = 50.0

N_GENES   = 7
N_BITS    = 8
BITS_TOT  = N_GENES * N_BITS   # 56 bits por cromosoma

X_EVAL    = np.linspace(0, 2 * np.pi, 200)


def cuasi_polinomio(x, params):
    """f(x) = a·sin(x) + b·sin(2x) + c·sin(3x) + d·cos(x) + e·cos(2x) + f·x + g"""
    a, b, c, d, e, f, g = params
    return (a * np.sin(x)
            + b * np.sin(2 * x)
            + c * np.sin(3 * x)
            + d * np.cos(x)
            + e * np.cos(2 * x)
            + f * x
            + g)


# Curva de referencia (objetivo)
Y_OBJETIVO = cuasi_polinomio(X_EVAL, PARAMS_REALES)


# ═══════════════════════════════════════════════════════════════════
#  CODIFICACIÓN / DECODIFICACIÓN
# ═══════════════════════════════════════════════════════════════════
def decodificar(cromosoma):
    """Convierte genes uint8 a valores reales dividiendo por PESO."""
    return cromosoma.astype(float) / PESO


# ═══════════════════════════════════════════════════════════════════
#  FUNCIÓN DE APTITUD (minimizar)
# ═══════════════════════════════════════════════════════════════════
def aptitud(cromosoma):
    """Sumatoria del error absoluto |y_objetivo − y_generada| en cada punto."""
    params = decodificar(cromosoma)
    y_gen  = cuasi_polinomio(X_EVAL, params)
    return np.sum(np.abs(Y_OBJETIVO - y_gen))


def evaluar_poblacion(pop):
    return np.array([aptitud(ind) for ind in pop])


# ═══════════════════════════════════════════════════════════════════
#  SELECCIÓN POR TORNEO
# ═══════════════════════════════════════════════════════════════════
def torneo(pop, aptitudes):
    """Elige TORNEO_K individuos al azar y retorna al más apto (menor error)."""
    idx     = np.random.choice(len(pop), TORNEO_K, replace=False)
    ganador = idx[np.argmin(aptitudes[idx])]
    return pop[ganador].copy()


# ═══════════════════════════════════════════════════════════════════
#  CRUZAMIENTO A NIVEL DE BIT — single-point con máscaras AND / OR
# ═══════════════════════════════════════════════════════════════════
def cruzamiento(p1, p2):
    """
    1. Elegir corte aleatorio en [1, 55].
    2. Máscara baja  = 1 en bits 0..corte-1   (compuerta AND → parte baja)
       Máscara alta  = 1 en bits corte..55     (compuerta AND → parte alta)
    3. Hijo 1 = parte_baja(p1) OR parte_alta(p2)
       Hijo 2 = parte_baja(p2) OR parte_alta(p1)
    """
    corte = np.random.randint(1, BITS_TOT)       # punto de corte: 1..55

    bits1 = np.unpackbits(p1)                    # expansión a 56 bits (MSB primero)
    bits2 = np.unpackbits(p2)

    mascara_baja        = np.zeros(BITS_TOT, dtype=np.uint8)
    mascara_baja[:corte] = 1
    mascara_alta        = 1 - mascara_baja

    hijo1_bits = (bits1 & mascara_baja) | (bits2 & mascara_alta)
    hijo2_bits = (bits2 & mascara_baja) | (bits1 & mascara_alta)

    return np.packbits(hijo1_bits), np.packbits(hijo2_bits)


# ═══════════════════════════════════════════════════════════════════
#  MUTACIÓN — compuerta NOT (XOR con máscara de 1 bit)
# ═══════════════════════════════════════════════════════════════════
def mutar(pop):
    """
    Niega aleatoriamente bits de la población.
    n_mut = TASA_MUTACION × N_POBLACION × BITS_TOT bits por generación.
    Se aplica antes del cálculo de aptitud de los hijos.
    """
    n_mut = max(1, int(TASA_MUTACION * len(pop) * BITS_TOT))
    for _ in range(n_mut):
        ind  = np.random.randint(len(pop))
        gen  = np.random.randint(N_GENES)
        bit  = np.random.randint(N_BITS)
        pop[ind, gen] ^= np.uint8(1 << bit)     # NOT sobre ese bit
    return pop


# ═══════════════════════════════════════════════════════════════════
#  ALGORITMO GENÉTICO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════
def ejecutar_ag():
    if SEMILLA is not None:
        np.random.seed(SEMILLA)

    # Población inicial: matriz 100 × 7, genes uint8 en [0, 255]
    poblacion = np.random.randint(0, 256, (N_POBLACION, N_GENES), dtype=np.uint8)

    historial = []          # aptitud del mejor individuo por generación

    # ── Ventana 1: ajuste de curva ────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1.canvas.manager.set_window_title('Ventana 1 — Ajuste de curva')

    # ── Ventana 2: función de aptitud ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.canvas.manager.set_window_title('Ventana 2 — Función de aptitud')

    plt.ion()

    # ── Bucle generacional ────────────────────────────────────────
    for gen in range(N_GENERACIONES):

        # a) Evaluación
        aptitudes = evaluar_poblacion(poblacion)
        idx_mejor = np.argmin(aptitudes)
        historial.append(aptitudes[idx_mejor])
        mejor_curva = cuasi_polinomio(X_EVAL, decodificar(poblacion[idx_mejor]))

        # b) Reproducción: 50 torneos padre + 50 torneos madre → 100 hijos
        hijos = []
        for _ in range(N_POBLACION // 2):
            padre = torneo(poblacion, aptitudes)
            madre = torneo(poblacion, aptitudes)
            h1, h2 = cruzamiento(padre, madre)
            hijos.extend([h1, h2])
        hijos = np.array(hijos, dtype=np.uint8)

        # c) Mutación
        if USAR_MUTACION:
            hijos = mutar(hijos)

        # d) Elitismo o reemplazo generacional completo
        if USAR_ELITISMO:
            combinados      = np.vstack([poblacion, hijos])
            apt_combinada   = evaluar_poblacion(combinados)
            mejores_indices = np.argsort(apt_combinada)[:N_POBLACION]
            poblacion       = combinados[mejores_indices]
        else:
            poblacion = hijos

        # e) Visualización en tiempo real
        if gen % 5 == 0 or gen == N_GENERACIONES - 1:
            error_actual = historial[-1]

            # — Ventana 1: curva generada vs curva objetivo —
            ax1.clear()
            ax1.plot(X_EVAL, Y_OBJETIVO, 'r-',  lw=2.0, label='Curva objetivo')
            ax1.plot(X_EVAL, mejor_curva, 'b--', lw=1.5,
                     label=f'Mejor individuo — Gen {gen + 1}')
            ax1.set_title('Ajuste de curva — individuo más apto por generación')
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3)
            fig1.tight_layout()
            fig1.canvas.draw()

            # — Ventana 2: función de aptitud + etiqueta del error —
            ax2.clear()
            ax2.plot(range(1, len(historial) + 1), historial, 'g-', lw=1.5)
            ax2.set_title('Función de aptitud — individuo más apto')
            ax2.set_xlabel('Generación')
            ax2.set_ylabel('|Error absoluto| acumulado')
            ax2.grid(True, alpha=0.3)

            # Etiqueta de error publicada sobre la gráfica
            ax2.text(
                0.98, 0.95,
                f'Generación: {gen + 1}\nError: {error_actual:.4f}',
                transform=ax2.transAxes,
                fontsize=11, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9),
            )

            fig2.tight_layout()
            fig2.canvas.draw()

            plt.pause(0.001)

    plt.ioff()
    fig1.savefig('resultado_ventana1_curva.png',   dpi=150, bbox_inches='tight')
    fig2.savefig('resultado_ventana2_aptitud.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Reporte final ─────────────────────────────────────────────
    apt_final   = evaluar_poblacion(poblacion)
    mejor_final = poblacion[np.argmin(apt_final)]
    params_enc  = decodificar(mejor_final)

    print("\n" + "═" * 54)
    print("  RESULTADO FINAL DEL ALGORITMO GENÉTICO")
    print("═" * 54)
    print(f"{'Param':>6}  {'Real':>8}  {'Encontrado':>10}  {'Error %':>8}")
    print("─" * 54)
    for nombre, real, enc in zip(NOMBRES, PARAMS_REALES, params_enc):
        err = abs(real - enc) / abs(real) * 100 if real != 0 else 0.0
        print(f"  {nombre:>4}  {real:>8.4f}  {enc:>10.4f}  {err:>7.2f} %")
    print("─" * 54)
    print(f"  Error total (generación 1):   {historial[0]:>10.4f}")
    print(f"  Error total (generación {N_GENERACIONES}): {historial[-1]:>10.4f}")
    reduccion = (1 - historial[-1] / historial[0]) * 100
    print(f"  Reducción del error:          {reduccion:>9.1f} %")
    print("═" * 54)
    print(f"\n  Gráficas guardadas en:")
    print(f"    resultado_ventana1_curva.png")
    print(f"    resultado_ventana2_aptitud.png")


# ═══════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ejecutar_ag()