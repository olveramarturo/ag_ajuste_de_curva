# Ajuste de Curva con Algoritmo Genético Clásico

Implementación de un **algoritmo genético clásico (AG)** en Python para resolver el problema de síntesis paramétrica de una curva. El AG busca los 7 coeficientes desconocidos de un cuasi-polinomio comparando la curva generada contra una curva objetivo.

Proyecto desarrollado como parte del curso de Algoritmos Evolutivos — Universidad Autónoma de Guadalajara / Arizona State University.

---

## Problema

Dada la estructura de un cuasi-polinomio:

```
f(x) = a·sin(x) + b·sin(2x) + c·sin(3x)
      + d·cos(x) + e·cos(2x)
      + f·x + g
```

El AG encuentra los parámetros `{a, b, c, d, e, f, g}` sin conocerlos de antemano, únicamente minimizando el error entre la curva que genera y la curva objetivo.

---

## Representación cromosómica

| Elemento | Detalle |
|---|---|
| Cromosoma | 7 genes de 8 bits (56 bits totales) |
| Rango por gen | 0 – 255 (uint8) |
| Decodificación | `parámetro = gen / 50` |
| Rango representable | 0.0 – 5.1 por parámetro |
| Resolución | 0.02 por parámetro |

---

## Operadores implementados

- **Selección por torneo** — 5 participantes por torneo (5 % de la población)
- **Cruzamiento a nivel de bit** — punto de corte aleatorio sobre los 56 bits; segmentación con compuertas AND y reunificación con OR
- **Mutación** — compuerta NOT sobre bits seleccionados aleatoriamente (tasa: 3 %)
- **Elitismo** — mezcla de padres e hijos (200 individuos), conserva los 100 mejores

---

## Resultados (200 generaciones, semilla 42)

| Parámetro | Real | Encontrado | Error |
|---|---|---|---|
| a | 3.0 | 3.08 | 2.7 % |
| b | 1.5 | 1.54 | 2.7 % |
| c | 1.0 | 1.02 | 2.0 % |
| d | 4.0 | **4.00** | 0.0 % |
| e | 2.0 | **2.00** | 0.0 % |
| f | 1.0 | 1.04 | 4.0 % |
| g | 2.5 | 2.38 | 4.8 % |
| **Error total** | — | 3.70 | — |
| **Reducción** | — | — | **99.0 %** |

---

## Instalación

```bash
pip install numpy matplotlib
```

Python 3.8 o superior.

---

## Uso

```bash
python ag_ajuste_curva.py
```

El script genera dos ventanas en tiempo real:

- **Panel izquierdo** — evolución del ajuste: curva objetivo (rojo) vs. curva del mejor individuo por generación (azul)
- **Panel derecho** — función de aptitud del mejor individuo en escala logarítmica

Al terminar imprime una tabla comparativa en consola y guarda `resultado_ag_ajuste_curva.png`.

### Opciones configurables

Al inicio de `ag_ajuste_curva.py`:

```python
USAR_MUTACION  = True    # activar / desactivar mutación
USAR_ELITISMO  = True    # activar / desactivar elitismo
N_POBLACION    = 100     # tamaño de la población
N_GENERACIONES = 200     # número de generaciones
TASA_MUTACION  = 0.03    # fracción de bits a mutar
SEMILLA        = 42      # None para resultados variables
```

---

## Archivos

```
.
├── propuesta_ejemplo.md    # Descripción técnica del ejemplo (Markdown)
├── propuesta_ejemplo.docx  # Descripción técnica del ejemplo (Word)
├── referencias.md          # Referencias académicas verificables
└── ag_ajuste_de_curva/     # Repositorio
```

---

## Referencias clave

- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
- Miller, B. L. & Goldberg, D. E. (1995). Genetic algorithms, tournament selection, and the effects of noise. *Complex Systems*, 9(3), 193–212. [PDF](https://wpmedia.wolfram.com/sites/13/2018/02/09-3-2.pdf)
- Karr, C. L., Stanley, D. A. & Scheiner, B. J. (1991). *Genetic Algorithm Applied to Least Squares Curve Fitting*. U.S. Bureau of Mines RI-9339. [PDF](https://stacks.cdc.gov/view/cdc/10511/cdc_10511_DS1.pdf)

Ver [`referencias.md`](referencias.md) para la lista completa con 15 referencias organizadas por concepto.
