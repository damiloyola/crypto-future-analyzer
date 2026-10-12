"""Ordena los estados del HMM y les pone nombre.

Se implementa en el Hito 3.

- Ordena los estados por retorno medio (desempate por volatilidad) y les asigna
  los nombres de ``regime.labels[n]`` en config.yaml, del más bajista al más
  alcista.
- ``regime_stats(model)``: retorno medio por vela, volatilidad media y
  duración esperada de cada estado, más la matriz de transición ya etiquetada.
"""
