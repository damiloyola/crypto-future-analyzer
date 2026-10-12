"""Features para el detector de régimen (HMM).

Se implementa en el Hito 3. Todas causales: el valor en la vela *t* solo usa
datos hasta *t*.

- retorno logarítmico;
- volatilidad realizada (desvío de retornos en ventana de 24 velas);
- rango de la vela (high - low) / close;
- volumen relativo como z-score rodante.

La estandarización se ajusta solo con la ventana de entrenamiento y se aplica
igual hacia adelante.
"""
