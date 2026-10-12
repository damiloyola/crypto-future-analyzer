"""Detector de régimen de mercado con un Gaussian HMM por moneda.

No predice el precio: clasifica el entorno (crash, bajista, neutral, alcista,
euforia) para que el resto del sistema se adapte.

Regla de oro: cero look-ahead. El régimen de cada vela sale del algoritmo
forward (probabilidades filtradas), nunca de ``predict()`` ni de Viterbi sobre
la serie completa.
"""
