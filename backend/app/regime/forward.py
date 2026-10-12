"""Forward filtering: régimen de cada vela usando solo datos hasta esa vela.

Se implementa en el Hito 3.

- ``filtered_probabilities(model, X) -> ndarray (T, K)``: algoritmo forward
  normalizado en espacio log. La fila *t* depende solo de ``X[:t+1]``.
- ``regime_series(...)``: estado más probable por vela y su probabilidad
  (la "confianza", 0-100 %).

Prohibido: ``model.predict``, ``model.decode`` y ``predict_proba`` sobre la
serie entera (usan el algoritmo forward-backward, que mira el futuro). Hay un
test que verifica que el régimen de la vela *t* no cambia al agregar velas
posteriores.
"""
