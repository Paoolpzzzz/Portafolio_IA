# Rutas hacia un refugio durante una inundación

Prototipo académico en Python: BFS, Dijkstra y A* sobre mapas sintéticos de 10 × 10. La entrega principal es [rutas_inundacion.ipynb](rutas_inundacion.ipynb), con explicaciones, código completo por celdas, pruebas, tablas, gráficos y preguntas de comprensión.

## Instalación y ejecución

Recomendado: Python 3.12. En PowerShell, desde este repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab rutas_inundacion.ipynb
```

Selecciona el kernel del entorno `.venv`, reinícialo y ejecuta todas las celdas en orden. Las salidas guardadas corresponden a la ejecución de entrega; al repetir, se sustituyen con mediciones nuevas. No se necesita Internet después de instalar las dependencias.

Para comprobarlo sin interfaz:

```powershell
.\.venv\Scripts\python.exe verificar_notebook.py
```

El verificador crea una configuración de kernel local en `.jupyter/`, ejecuta todas las celdas con `nbclient`, valida el esquema del notebook y guarda sus salidas. Exporta el gráfico a `comparacion.png`. El script `construir_notebook.py` conserva la fuente organizada por celdas; **regenerar el notebook borra las salidas**, por lo que después debe ejecutarse el verificador.

## Estructura

```text
rutas_inundacion.ipynb     Entrega principal ejecutada
README.md                 Instrucciones y registro de verificación
requirements.txt          Dependencias con rangos compatibles
.gitignore                Entornos y archivos temporales excluidos
construir_notebook.py      Fuente editable del notebook
verificar_notebook.py      Ejecución reproducible en kernel limpio
comparacion.png           Exportación de la figura de comparación
```

## Modelo y resultados

Solo niveles 0, 1 y 2 permiten el paso; nivel 3 y obstáculos 4 bloquean. Costo de entrada = `1 + alfa × riesgo`, sin riesgo inicial. Los algoritmos no alteran el mapa ni conservan estado entre llamadas. Alfa debe ser finito y no negativo. La heurística de A* es Manhattan, multiplicada por el costo mínimo posible (1).

En el escenario base con alfa = 1, BFS obtiene 9 movimientos, riesgo 16 y costo 25. Dijkstra y A* obtienen 15 movimientos, riesgo 0 y costo 15. Las tablas del notebook incluyen los tiempos y nodos medidos de cada algoritmo; no se afirma una ventaja temporal universal.

## Verificación y uso de IA

Codex generó el código, explicaciones y documentación con asistencia de IA. La entrega fue ejecutada mediante `nbclient` en un kernel Python limpio; las seis pruebas obligatorias, los casos inicio = refugio y alfa cero, el análisis de sensibilidad y las pruebas negativas del validador usan aserciones. También se valida el formato del notebook y se inspecciona su figura. No se inventaron tiempos ni se creó un historial de commits.

La igualdad entre Dijkstra y A* no es una verificación independiente completa porque comparten una rutina. Se complementa con rutas de costo conocido y un validador que recalcula movimientos y costos sin llamar a las funciones de búsqueda. No se ha contrastado el modelo con inundaciones reales ni el equipo humano ha confirmado todavía su propia revisión.

Las dependencias se declaran por rangos, no como un entorno binariamente idéntico: pequeños cambios de versión pueden cambiar tiempos y presentación. La ejecución de entrega usa Python 3.12, NumPy 2.5.4 y Matplotlib 3.11.2.

## Etapas y commits sugeridos para el equipo

Estos mensajes son propuestas para cuando el equipo complete y revise cada etapa, no un historial existente:

1. `docs: definir modelo de terreno, riesgo y alcance académico`
2. `feat: implementar BFS, Dijkstra y A estrella`
3. `test: validar rutas y seis escenarios reproducibles`
4. `feat: comparar métricas y visualizar rutas antes y después`
5. `docs: registrar ejecución, limitaciones y uso de IA`

Cada integrante debería explicar la convención del riesgo inicial, el significado de alfa, la prueba de Manhattan y el error de estado residual antes de presentar el proyecto. Este es un modelo didáctico; no una herramienta operativa de evacuación.
