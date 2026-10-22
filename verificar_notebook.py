"""Ejecuta todas las celdas en un kernel limpio y conserva salidas reales."""
from pathlib import Path
import json
import os
import sys
import nbformat
from nbclient import NotebookClient

raiz = Path(__file__).resolve().parent
# Kernel local: no modifica la configuración global del usuario.
kernel = raiz / '.jupyter' / 'kernels' / 'python3'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Python 3', 'language': 'python'}), encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(raiz / '.jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(raiz / '.jupyter' / 'runtime')
os.environ['MPLCONFIGDIR'] = str(raiz / '.jupyter' / 'matplotlib')
nb = nbformat.read(raiz / 'rutas_inundacion.ipynb', as_version=4)
NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(raiz)}}).execute()
nbformat.validate(nb)
nbformat.write(nb, raiz / 'rutas_inundacion.ipynb')
for celda in nb.cells:
    for salida in celda.get('outputs', []):
        if salida.output_type == 'stream':
            print(salida.text)
        if 'text/markdown' in salida.get('data', {}):
            print(salida.data['text/markdown'])
        if 'image/png' in salida.get('data', {}):
            import base64
            (raiz / 'comparacion.png').write_bytes(base64.b64decode(salida.data['image/png']))
print('Notebook completo ejecutado y esquema validado.')
