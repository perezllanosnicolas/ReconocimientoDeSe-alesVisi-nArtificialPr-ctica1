# ReconocimientoDeSe-alesVisi-nArtificialPr-ctica1

/practica1_vision/
│
├── .gitignore               # Ignorar la carpeta data/ y otros archivos no relevantes.
├── README.md                
├── main.py                  # Script principal.
├── evaluar_resultados.py    # Script proporcionado por los profesores.
│
├── src/                     # Carpeta con código fuente 
│   ├── __init__.py
│   ├── utils.py             # Funciones auxiliares (IoU, pintar cajas, leer/escribir txt).
│   ├── detector_base.py     # Clase padre o interfaz.
│   ├── detector_mser.py     # Clase con la detección obligatoria (MSER + Color).
│   └── detector_alt.py      # Clase con la detección alternativa.
│
├── data/                    # (NO SUBIR A GITHUB - Añadir a .gitignore)
│   ├── train/               # Imágenes de entrenamiento y gt.txt[cite: 153].
│   └── test/                # Imágenes de test y gt.txt[cite: 154].
│
└── memoria/                 # Archivos de la documentación
    └── memoria_practica1.pdf # El PDF final a entregar[cite: 171].