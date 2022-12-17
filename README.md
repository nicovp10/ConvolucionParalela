# Paralelización de una convolución de una imagen en niveles de gris

## Introducción
Este estudio ha sido realizado para aprender de forma didáctica como de escalable es la paralelización de una convolución de una imagen en niveles de gris. Se ha analizado la escalabilidad en función del número de procesos involucrados y del tamaño de los bloques de columnas en los que se ha dividido la imagen para su paralelización.

El repositorio está formado por una [carpeta de imágenes](images), dónde se guardarán las imágenes resultantes de la ejecución así como se insertarán si se desea imágenes de entrada nuevas, el [código](parallel_conv.c) en C del programa, los ficheros asociados a la librería [stb](https://github.com/nothings/stb) para manejo de imágenes, un par de scripts en Bash para ayudar a la ejecución con MPI, un [script](plots.py) en Python para graficar los valores de análisis obtenidos y el [informe](Informe.pdf) asociado al estudio completo de la escalabilidad.


## Ejecución
> **NOTA 1:**  los scripts de Bash están adaptados a la ejecución del código en el supercomputador Finisterrae III del [Centro de Supercomputación de Galicia](https://www.cesga.es/), por lo que no se asegura su correcto funcionamiento en otros sistemas.

> **NOTA 2:** el programa está pensado para imágenes de entrada en formato JPG, por lo que no se asegura su correcto funcionamiento con imágenes en otros formatos.

Para ejecutar el programa se clonará o descargará este repositorio insertando todos los archivos en un mismo directorio. A continuación se ejecutará el script [`global_execution.sh`](global_execution.sh). Los resultados de la ejecución se podrán ver en:
* `slurm_files`: ficheros Slurm
* `output_files`: ficheros de salida con nombre `output_X.txt`, siendo _X_ el número de procesos involucrados en dicha ejecución.
* `images`: imágenes de salida con nombre `output_Y_X.jpg`, siendo _Y_ el número de columnas que tiene cada bloque y _X_ el número de procesos involucrados en dicha ejecución.

Si se desea utilizar otra imagen de entrada, se deberá añadir un nuevo parámetro de entrada a la llamada a ejecución del programa en la línea 27 del script [`single_execution.sh`](single_execution.sh), o bien modificar la línea 54 del [código fuente](parallel_conv.c), indicando la ruta de la imagen en ambos casos.

Si se desea graficar los resultados se ejecutará el script [`plots.py`](plots.py) una vez finalizada completamente la ejecución. Los gráficos resultantes estarán en la carpeta `plots`.
