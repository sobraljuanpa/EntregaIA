# INTRODUCCION

Obligatorio de IA, por Federico Carbonell (224359) y Juan Pablo Sobral (192247)

# Ejercicio 1

## Intro problema

El problema planteado al equipo fue realizar un sistema de estabilizacion de una nave espacial completamente automatica y no tripulada con destino al planeta Marte. Para la resolución de dicho problema se decidio utilizar la tecnica Q-Learning. Está tecnica se basa en el aprendizaje reforzado basado en el cambio de estados y retroalimentacion.
Hay 3 factores a tener en cuenta diseñando :

1. Estados
2. Acciones
3. Recompensas

## Abordaje

Pueden ver el entrenamiento realizado en el archivo ```training.py```, si se quiere probar alguno de los modelos respaldados, hacerlo usando el ```simulateFromQModel.py```, quedan adjuntados notebooks relevantes en la carpeta html y cuando se haga referencia a algun .html se lo va a encontrar ahi. Tambien dejamos disponible una carpeta notebooks en la cual encontrar notebooks usados en el desarrollo de las tareas.

### Interaccion con el simulador

La interacción con el simulador fue llevada a cabo dentro de un ambiente gym del tipo cartpole v1 (version limita cantidad max de iteraciones).

Cada paso nos daba como resultado una percepción con 4 datos, los cuales en un principio no utilizamos en su totalidad, y terminamos utilizando para mejorar la performance de nuestro agente.

Bitacora de primeras interacciones:

``` python
# Se agrega una segunda dimension, si se entrena con 1000 episodios son pocos para la cantidad de posibles estados (pasa de 12 a 120)
# y muestra performance promedio peor que la iteracion previa solamente con 12 estados.
# Se arranca a entrenar con 10000 episodios y tomar muestreos promedio del valor de la policy con 10000 episodios tambien.
# Se arrancan a ver valores arriba de 22 con constancia (con 10 bins, rango de -5 a 5), pero se ven muchos bins vacios o semi vacios

# Se sugiere probar cambiar el rango a valores mas cercanos a los observados (-2.5 a 2.5) y aumentra la cant de episodios.
# Con un espacio de 10 para la posicion, 5 para las velocidades como descrito rpeviamente, con 10k episodios de entrenamiento
# el valor esperado es de aprox 150
```
### Params utilizados


El ambiente nos proveía 4 parámetros por observación:

| Parámetro                  | Min           | Max  |
| :------------------------: |:------------: | :---:|
| Posición del carro         | -4.8 | 4.8 |
| Velocidad del carro        | -inf      |   inf |
| Ángulo del palo            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
| Velocidad angular del palo | -inf | inf |

Como se puede ver, todos los valores obtenidos pertenecen a espacios continuos, por lo cual tuvimos que discretizar las percepciones para poder llevar a un modelo de q learning. Para llevar a cabo esto, utilizamos dos funciones de la librería ```numpy``` :  ```linspace``` y ```digitize``` para mapear los espacios continuos a un espacio discreto de bins/buckets y colocar las percepciones obtenidas dentro de los mismos.

A la hora de armar los buckets/bins, tuvimos en cuenta varias cosas que observamos y leimos de la documentación. 

La primera es que los valores de posicion del carro y ángulo del palo tenían un rango mayor al que se iban a encontrar la mayoría de las lecturas. Por ejemplo, en el caso de la posición, los valores posibles eran entre -4.8 y 4.8, pero si el carro salía del rango (-2.4,2.4) el juego se terminaba, por lo cual a la hora de armar los buckets los hicimos con los rangos en los que iban a estar la mayoria de valores.

Cuando arrancamos a ejecutar pruebas y ver los Q resultantes de estos buckets, nos dimos cuenta que muchos de los mismos quedaban vacíos, es decir, nunca se actualizaba su valor, lo cual es una perdida para nosotros dado que no mejoraba el modelo.

Fue por eso que tomamos la decisión de realizar las pruebas que se van a encontrar en el archivo ```graphs.ipynb```, para determinar de manera correcta como utilizar los buckets para obtener una distribucion al menos similar a normal. No buscamos una distribucion uniforme porque claramente los episodios tendian a distribuirse de otra forma, y el valor que iban a agregar en esos valores borde estaria mejor utilizado aprovechando a actualizar mas los valores mas utilizados.

Fue por eso que finalmente acotamos los espacios lineales de la siguiente manera:
| Parámetro                  | Min           | Max  |
| :------------------------: |:------------: | :---:|
| Posición del carro         | -0.2 | 0.2 |
| Velocidad del carro        | -1 | 1 |
| Ángulo del palo            | -0.25 | 0.25 |
| Velocidad angular del palo | -1.5 | 1.5 |

Haciendo esto y trabajando la cantidad de buckets utilizados para cada param, logramos obtener una matriz Q mucho mas densa y mejor calculada que nuestros primeros intentos.

### Tiempo de ejecucion

Se realizaron varias iteraciones de entrenamiento y simulacion con el agente entrenado, en general, las de entrenamiento fueron de entre 1 y 2 hs, con entre 1 y 10 millones de episodios (distinto hardware).

A la hora de probar el agente entrenado, variaba bastante según la calidad del aprendizaje, es decir, cuando aprendía mejor, los episodios eran más largos y por lo tanto duraban más las ejecuciones, pero las pruebas que realizamos con 10 mil iteraciones para promediar el valor obtenido, no pasaban de los cinco minutos.

Para minimizar y agilizar las pruebas se realizo un mecanismo para persistir los resultados de los entrenamientos anteriores.

### Resultados obtenidos

Primero que nada utilizamos como benchmark el valor esperado promediando las recompensas obtenidas en 10 mil iteraciones, con el algoritmo aleatorio. Obtuvimos que el valor esperado de este algoritmo es de 22, por lo cual teniamos que obtener valores iniciales por arriba de esto para estar tranquilos de que estaba aprendiendo el agente.

Primero probamos solo con posicion, logrando valores de 15, lo cual era completamente inaceptable.
Luego con posicion y velocidad, obteniendo valores por arriba de 22 con consistencia. 
Al utilizar todos los valores obtenidos por las percepciones, arrancamos a ver valores por arriba de 150 con consistencia, indicando que ibamos por el camino correcto y solo restaba afinar buckets y entrenamiento.

Finalmente, al arrancar a entrenar con más de 2 millones de episodios, arrancamos a ver resultados ampliamente satisfactorios, obteniendo varias veces el valor máximo posible y promediando en el entorno de 350.

La evidencia de estas ejecuciones, se encuentra en los archivos third_approachJP y 

## Performance y Resultados

### Variacion de buckets

Al variar la cantidad de buckets, principalmente vemos dos factores.

El primero es que a mayor cantidad de buckets, más episodios de entrenamiento son necesarios para obtener una matriz de valores Q util.

El segundo es que cuantos mas buckets usamos para un valor, es altamente probable que los de las puntas vayan quedando sin ser utilizados nunca, por lo cual no aportan y tienen un costo.

Al final decidimos utilizar dos variaciones: 2,2,12,10 y 2,2,6,6.

### Variacion de cantidad de episodios de entrenamiento

Como mencionabamos en el punto anterior, la cantidad de episodios de entrenamientos requeridos para generar una matriz relativamente util va de la mano de la cantidad de casilleros de la matriz, y la densidad de la distribucion de valores en la misma.

Una de nuestras preubas intermedias, con 50 buckets, entrenada con 10mil episodios daba como resultado un valor esperado de aprox 150. Asumiendo una relacion lineal, para los buckets que terminamos utilizando esto deberia

En nuestra experiencia, con menos de 2 millones de iteraciones con estas matrices de aprox 500 casilleros, no se obtienen valores interesantes luego al utilizar la matriz como parte de una policy.

### Resultados finales

Logramos las mejores ejecuciones luego de una sesion de entrenamientos con 2 millones de episodios, con la distribucion de buckets 2,2,12,10 y los rangos:

| Parámetro                  | Min           | Max  |
| :------------------------: |:------------: | :---:|
| Posición del carro         | -inf | inf |
| Velocidad del carro        | -inf | inf |
| Ángulo del palo            | -0.2095 | 0.2095 |
| Velocidad angular del palo | -2.5 | 2.5 |

La cual luego de 10 mil ejecuciones, dio de minimo de recompensas acumuladas 250, maximo 500 y promedio 483.3
Esta ejecucion esta registrada en el notebook ```third_approachJP.ipynb```
# Ejercicio 2 

## Intro problema

El problema planteado al equipo fue realizar un sistema que jugase solo al 2048 y ganase con una frecuencia razonable. Para resolver este problema se podia utilizar MiniMax o ExpectiMax.

Usamos minimax o expectimax? creo que utilizamos minimax al final.

Explicar nuestra implementacion de pruning.

## Interaccion con el simulador

Las interacciones con el simulador fueron bastante menos interesantes en este caso, nos apoyamos mucho en las funciones auxiliares para obtener los casilleros libres y la cantidad de movimientos, a la hora de establecer los posibles en nuestro minimax, y a la hora de calcular una de las heuristicas utilizadas para obtener el valor de el tablero.

## Funcion de evaluacion
### Pruebas iniciales
Cantidad espacios vacios
Cantidad merges posibles


### Version final
