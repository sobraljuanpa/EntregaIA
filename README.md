# Primer sesion

## Instalacion y prueba de ambiente.

Se arranca a probar observation y action space, leer mas sobre el ambiente.

Se arranca a elaborar una lista respecto a lo que deberia hacer el agente y cosas que vamos a necesitar.

## Primer approach - aleatorio absoluto

Para verificar que nuestro algoritmo de Q learning es bueno en lo que hace (eficiente/eficaz/efectivo), primero quisimos establecer un benchmark de cuales serian las reward obtenidas por una policy completamente aleatoria. Para esto ejecutamos 10, 100 y 1000 iteraciones, calculando la recompensa promedio en cada uno de los casos, de forma de ver si estabamos llegando a algun valor esperado para la policy aleatoria.

Los valores obtenidos en la experimentacion fueron convergiendo a 22, por lo que podemos decir que nuestro benchmark es 22.
Es decir, si luego de aplicar q learning, la performance de nuestro agente es en promedio peor que 22, no tiene sentido lo que hicimos.

## Segundo approach - entreno usando un q en funcion unicamente de la posicion

Para un primer approach con q learning pero limitandonos a un unico factor, vamos a tomar en cuenta unicamente un factor de los obtenidos en funcion del cual determinar las acciones. 

Hay 3 factores a tener en cuenta diseñando un sistema de Q-Learning:

1. Estados
2. Acciones
3. Recompensas

En este caso, las Acciones y Recompensas vienen pre establecidas, por lo cual vamos a limitar la información sobre el estado a simplemente la posicion del carro.
Además, al ser un valor continuo entre -4.8 y 4.8, lo que vamos a hacer para acotar el espacio de posiciones, es tomar una cifra luego de la coma, y por lo tanto vamos a estar trabajando con un espacio discreto de valores de 96 posibles.

para cada valor la idea es hacer que si valor < 0 entonces
    q_values[(valor*-10)-1] => del 0 al 47
si valor >=0
    q_values[valor+48] => del 48 al 96


El informe debe incluir:
- Resumen de cómo abordó cada tarea. Incluyendo información relevante. (Ej: Interacción con
el simulador, parámetros utilizados, tiempo de ejecución y resultados obtenidos).
- Apoyo visual (gráficos) y comentarios que permita entender el desempeño de sus soluciones.
- Cualquier nota de advertencia que desee comunicarle a la empresa antes del lanzamiento.

# Ejercicio 1

## Intro problema
## Abordaje

## Interaccion con el simulador

La interacción con el simulador fue llevada a cabo dentro de un ambiente gym del tipo cartpole v1 (version limita cantidad max de iteraciones).

Cada paso nos daba como resultado una percepción con 4 datos, los cuales en un principio no utilizamos, y terminamos utilizando para mejorar la performance de nuestro agente.

## Params utilizados


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

Tambien se hizo esto con la velocidad angular luego de realizar varias simulaciones y mirar los valores obtenidos.

## Tiempo de ejecucion

Se realizaron varias iteraciones de entrenamiento y simulacion con el agente entrenado, en general, las de entrenamiento fueron de entre 1 y 2 hs, con entre 1 y 10 millones de episodios (distinto hardware).

A la hora de probar el agente entrenado, variaba bastante según la calidad del aprendizaje, es decir, cuando aprendía mejor, los episodios eran más largos y por lo tanto duraban más las ejecuciones, pero las pruebas que realizamos con 10 mil iteraciones para promediar el valor obtenido, no pasaban de los cinco minutos.

## Resultados obtenidos

Primero que nada utilizamos como benchmark el valor esperado promediando las recompensas obtenidas en 10 mil iteraciones, con el algoritmo aleatorio. Obtuvimos que el valor esperado de este algoritmo es de 22, por lo cual teniamos que obtener valores iniciales por arriba de esto para estar tranquilos de que estaba aprendiendo el agente.

Primero probamos solo con posicion, logrando valores de 15, lo cual era completamente inaceptable.
Luego con posicion y velocidad, obteniendo valores por arriba de 22 con consistencia. 
Al utilizar todos los valores obtenidos por las percepciones, arrancamos a ver valores por arriba de 150 con consistencia, indicando que ibamos por el camino correcto y solo restaba afinar buckets y entrenamiento.

Finalmente, al arrancar a entrenar con más de 2 millones de episodios, arrancamos a ver resultados ampliamente satisfactorios, obteniendo varias veces el valor máximo posible y promediando en el entorno de 350.

## Performance y Resultados

PROBAR CON MAS BUCKETS PARA POSICION/VELOCIDAD (4 CADA UNO)
PROBAR ACHICNDO EL LINSPACE DE VELOCIDAD ANGULAR (-1.5, 1.5)