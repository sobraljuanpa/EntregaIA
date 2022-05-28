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

