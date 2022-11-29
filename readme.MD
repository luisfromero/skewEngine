# SkewEngine

Un algoritmo para el cálculo intensivo de datos bidimensionales con una fuerte dependencia radial

## Introducción

Existen problemas en la literatura en los que es necesario realizar un cálculo extremadamente intensivo sobre cada uno de los puntos de una malla bidimensional de datos, como por ejemplo, sobre cada uno de los píxeles de una imagen, o sobre cualquier punto de un mapa. 

En algunos casos, la intensidad aritmética es tan alta, que hace que el problema sea intratable, desde el punto de vista computacional.

Veamos el ejemplo de la cuenca visual de un punto, en un modelo digital de elevaciones. Imagine que desea conocer qué parte de un territorio es visible desde un determinado lugar del territorio:

<p style=text-align:center;>
<img src="https://www.supergeotek.com/SpatialAnalyst_ENG_HTML/visible_invisible.jpg" width=300>
</p>

Como es obvio, para saber si desde A se puede ver B (donde B puede ser cualquier punto), habría que tener en cuenta la altura del resto de puntos, por ejemplo, C, que pueden ser un obstáculo para la visión. 

En un modelo digital (**DEM**, de *digital elevation model*), con N = dimx·dimy datos, la complejidad del problema, usando la notación big-O, sería O(N<sup>2</sup>), aunque es evidente que dicha complejidad se puede reducir si sólo se consideran como obstáculos los puntos C que están en la línea A-B. Aún así, el problema, de complejidad O(N<sup>1.5</sup>) es bastante lento. La aplicación Google Earth, por ejemplo, tarda varios segundos en obtener un resultado aproximado.


<p style=text-align:center;>
<img src="https://community.esri.com/t5/image/serverpage/image-id/49358i6FAE82C134BE3093/image-size/large?v=v2&px=999" width=300>
</p>

Pues bien, imagine ahora que desea averiguar el punto de un territorio que tiene "las mejores vistas posibles". En tal caso, tendría que calcular la cuenca visual desde todos los puntos de un territorio, y estaría abordando el problema conocido como ***total viewshed***, de complejidad O(N<sup>2.5</sup>).

Este sería un ejemplo de problema computacionalmente intratable (salvo para mapas con muy pocos datos y baja resolución) si no fuera porque tienen una fuerte *dependencia radial*, que hace que la información tiene mucho más valor cuanto más próxima se sitúa respecto al punto de interés (**pov**, de *point of view*). En consecuencia, para puntos alejados, no necesitamos usar toda la información, sino sólo una parte, o ninguna.

## Algoritmo de Romero, *et al*

En 2012, Romero *et al* [publican](https://www.ac.uma.es/~siham/visibility-2012.pdf) un algoritmo que considera dicha dependencia radial calculando la cuenca visual, para todos los puntos, en un conjunto discreto de direcciones (360, normalmente). 

<pre>
for pov in 1,N
    for sector in 1,360
        viewshed[pov] = function(sector,DEM)
</pre>


Además, aprovecha que todos los datos estén alineados en una dirección para reaprovechar la información en las memorias caches. Su aportación consiste en invertir el orden de los lazos:

<pre>
for sector in 1,360
    for pov in 1,N
        viewshed[pov] = function(sector,DEM)
</pre>

En particular, dado una dirección, por ejemplo, oeste-suroeste (22º), se hace un barrido sobre todo el modelo, y se calcula la cuenca visual, sólo en esa dirección, acumulando finalmente los resultados del resto de direcciones.

<p style=text-align:center;>
<img src="https://onlinelibrary.wiley.com/cms/asset/00f4f7f7-d7c7-4275-89e0-fbbea69de97f/tgis12216-fig-0004-m.png" width=300>
</p>

En este [video](https://www.youtube.com/watch?v=Ohs8ioyYpX09) se muestra una animación del algoritmo.


## Algoritmo SDEM

Recientemente, Romero et al publican una importante modificación del algoritmo en el que se consideran dos aspectos claves para mejorar aún más el rendimiento:

* Si, dado un sector, sólo vamos a utilizar el resto de puntos que están en la misma línea ¿por qué no colocarlos directamente en la misma línea de memoria?

* Si, dado un sector, sólo vamos a utilizar el resto de puntos que están en la misma línea (no hay dependencia con otras líneas), ¿por qué no trabajar en paralelo con todas las líneas simultáneamente, usando además, GPUs?

Nace así el algoritmo skew-DEM (modelo sesgado del terreno), que se basa en la idea de colocar los datos en memoria de una forma apropiada para el cálculo en una determinada dirección, ya que el coste de "deconstruir y reconstruir el mapa" merece la pena, teniendo en cuenta la intensidad de los cálculos que se van a realizar sobre los datos "sesgados".

<p style=text-align:center;>
<img src="img/skew1.png" width=600>
</p>
