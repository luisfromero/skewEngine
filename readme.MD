# SkewEngine

Un algoritmo para el cálculo intensivo de datos bidimensionales con una fuerte dependencia radial

## Introducción

Existen problemas en la literatura en los que es necesario realizar un cálculo extremadamente intensivo sobre cada uno de los puntos de una malla bidimensional de datos, como por ejemplo, sobre cada uno de los píxeles de una imagen. 

En algunos casos, la intensidad aritmética es tan alta, que hace que el problema sea intratable, desde el punto de vista computacional.

Veamos el ejemplo de la cuenca visual de un punto, en un modelo digital de elevaciones. Imagine que desea conocer qué parte de un territorio es visible desde un determinado lugar del territorio:

<p style=text-align:center;>
<img src="https://www.supergeotek.com/SpatialAnalyst_ENG_HTML/visible_invisible.jpg" width=300>
</p>

Como es obvio, para saber si desde A se puede ver B (donde B puede ser cualquier punto), habría que tener en cuenta la altura del resto de puntos, por ejemplo, C, que pueden ser un obstáculo para la visión. 

En un modelo digital, con N = dimx·dimy puntos, la complejidad del problema, usando la notación big-O, sería O(N<sup>2</sup>)

