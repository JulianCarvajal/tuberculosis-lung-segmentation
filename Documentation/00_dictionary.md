---
title: "Guía de términos CAD Tuberculosis"
authors: "Santiago Correa Marulanda, Sara Galván Ortega"
tutors: "Maria Bernarda Salazar, Luis Felipe Buitrago Castro"
bibliography: references.bib
csl: ieee.csl
---

***

# Diccionario


Como parte de la documentación del CAD se incluye este diccionario que explica el significado de las características (features) extraídas de las imágenes a analizar. 

Como se puede ver en el README dichas imágenes son un conjunto de radiografías de tórax.

## Extracción de características

> Es el proceso mediante el cual se transforma información compleja, como una imagen, en un conjunto de variables numéricas que contienen propiedades consideradas relevantes para el problema.

La extracción de características es una fase crucial en la clasificación de enfermedades respiratorias, ya que permite identificar patrones y rasgos asociados a afecciones específicas. Consiste en obtener información de alto nivel a partir de imágenes, como el color, la forma y la textura, asegurando que solo se consideren las áreas relevantes para el análisis [@tonni2025framework].

Acontinuación se explican las caracteríticas de manera agrupada 

 ### Gray-Level Co-Occurrence Matrix (GLCM) features

 El GLCM es un modelo estadístico que considera las conexiones espaciales entre píxeles, capturando información textural significativa sobre la distribución espacial de las intensidades de los píxeles y permitiendo la caracterización de los patrones de los tejidos. [@tonni2025framework;@prince2025interpretable]

 Las enfermedades respiratorias presentan patrones de textura distintivos y complejos, y el uso de descriptores de características de textura como GLCM ayuda a capturar la complejidad y ambigüedad inherentes a estos patrones.

 Los descriptores de características derivadas de este modelo, son las siguientes:

 #### Contraste (Contrast)

 Es una característica de textura que cuantifica cuánto difieren entre sí las intensidades entre los píxeles de referencia y los pixeles vecinos;  un contraste significativo indica variaciones significativas dentro de la matriz de coocurrencia de niveles de gris (GLCM), mientras que valores más altos indican anomalías en el tejido pulmonar. 

 #### Disimilitud (Dissimilarity)

 Es una medida de textura que cuantifica la diferencia absoluta entre los niveles de gris de pares de píxeles. A diferencia del contraste, utiliza la diferencia absoluta en lugar de elevarla al cuadrado.

 #### Homogeneidad (Homogeneity)

 1. Medida que indica qué tan concentrados están los elementos de una matriz de coocurrencia alrededor de su diagonal. Una mayor homogeneidad significa que los niveles de gris de los píxeles relacionados tienden a ser similares. [@prince2025interpretable]

 2. La homogeneidad en la GLCM mide la distribución de los elementos; una mayor homogeneidad resulta en un menor contraste. Los pulmones sanos muestran valores de homogeneidad y energía más altos, lo que indica uniformidad de la textura. [@tonni2025framework]
 
 #### Energía (Energy)

 La energía cuantifica la uniformidad local de los niveles de gris; valores más altos indican una mayor similitud entre píxeles.

 #### Correlación (Correlation)

 La característica de correlación indica la relación lineal entre los valores de los niveles de gris dentro de la matriz de coocurrencia.

### Caracteríticas basadas en la forma - *Shape-based features*

 Las características basadas en la forma son cruciales para identificar enfermedades respiratorias a través de imágenes de rayos X, ya que capturan características geométricas esenciales para la clasificación de enfermedades [@ayaz2021ensemble].

 El estudio [@tonni2025framework] identifica seis características basadas en la forma: *perímetro*, *área*, *circularidad*, *extensión*, *solidez* y *área rectangular*, que son cruciales para identificar lesiones pulmonares irregulares y comprender las propiedades de forma distintivas de las enfermedades respiratorias

 #### Perímetro
 El perímertro de una lesión pulmonar indica su extensión; un perímetro aumentado indica una infección.

 #### Área
 La métrica del área mide el tamaño de una región pulmonar, a partir del número de pixeles dentro de un controno específico.

 #### Circularidad
 La circularidad mide la redondez del área lesionada, indicando anomalías en el pulmón.

 #### Extensión

 La tecnología de extensión captura eficazmente la distribución espacial de las anomalías pulmonares al resaltar bordes, curvas y extremos de bordes, extraer información mediante formas y codificar la información del píxel central.

 #### Solidez 
 La solidez representa la compacidad de una lesión pulmonar, calculada dividiendo el área de la lesión por el área de su superficie convexa.

### Características LBP - *Local Binary Pattern (LBP) features*

 LBP es un modelo estadístico utilizado en el procesamiento de imágenes para extraer características de las mismas, de manera eficiente. Los valores LBP se obtienen binarizando las disparidades entre píxeles adyacentes en cada imagen.

 Este proceso implica reorganizar los píxeles y aplicar una función escalonada para obtener la representación binaria de las variaciones locales.
 
 El operador LBP proporciona una representación compacta de la textura local de la imagen al codificar las relaciones espaciales entre los píxeles de un vecindario.

 Las LBP features son: **ConvexArea**, **lbp_energy** y **lbp_entropy**

### Características pulmonares - *Lung Features*

 Los profesionales médicos evalúan la **densidad pulmonar** anormal comparando las zonas superior, media e inferior de ambos pulmones, izquierdo y derecho. El estudio calcula la blancura anormal, que indica una mayor densidad, o la negrura anormal, que indica una menor densidad, utilizando los niveles de intensidad de la imagen. Aquí se extraen la densidad media, la **densidad estándar**, el **volumen pulmonar** y **los puntos clave de la Transformación de Características Invariantes a la Escala (SIFT)**.
 La ​​densidad media es una métrica que se utiliza para cuantificar la distribución de las intensidades de los píxeles de nivel de gris dentro de una imagen de la región de interés (ROI) del pulmón. Indica la intensidad promedio de los niveles de gris presentes en la imagen, donde un valor más alto indica una suma acumulativa más significativa de niveles de gris dentro de la imagen.

 La desviación estándar mide la variabilidad en los valores de intensidad de los píxeles, indicando la diferencia entre las intensidades de los píxeles y los valores medios de las características.

 La representación del descriptor SIFT para la localización de puntos críticos extrae características locales distintivas mediante coordenadas invariantes a la escala. Este método filtra eficazmente los patrones irregulares y de bajo contraste, resaltando los puntos relevantes, y es resistente a las deformaciones locales y a los errores de detección de características.

 Los volúmenes pulmonares pueden ser dinámicos o estáticos, dependiendo del flujo de aire y de la inspiración o espiración. Los volúmenes pulmonares dinámicos son cruciales para el diagnóstico y el seguimiento de las enfermedades respiratorias obstructivas, mientras que los volúmenes pulmonares estáticos son esenciales para evaluar los defectos respiratorios obstructivos y restrictivos. El volumen pulmonar es un indicador clínico para el diagnóstico de enfermedades infecciosas que afectan al volumen pulmonar.
 Los cambios en la capacidad pulmonar debidos al aumento de la actividad respiratoria afectan a la flexibilidad y la presión durante la respiración. Un volumen pulmonar total anormal puede indicar afecciones graves como cardiomegalia, neumonía viral o bronquitis.


### Característica de textura derivadas de (DC-GLM) - Texture features derived from Dynamic Co-Occurrence Grey Level Matrix (DC-GLM)

 medida numérica obtenida de una imagen para representar propiedades de su textura, como contraste, homogeneidad, energía o correlación. Estas características convierten información visual de una imagen en variables que posteriormente pueden utilizarse como entrada de un algoritmo de aprendizaje automático.

 #### Momento angular secundario (Angular Second Moment, ASM):
  medida estadística utilizada para caracterizar la uniformidad de una textura, valores elevados indican una distribución de textura más uniforme.

### Otros términos asociados

  - Coocurrencia: concepto estadístico que describe la aparición conjunta de dos valores dentro de una determinada relación espacial

 - Filtro de Gabor (Gabor filter): operador de procesamiento de imágenes diseñado para detectar estructuras que poseen determinadas frecuencias y orientaciones. Combina una función gaussiana con una señal sinusoidal, por lo que resulta especialmente útil para identificar patrones de textura y estructuras orientadas.

 - Filtrado multiescala (Multiscale filtering): estrategia que analiza una imagen utilizando diferentes escalas para detectar simultáneamente estructuras pequeñas, medianas y grandes. 


 - Difusión anisotrópica (Anisotropic diffusion): técnica de procesamiento de imágenes utilizada para reducir ruido sin eliminar completamente los bordes importantes. A diferencia de un suavizado uniforme, la difusión depende de las características locales de la imagen, permitiendo preservar estructuras relevantes. 


 - Difusión adaptativa contextual (Contextual Adaptive Diffusion): método de procesamiento utilizado por CAMSGNeT que controla la difusión de cada región considerando tanto el gradiente de la imagen como la varianza local de las intensidades. De esta manera, puede suavizar regiones relativamente uniformes mientras conserva detalles en zonas con estructuras importantes.


 - Gradiente de imagen (Image gradient): representación matemática de cómo cambia la intensidad de una imagen espacialmente. Un gradiente elevado suele indicar una transición fuerte entre regiones, por ejemplo, un borde.










