# selfie_filter

Mateo Torres, PhD @torresmateo

#### Filtros para Selfies

* Implementar una aplicación que utilice la cámara de la computadora para

    * Reconocer caras (al menos 2 caras)

    * Agregar filtros a esas caras

        * lentes, máscaras, coronas,

        * es suficiente colocar la imagen de forma correcta (rotación y escala proporcional al tamaño de la cara)

    * La aplicación debe “seguir” a la cara y mantener el filtro aplicado en tiempo real

* Se debe implementar usando python 

    * OpenCV

    * Keras / Tensorflow

    * Se puede usar todo tipo de bibliotecas mientras sean open source

#### Desafíos

* Distorsiones al rostro (que se vea con un alien, agrandar la nariz, etc)

* Control de la interfaz con gestos (por ejemplo, haciendo una señal con las manos para aplicar el siguiente filtro)

	* Detección y localización del gesto

	* Múltiples gestos: siguiente filtro, anterior filtro, remover filtro, activar filtro

	* Tip: Es válido usar solo una región de la pantalla para los gestos

#### Componentes

* Stream desde OpenCV

* Reconocimiento de caras

* Procesamiento de caras en tiempo real

* Ubicación de los filtros (escalar, rotar, agregar a la imágen)

* Distorsión de las sección del rostro

Subir una captura de la app funcionando

Se puede usar loom.com

Pueden enviar un archivo por correo si no quieren hacer público el video 

#### Keypoint detection

![](keypoint_detection.png =150x80)

Dataset: [https://www.kaggle.com/c/facial-keypoints-detection/data](https://www.kaggle.com/c/facial-keypoints-detection/data)