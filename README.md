# Spam Email Classification

Este proyecto implementa un modelo de clasificación de correos electrónicos para identificar si un correo es spam o no spam utilizando Python y bibliotecas de aprendizaje automático como `scikit-learn` y `pandas`.

## Descripción

El proyecto emplea técnicas de procesamiento de texto como TF-IDF y Count Vectorization para convertir el texto de los correos en datos numéricos. Luego, utiliza un modelo de regresión logística para entrenar y predecir si un correo electrónico pertenece a la categoría de spam o no spam.

## Requisitos

Antes de ejecutar el código, asegúrate de instalar las siguientes bibliotecas:

- `pandas`
- `scikit-learn`

Puedes instalarlas ejecutando:
```bash
pip install pandas scikit-learn
```

## Estructura del código

1. **Carga de datos**:
   - El archivo CSV (`spam.csv`) se carga en un DataFrame de `pandas`.
   - Las etiquetas `ham` (no spam) y `spam` se convierten a valores numéricos (0 y 1).

2. **Selección de datos**:
   - Se seleccionan las columnas relevantes: `v2` (contenido del correo) y `v1` (etiqueta spam).

3. **Procesamiento de texto**:
   - Se aplica TF-IDF para transformar el texto en una representación numérica.
   - También se utiliza Count Vectorization para comparar diferentes enfoques.

4. **Entrenamiento del modelo**:
   - Los datos se dividen en conjuntos de entrenamiento y prueba.
   - Se utiliza una regresión logística para entrenar el modelo.

5. **Evaluación del modelo**:
   - Se calcula la precisión del modelo en el conjunto de prueba.

6. **Predicción**:
   - Se define una función para predecir si un correo es spam o no basándose en el modelo entrenado.

## Archivos necesarios

- `spam.csv`: Archivo CSV con los datos del proyecto. Debe contener al menos las siguientes columnas:
  - `v1`: Etiqueta (`ham` o `spam`).
  - `v2`: Texto del correo electrónico.

## Cómo ejecutar el código

1. Asegúrate de que `spam.csv` esté en el mismo directorio que el archivo Python.
2. Ejecuta el script.
3. La precisión del modelo y las predicciones se mostrarán en la consola.

### Ejemplo de uso

El script incluye un ejemplo de predicción:
```python
nuevo_correo = "Hi team, Just a reminder that we have a work meeting tomorrow at 10 AM in the conference room. Please bring your updated reports and any relevant materials for discussion. Best, jose"
resultado = predecir_spam(nuevo_correo)
print(f'El correo "{nuevo_correo}" es {resultado}.')
```

### Salida esperada

El modelo predice si el correo es `spam` o `NoSpam`. Por ejemplo:
```plaintext
El correo "Hi team, Just a reminder that we have a work meeting tomorrow at 10 AM in the conference room. Please bring your updated reports and any relevant materials for discussion. Best, jose" es NoSpam.
```

## Contacto

Si tienes preguntas o sugerencias sobre este proyecto, no dudes en ponerte en contacto.
