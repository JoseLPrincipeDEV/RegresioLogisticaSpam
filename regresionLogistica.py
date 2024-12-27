import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Leer el archivo CSV y crear un DataFrame
df = pd.read_csv('spam.csv')

# Convertir las etiquetas de 'ham' y 'spam' a 0 y 1
df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})

# Seleccionar las columnas Email y Spam
df_seleccionado = df[['v2', 'v1']]

# Definir las características (X) y la variable objetivo (y)
X = df_seleccionado['v2']
y = df_seleccionado['v1']

# Convertir el texto en una representación numérica usando TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Convertir el texto en una representación numérica usando CountVectorizer
cv = CountVectorizer(stop_words="english")
data_cv = cv.fit_transform(X)

# Crear un diccionario de palabras y sus puntuaciones TF-IDF
tfidf_matrix = vectorizer.fit_transform(X)
palabras = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

# Imprimir las palabras y sus puntuaciones
for palabra, puntuacion in palabras.items():
    print(palabra, puntuacion)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Evaluar el modelo
score = modelo.score(X_test, y_test)
print(f'Precisión del modelo: {score}')

# Función para predecir si un correo es spam o no
def predecir_spam(correo):
    correo_tfidf = vectorizer.transform([correo])
    prediccion = modelo.predict(correo_tfidf)
    return 'spam' if prediccion[0] == 1 else 'NoSpam'

# Ejemplo de uso de la función
nuevo_correo = "Hi team, Just a reminder that we have a work meeting tomorrow at 10 AM in the conference room. Please bring your updated reports and any relevant materials for discussion. Best, jose"
resultado = predecir_spam(nuevo_correo)
print(f'El correo "{nuevo_correo}" es {resultado}.')
