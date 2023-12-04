# Importa las bibliotecas necesarias
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Carga tus datos (asegúrate de tener un conjunto de datos etiquetado)
# Supongamos que tienes un conjunto de datos en un archivo CSV con columnas como 'Longitud', 'Color', 'Patrón', 'Veneno' (etiqueta)
data = pd.read_csv('csv.csv')

# Preprocesamiento de datos
le = LabelEncoder()
data['Color'] = le.fit_transform(data['Color'])
data['Patrón'] = le.fit_transform(data['Patrón'])

# Divide el conjunto de datos en características (X) y etiquetas (y)
X = data[['Longitud', 'Color', 'Patrón']]
y = data['Veneno']

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea el modelo de clasificación (usaremos Random Forest como ejemplo)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
predictions = model.predict(X_test)

# Calcula la precisión del modelo
accuracy = accuracy_score(y_test, predictions)
print(f'Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%')

# Solicita al usuario ingresar las características de la serpiente
longitud = float(input("Ingresa la longitud de la serpiente: "))
color = input("Ingresa el color de la serpiente: ")
patron = input("Ingresa el patrón de la serpiente: ")

# Aplica la transformación usando las clases aprendidas en el conjunto de entrenamiento
color_encoded = le.transform([color])[0] if color in le.classes_ else -1
patron_encoded = le.transform([patron])[0] if patron in le.classes_ else -1

# Maneja las nuevas categorías de manera más robusta, por ejemplo, asignando un valor por defecto
color_encoded = color_encoded if color_encoded != -1 else 0  # Asigna 0 si es una categoría nueva
patron_encoded = patron_encoded if patron_encoded != -1 else 0  # Asigna 0 si es una categoría nueva

# Realiza la predicción
prediction = model.predict([[longitud, color_encoded, patron_encoded]])

# Mapea la salida a una interpretación legible
resultado = "venenosa" if prediction[0] == 'Sí' else "no venenosa"

print(f"Según las características ingresadas, la serpiente es clasificada como: {resultado}")
