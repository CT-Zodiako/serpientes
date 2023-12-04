# Importa las bibliotecas necesarias
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


data = pd.read_csv('csv.csv')


le = LabelEncoder()
data['Color'] = le.fit_transform(data['Color'])
data['Patrón'] = le.fit_transform(data['Patrón'])


X = data[['Longitud', 'Color', 'Patrón']]
y = data['Veneno']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Precisión del modelo en el conjunto de prueba: {accuracy * 100:.2f}%')

longitud = float(input("Ingresa la longitud de la serpiente: "))
color = input("Ingresa el color de la serpiente: ")
patron = input("Ingresa el patrón de la serpiente: ")

color_encoded = le.transform([color])[0] if color in le.classes_ else -1
patron_encoded = le.transform([patron])[0] if patron in le.classes_ else -1


color_encoded = color_encoded if color_encoded != -1 else 0  
patron_encoded = patron_encoded if patron_encoded != -1 else 0 

prediction = model.predict([[longitud, color_encoded, patron_encoded]])

resultado = "venenosa" if prediction[0] == 'Sí' else "no venenosa"

print(f"Según las características ingresadas, la serpiente es clasificada como: {resultado}")
