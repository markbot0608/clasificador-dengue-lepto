import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==================== Entrenamiento del modelo ====================
excel_path = r"C:\Users\USER\Downloads\base_de_datos_app.xlsx"

# Cargar los datos
data = pd.read_excel(excel_path)
data.columns = data.columns.str.strip()  # Limpia espacios en los nombres

# Separar características y objetivo
X = data.drop(columns=['diagnóstico'])  # ¡Ojo! lleva tilde
y = data['diagnóstico']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
reporte = classification_report(y_test, y_pred, output_dict=True)

# Sensibilidades
sensibilidad_lepto = reporte['1']['recall']    # 1 = Leptospirosis
sensibilidad_dengue = reporte['0']['recall']   # 0 = Dengue

# ==================== Interfaz ====================
st.title('🧪 Clasificador Dengue - Leptospirosis')
st.markdown('Ingrese los datos del paciente para obtener una predicción.')

# Mostrar métricas del modelo
st.info(f"🔍 Precisión del modelo: {accuracy:.2%}")
st.info(f"🧬 Sensibilidad para Leptospirosis: {sensibilidad_lepto:.2%}")
st.info(f"🧬 Sensibilidad para Dengue: {sensibilidad_dengue:.2%}")

# Crear entradas automáticas para todas las variables
inputs = {}
for col in X.columns:
    if X[col].nunique() <= 2:
        inputs[col] = st.selectbox(col, [0, 1])
    else:
        inputs[col] = st.number_input(col, value=float(X[col].mean()))

# Botón de predicción
if st.button('Predecir'):
    input_df = pd.DataFrame([inputs])
    pred = model.predict(input_df)[0]
    resultado = 'Leptospirosis' if pred == 1 else 'Dengue'
    st.success(f'🩺 Predicción: {resultado}')
