from flask import Flask, render_template, request
import pickle

# Crear la app de Flask
app = Flask(__name__)

# ========================
# CARGAR EL MODELO .sav
# ========================
# Ojo: el archivo .sav debe estar en la misma carpeta que app.py
with open("boosting_classifier_nestimators-20_learnrate-0.001_42.sav", "rb") as f:
    model = pickle.load(f)

# ========================
# RUTA PRINCIPAL (formulario vacío)
# ========================
@app.route("/", methods=["GET"])
def home():
    # Mostramos la plantilla sin predicción todavía
    return render_template("index.html")

# ========================
# RUTA DE PREDICCIÓN
# ========================
@app.route("/predict", methods=["POST"])
def predict():
    # 1. Recoger los valores del formulario y convertirlos a float
    vals = [float(request.form[f]) for f in [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]]

    # 2. Pasar los valores al modelo
    # Nota: se mete dentro de otra lista porque el modelo espera [ [x1, x2, ...] ]
    y_hat = model.predict([vals])[0]

    # 3. Renderizar otra vez index.html pero ahora con la predicción
    return render_template("index.html", prediction=int(y_hat))

# ========================
# EJECUTAR LA APP
# ========================
if __name__ == "__main__":
    app.run(debug=True)