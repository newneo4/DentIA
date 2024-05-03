import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np

# Ruta del modelo previamente entrenado (solo los pesos)
weights_path = 'D:/Curso Python/Tarea Universidad/Proyecto_IA/modelo_salud_dental.h5'

# Crear una función para construir el modelo
def build_model():
    base_model = MobileNetV2(weights=None, include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Crear una función para cargar los pesos en el modelo
def load_weights(model, weights_path):
    model.load_weights(weights_path)
    return model

# Crear una función para procesar imágenes
def process_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    # Realizar la predicción
    predictions = model.predict(img)
    
    # Decodificar las predicciones
    labels = {0: "Caries", 1: "Gingivitis", 2: "Herpes", 3: "Dientes sanos"}
    max_prediction_index = np.argmax(predictions, axis=1)[0]
    prediction_label = labels[max_prediction_index]
    
    return prediction_label

# Crear una función para cargar una imagen y mostrar la predicción
def load_and_predict():
    file_path = filedialog.askopenfilename(initialdir="/", title="Seleccione una imagen",
                                          filetypes=(("Archivos de imagen", "*.jpg;*.jpeg;*.png"), ("Todos los archivos", "*.*")))
    if file_path:
        prediction_label = process_image(file_path)
        result_label.config(text=f"Predicción: {prediction_label}")

# Crear una ventana de GUI simple
root = tk.Tk()
root.title("Detector de problemas dentales")

# Construir el modelo y cargar los pesos
model = build_model()
model = load_weights(model, weights_path)

load_button = tk.Button(root, text="Cargar imagen y predecir", command=load_and_predict)
load_button.pack(pady=20)

result_label = tk.Label(root, text="", font=("Calibri", 16))
result_label.pack()

root.mainloop()
