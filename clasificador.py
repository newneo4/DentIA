import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, PhotoImage
from PIL import Image, ImageTk, ImageFont
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import os
import cv2 as cv
import numpy as np

carpeta_principal = os.path.dirname(__file__)
carpeta_imagenes = os.path.join(carpeta_principal, "imagenes")
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ImageClassifierGUI:

    def __init__(self, model_path):
        self.model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        self.labels = {0: "Caries", 1: "Gingivitis", 2: "Herpes", 3: "Dientes sanos"}
        self.root = ctk.CTk()
        self.root.title("Detector de problemas dentales")
        self.root.iconbitmap(os.path.join(carpeta_imagenes,"icono.ico"))
        self.root.geometry("400x480")
        self.root.resizable(False,False)

        imagen2 = ctk.CTkImage(dark_image=Image.open(os.path.join(carpeta_imagenes,"fondo_pred.png")), size=(400,480))
        self.etiqueta_img2 = ctk.CTkLabel(master=self.root, image=imagen2, text=" ")
        self.etiqueta_img2.place(relx = 0.5, rely = 0.5, anchor = tk.CENTER)

        self.image_label = tk.Label(self.root, text = None)
        self.image_label.place(relx = 0.5, rely= 0.53, anchor = tk.CENTER)
        self.result_label = ctk.CTkLabel(self.root, text = "", font=("Calibri", 27))
        self.result_label.place(relx = 0.5, rely = 0.23, anchor = tk.CENTER)

        imagen = ctk.CTkImage(dark_image=Image.open(os.path.join(carpeta_imagenes,"fondofinal.png")), size=(400,480))
        self.etiqueta_img = ctk.CTkLabel(master=self.root, image=imagen, text=" ")
        self.etiqueta_img.place(relx = 0.5, rely = 0.5, anchor = tk.CENTER)

        boton_image = Image.open(os.path.join(carpeta_imagenes, "botonfinal2.png"))
        boton_image = boton_image.resize((300, 100))
        boton_sel = ImageTk.PhotoImage(boton_image)

        # Crear el botón con el diseño personalizado
        self.button = tk.Button(self.root, image=boton_sel, command=self.select_image, highlightthickness=0, bd=0)
        self.button.image = boton_sel  # Mantener referencia a la imagen
        self.button.place(relx=0.5, rely=0.45, anchor=tk.CENTER)

        boton_image2 = Image.open(os.path.join(carpeta_imagenes, "botonfinal.png"))
        boton_image2 = boton_image2.resize((300, 100))
        boton_sel2 = ImageTk.PhotoImage(boton_image2)

        # Crear el botón con el segundo diseño personalizado
        self.button2 = tk.Button(self.root, image=boton_sel2, command=self.camara, highlightthickness=0, bd=0)
        self.button2.image = boton_sel2  # Mantener referencia a la imagen
        self.button2.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

        self.mostrar = True

    def camara(self):
        #El puerto para el celular es 2
        #El puerto para la webcam es 0
        capture = cv.VideoCapture(0)
        imagen = None

        while True:
            isTrue, frame = capture.read()

            cv.imshow('Video', frame)

            if cv.waitKey(20) & 0xFF==ord('d'):
                imagen = "muestra.png"
                cv.imwrite(imagen, frame)
                break

        capture.release()
        cv.destroyAllWindows()
        
        self.select_image(imagen)

        boton_image = Image.open(os.path.join(carpeta_imagenes, "botonfinal2.png"))
        boton_image = boton_image.resize((200, 60))
        boton_sel = ImageTk.PhotoImage(boton_image)

        # Crear el botón con el diseño personalizado
        self.button = tk.Button(self.root, image=boton_sel, command=self.select_image, highlightthickness=0, bd=0)
        self.button.image = boton_sel  # Mantener referencia a la imagen
        self.button.place(relx=0.25, rely=0.85, anchor=tk.CENTER)

        boton_image2 = Image.open(os.path.join(carpeta_imagenes, "botonfinal.png"))
        boton_image2 = boton_image2.resize((200, 60))
        boton_sel2 = ImageTk.PhotoImage(boton_image2)

        # Crear el botón con el segundo diseño personalizado
        self.button2 = tk.Button(self.root, image=boton_sel2, command=self.camara, highlightthickness=0, bd=0)
        self.button2.image = boton_sel2  # Mantener referencia a la imagen
        self.button2.place(relx=0.75, rely=0.85, anchor=tk.CENTER)

    def select_image(self, muestra = None):

        if muestra == None:
            file_path = filedialog.askopenfilename(initialdir="/", title="Seleccione una imagen",
                                              filetypes=(("Archivos de imagen", "*.jpg;*.jpeg;*.png"), ("Todos los archivos", "*.*")))

            if self.mostrar:
                self.button.destroy()
                self.button2.destroy()
                self.etiqueta_img.destroy()
                self.mostrar = False

            if file_path:
                image = Image.open(file_path)
                image = image.resize((300, 300))
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo

                boton_image = Image.open(os.path.join(carpeta_imagenes, "botonfinal2.png"))
                boton_image = boton_image.resize((200, 60))
                boton_sel = ImageTk.PhotoImage(boton_image)

                # Crear el botón con el diseño personalizado
                self.button = tk.Button(self.root, image=boton_sel, command=self.select_image, highlightthickness=0, bd=0)
                self.button.image = boton_sel  # Mantener referencia a la imagen
                self.button.place(relx=0.25, rely=0.85, anchor=tk.CENTER)

                boton_image2 = Image.open(os.path.join(carpeta_imagenes, "botonfinal.png"))
                boton_image2 = boton_image2.resize((200, 60))
                boton_sel2 = ImageTk.PhotoImage(boton_image2)

                # Crear el botón con el segundo diseño personalizado
                self.button2 = tk.Button(self.root, image=boton_sel2, command=self.camara, highlightthickness=0, bd=0)
                self.button2.image = boton_sel2  # Mantener referencia a la imagen
                self.button2.place(relx=0.75, rely=0.85, anchor=tk.CENTER)

                prediction_label = self.predict_image(file_path)
                self.result_label.configure(text=f"Predicción: {prediction_label}")


        if self.mostrar:
            self.button.destroy()
            self.button2.destroy()
            self.etiqueta_img.destroy()
            self.mostrar = False

        if muestra:
            image = Image.open(muestra)
            image = image.resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            prediction_label = self.predict_image(muestra)
            self.result_label.configure(text=f"Predicción: {prediction_label}")
        

    #funcion q contiene la IA predictora 
    def predict_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        img = tf.expand_dims(img, axis=0)

        predictions = self.model.predict(img)
        max_prediction_index = tf.argmax(predictions, axis=1)[0].numpy()
        max_prediction_label = self.labels.get(max_prediction_index, "Desconocido")

        return max_prediction_label

    def run(self):
        self.root.mainloop()

gui = ImageClassifierGUI('salud_dental.h5')
gui.run()
