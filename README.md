# Clasificador de Mobs de Minecraft con CNN

Proyecto de clasificación de imágenes usando Redes Neuronales Convolucionales (CNN); desde cero y con un modelo preentrenado (mobileNet para transfer learning) para identificar diferentes mobs de Minecraft.

## Notas

- **Tamaño de imágenes**: El modelo espera 224×224. (Actualmente) Se deben configurar las rutas a las carpetas de las imagenes originales de cada clase y correr el script de redimensionamiento.
- **Formato soportado**: PNG
- **GPU recomendada**: Para entrenamiento rápido, aunque funciona en CPU.
- **Multiples archivos de training y testing**: Se utilizó un PC portatil y de mesa, para cada uno se creo archivos de training y testing ya que no funcionó por versionamiento -> Utilizar un entorno virtual para solucionarlo


## Descripción

Este proyecto implementa un clasificador de imágenes que puede identificar tres tipos de mobs de Minecraft:
- Aldeanos
- Creepers
- Vacas

Se ofrecen dos enfoques de entrenamiento:
1. **Transfer Learning** con MobileNetV2 (recomendado para datasets pequeños)
2. **CNN desde cero** (para experimentación)


##  Estructura del Proyecto
```
Dataset/
├── Imagenes originales/               # Imagenes originales del dataset
│   ├── aldeanos/
│   ├── creepers/
│   └── vacas/
├── Imagenes redimensionadas/          # Dataset organizado por clases redmensionadas. Creada despues de correr el script para cada clase
│   ├── aldeanos/
│   ├── creepers/
│   └── vacas/
├── Imagenes originales etiquetadas/   # Imágenes anotadas con LabelMe
├── RedimensionarImagenes.ipynb   # Script para redimensionar las imagenes originales en imagenes 224x224 #Recordar cambiar la ruta
├── EntrenamientoCNN(portatil).ipynb   # Notebook de entrenamiento de CNN desde cero
├── TestingCNN(portatil).ipynb         # Notebook de predicción con modelo CNN desde Cero
├── Entrenamiento.ipynb   # Notebook de entrenamiento en PC de mesa con MobileNet
├── Entrenamiento_portatil.ipynb   # Notebook de entrenamiento en PC portatil MobileNet
├── Testing_Portatil.ipynb    # Notebook donde se prueba el modelo en PC portatil
├── minecraft_mob_classifier_final(portatil_cnn).h5  # Modelo entrenado
└── best_minecraft_model(portatil_cnn).h5            # Mejor modelo guardado

!: Las imagenes que estan por fuera de las carpetas son las imagenes que se usaron para testear los modelos guardados.

```

## Instalación

```bash
# Clonar el repositorio (o descargar archivos)
git clone <tu-repo>
cd Dataset

# Instalar dependencias
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow


## Preparación del Dataset

### Opción 1: Dataset ya organizado
Organiza tus imágenes en carpetas por clase:
```
Imagenes originales/
├── aldeanos/
│   ├── img001.png
│   └── img002.png
├── creepers/
└── vacas/
```

```

## Entrenamiento

### Uso Básico

```python
from EntrenamientoCNN import MinecraftCNNTrainer

# Configuración
DATASET_PATH = "ruta/a/Imagenes redimensionadas"
trainer = MinecraftCNNTrainer(DATASET_PATH, img_size=224, batch_size=32)

# Preparar datos
trainer.setup_data_generators()

# Crear modelo (Transfer Learning o desde cero)
trainer.create_model(use_transfer_learning=True)  # True = MobileNetV2, False = CNN custom

# Entrenar
trainer.train_model(epochs=30)

# Evaluar
trainer.plot_training_history()
trainer.evaluate_model()

# Guardar
trainer.save_model("mi_modelo.h5")
```

### Parámetros del Entrenamiento

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `img_size` | 224 | Tamaño de las imágenes |
| `batch_size` | 32 | Tamaño del batch |
| `epochs` | 30 | Número de épocas |
| `learning_rate` | 0.001 | Tasa de aprendizaje |
| `validation_split` | 0.2 | 20% para validación |

### Callbacks Implementados

- **EarlyStopping**: Detiene si no mejora en 10 épocas
- **ModelCheckpoint**: Guarda el mejor modelo automáticamente
- **ReduceLROnPlateau**: Reduce learning rate si se estanca

## Predicción

### Uso Básico

```python
from TestingCNN import MinecraftPredictor

# Cargar modelo
predictor = MinecraftPredictor(
    model_path="minecraft_mob_classifier_final(portatil_cnn).h5",
    class_names=['aldeanos', 'creepers', 'vacas']
)

# Predecir una imagen (Usar las que estan en la raiz, no hacen parte del entrenamiento)
predictor.predict_and_show("imagen_prueba.png")

# Predecir carpeta completa
predictor.batch_predict("carpeta_imagenes/")
```

### Salida de Ejemplo

```
📸 Imagen: 2025-09-24_08.03.45.png
🎯 Predicción: creepers
🎲 Confianza: 94.23%

📊 Todas las probabilidades:
  aldeanos: 2.15%
  creepers: 94.23%
  vacas: 3.62%
```

## Métricas y Resultados

El entrenamiento genera:
- **Gráficas de accuracy y loss** (training vs. validation)
- **Matriz de confusión** con Seaborn
- **Classification report** con precision, recall, F1-score

Ejemplo de salida:
```
              precision    recall  f1-score   support
   aldeanos       0.89      0.92      0.91        93
   creepers       0.95      0.93      0.94       107
      vacas       0.88      0.85      0.86        89

   accuracy                           0.91       289
```

## Arquitecturas Implementadas

### 1. Transfer Learning (MobileNetV2)
```
MobileNetV2 (ImageNet) → GlobalAveragePooling2D → Dense(128) → Dense(num_classes)
```

### 2. CNN desde Cero
```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → 
Conv2D(128) → MaxPool → Flatten → Dense(512) → Dense(num_classes)
```

## Configuración Avanzada

### Data Augmentation
```python
ImageDataGenerator(
    rotation_range=20,       # Rotaciones ±20°
    width_shift_range=0.2,   # Desplazamientos horizontal
    height_shift_range=0.2,  # Desplazamientos vertical
    zoom_range=0.2,          # Zoom in/out
    horizontal_flip=True     # Flip horizontal
)
```

### Ajustar Hiperparámetros
```python
# En create_model()
optimizer=Adam(learning_rate=0.0005)  # Reducir LR

# En setup_callbacks()
patience=15  # Más paciencia antes de parar
```

## Mejoras Futuras

- [ ] Implementar algoritomos para detección con bounding boxes
- [ ] Segmentación de instancias
- [ ] Aumentar clases (zombies, esqueletos, etc.)

## 📧 Contacto

**Autores**: Juan Pablo Rios Ortiz, Cristian Troncoso Guerra, Daniel Restrepo Villa
**Universidad**: Institución Universitaria de Envigado
**Curso**: IA II - Semestre 2025-2
---

