import mxnet as mx
import numpy as np
import pickle
import cv2
import os

def extract_images_and_labels(path, file):
    with open(os.path.join(path, file), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    images = data_dict[b'data']
    images = images.reshape((10000, 3, 32, 32))
    labels = data_dict[b'labels']
    image_array = mx.nd.array(images)
    label_array = mx.nd.array(labels)
    return image_array, label_array

def extract_categories(path, file):
    with open(os.path.join(path, file), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return [label.decode('utf-8') for label in data_dict[b'label_names']]

def save_cifar_image(array, path, file, target_size=(128, 128)):
    # Convertir MXNet NDArray en tableau NumPy et transposer pour correspondre au format OpenCV
    array = array.asnumpy().transpose(1, 2, 0)
    # Convertir RGB en BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # Redimensionner l'image avec interpolation bicubique
    array = cv2.resize(array, target_size, interpolation=cv2.INTER_CUBIC)
    # Enregistrer en tant que fichier JPG
    return cv2.imwrite(os.path.join(path, f"{file}.jpg"), array)

# Définir les chemins
data_path = "./data/cifar-10-batches-py/cifar-10-batches-py/"
image_save_path = "./static/"

# Créer le dossier de sauvegarde s'il n'existe pas
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

# Charger toutes les images et les étiquettes d'entraînement
img_arrays = []
lbl_arrays = []
for batch in range(1, 6):
    img_array, lbl_array = extract_images_and_labels(data_path, f"data_batch_{batch}")
    img_arrays.append(img_array)
    lbl_arrays.append(lbl_array)

# Charger les images et les étiquettes de test
test_img_array, test_lbl_array = extract_images_and_labels(data_path, "test_batch")
img_arrays.append(test_img_array)
lbl_arrays.append(test_lbl_array)

# Concaténer toutes les images et étiquettes
img_arrays = mx.nd.concat(*img_arrays, dim=0)
lbl_arrays = mx.nd.concat(*lbl_arrays, dim=0)
print(f"Image array shape: {img_arrays.shape}")
print(f"Label array shape: {lbl_arrays.shape}")

# Charger les noms des catégories
categories = extract_categories(data_path, "batches.meta")

# Définir la taille cible pour redimensionner les images (par exemple 256x256)
target_size = (256, 256)

# Sauvegarder toutes les images redimensionnées avec interpolation bicubique et imprimer leurs catégories
cats = []
for i in range(img_arrays.shape[0]):
    save_cifar_image(img_arrays[i], image_save_path, f"image_{i}", target_size=target_size)
    category_index = int(lbl_arrays[i].asscalar())
    cats.append(categories[category_index])
print(cats)
