import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import faiss
import mxnet as mx
from mxnet.gluon.model_zoo import vision
import time

app = Flask(__name__)

# Fonction pour télécharger et initialiser le modèle ResNet50
def download_resnet50_model(retries=5, delay=5):
    cache_dir = os.path.expanduser('~/.mxnet/models/')
    model_filename = 'resnet50_v2-ecdde353.zip'
    model_path = os.path.join(cache_dir, model_filename)
    
    if os.path.exists(model_path):
        print("ResNet50 model already exists. Skipping download.")
        model = vision.resnet50_v2(pretrained=True, ctx=mx.cpu())
        model.hybridize()  # Enable optimization
        return model

    download_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/models/resnet50_v2-ecdde353.zip'
    try:
        print(f"Downloading ResNet50 model from {download_url}...")
        mx.test_utils.download(download_url, path=cache_dir, overwrite=False)
        print("Download complete.")
        model = vision.resnet50_v2(pretrained=True, ctx=mx.cpu())
        model.hybridize()  # Enable optimization
        return model
    except Exception as e:
        print(f"Failed to download the ResNet50 model: {e}")
        raise RuntimeError("Failed to download the ResNet50 model.")

# Initialisation du modèle et de l'index FAISS
model = download_resnet50_model()
index = None  # Variable pour l'index FAISS

# Fonction pour prétraiter une image avant l'extraction des features
def preprocess_image(image):
    image = mx.nd.array(image)
    image = mx.image.imresize(image, 224, 224)  # Redimensionner à 224x224 pour ResNet50
    image = mx.nd.transpose(image, (2, 0, 1))  # Convertir en (C, H, W)
    image = image.expand_dims(axis=0)  # Ajouter une dimension de batch
    image = image.astype('float32') / 255  # Normaliser les valeurs des pixels
    return image

# Route principale pour l'upload d'image et la recherche
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Vérifier si un fichier a été envoyé
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # Vérifier si le fichier est vide
        if file.filename == '':
            return redirect(request.url)
        # Vérifier si le fichier est bien une image
        if file and allowed_file(file.filename):
            # Sauvegarder le fichier dans un dossier temporaire
            filepath = os.path.join('./static/', file.filename)
            file.save(filepath)
            # Effectuer la recherche d'images similaires
            similar_images = search_similar_images(filepath)
            # Afficher les résultats
            return render_template('result.html', query_image=filepath, similar_images=similar_images)
    # Afficher le formulaire pour l'upload d'image
    return render_template('index.html')

# Fonction pour rechercher des images similaires
def search_similar_images(filepath, top_k=5):
    global index  # Utiliser l'index FAISS initialisé globalement
    if index is None:
        # Charger les features et l'index FAISS s'ils existent déjà
        if os.path.exists("features.npy") and os.path.exists("index.faiss"):
            print("Loading features and FAISS index from disk.")
            features = np.load("features.npy")
            index = faiss.read_index("index.faiss")
        else:
            # Extraire les features et construire un nouvel index FAISS
            features, labels = extract_features_and_labels()
            d = features.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(features)
            # Sauvegarder les features et l'index FAISS pour une utilisation ultérieure
            np.save("features.npy", features)
            faiss.write_index(index, "index.faiss")
            print("Features and FAISS index saved to disk.")
    
    # Charger l'image et extraire ses features
    query_image = mx.image.imread(filepath)
    query_image = preprocess_image(query_image)
    query_features = model(query_image).asnumpy().flatten()
    query_features = np.expand_dims(query_features, axis=0).astype('float32')

    # Rechercher les images similaires dans l'index FAISS
    D, I = index.search(query_features, top_k)

    # Récupérer les chemins des images similaires
    similar_images = []
    for i in range(top_k):
        index_file = f"image_{I[0][i]}.jpg"  # Supposer que vos images sont nommées image_0.jpg, image_1.jpg, etc.
        distance = float(D[0][i])
        similar_images.append((index_file, distance))

    return similar_images

# Fonction pour extraire les features et labels des images CIFAR-10
def extract_features_and_labels():
    data_path = "./data/cifar-10-batches-py/cifar-10-batches-py/"
    img_arrays = []
    lbl_arrays = []
    for batch in range(1, 6):
        img_array, lbl_array = extract_images_and_labels(data_path, f"data_batch_{batch}")
        img_arrays.append(img_array)
        lbl_arrays.append(lbl_array)

    img_arrays = mx.nd.concat(*img_arrays, dim=0)
    features = extract_features(img_arrays, model)
    labels = np.concatenate(lbl_arrays, axis=0)

    return features, labels

# Fonction pour extraire les images et labels du fichier CIFAR-10
def extract_images_and_labels(path, file):
    with open(os.path.join(path, file), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    images = data_dict[b'data']
    images = images.reshape((10000, 3, 32, 32))
    labels = data_dict[b'labels']
    image_array = mx.nd.array(images)
    label_array = np.array(labels)  # Convertir en numpy array
    return image_array, label_array

# Fonction pour extraire les features d'un ensemble d'images
def extract_features(images, model):
    features = []
    for img in images:
        img = preprocess_image(img)
        feat = model(img)
        features.append(feat.asnumpy().flatten())
    return np.array(features)

# Vérifier si le fichier est autorisé (extension d'image)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
