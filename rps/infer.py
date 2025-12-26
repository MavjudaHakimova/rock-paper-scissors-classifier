import torch
import lightning as L
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import joblib
from model import FeatureExtractor
from data import RPSDataModule

def infer(model_file: str, image_path: str, batch_size: int = 1):
    # Загружаем обученный CatBoost
    classifier = joblib.load('rps_classifier.pkl')
    feature_extractor = FeatureExtractor()
    
    # Загружаем и предобрабатываем изображение
    transform = torch.nn.Sequential(
        torch.nn.Upsample(size=(224, 224)),
        torch.nn.Lambda(lambda x: x * 2 - 1.0)
    )
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(torch.from_numpy(np.array(img).transpose(2,0,1)[None].astype(np.float32)/255.0)).squeeze(0)
    
    # Feature extraction
    with torch.no_grad():
        features = feature_extractor(img_tensor[None]).numpy()
    
    # Prediction
    predictions = classifier.predict_proba(features)[0]
    class_names = ['Rock', 'Paper', 'Scissors']
    
    max_idx = np.argmax(predictions)
    max_class = class_names[max_idx]
    confidence = predictions[max_idx]
    
    plt.imshow(img)
    plt.title(f"Predicted: {max_class} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()
    
    return max_class, confidence, predictions

if __name__ == "__main__":
    infer(None, '/kaggle/input/rock-paper-scissors-dataset/Rock-Paper-Scissors/validation/paper-hires2.png')
