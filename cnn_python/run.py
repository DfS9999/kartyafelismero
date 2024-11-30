import argparse
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from augment import correctExifRotation
from train import CardClassifierModel_A, CardClassifierModel_B, CardClassifierModel_Efficientnet_b0


def load_model(model_type, model_path, class_labels, device):
    class_count = len(class_labels)
    model = model_type(class_count=class_count)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    model.eval()
    
    return model

def show_model_predictions(images_dir, model, class_labels, transform, device):
    images = list(Path(images_dir).glob("*.jpg"))
    
    for image_path in images:
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = torch.nn.functional.softmax(model(input_tensor), dim=1)
            predictions = output.numpy().flatten()
        
        # top 5
        top5_idxs = np.argsort(predictions)[-5:]
        top5_labels      = [class_labels[idx] for idx in top5_idxs]
        top5_predictions = [predictions[idx]  for idx in top5_idxs]
        
        # show image + top 5 predicitons
        fig, (subplot_1, subplot_2) = plt.subplots(1, 2, figsize=(12, 6))
        subplot_1.imshow(np.asarray(img))
        subplot_1.axis('off')

        subplot_2.barh(top5_labels, top5_predictions)
        subplot_2.set_xlim(0, 1)
        subplot_2.set_xlabel('Probability')
        for i, p in enumerate(top5_predictions):
            subplot_2.text(p, i, f"{p * 100:.1f}%", va='center', ha='left')

        plt.tight_layout()
        plt.show()

        input("Press enter for next")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=str)
    parser.add_argument("model", type=str, choices=['A', 'B', 'C'])
    parser.add_argument("model_path", type=str)
    parser.add_argument("model_train_images_dir", type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if args.model == 'A':
        model_type = CardClassifierModel_A
    elif args.model == 'B':
        model_type = CardClassifierModel_B
    elif args.model == 'C':
        model_type = CardClassifierModel_Efficientnet_b0
    else: 
        print("Unknown model. Exiting.")
        exit(1)
    
    class_labels = { v: k for k,v in ImageFolder(args.model_train_images_dir).class_to_idx.items() }
    
    model = load_model(model_type, args.model_path, class_labels, device)

    transform = transforms.Compose([
        transforms.Lambda(lambd=correctExifRotation),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    show_model_predictions(args.images_dir, model, class_labels, transform, device)

if __name__ == "__main__":
    main()

