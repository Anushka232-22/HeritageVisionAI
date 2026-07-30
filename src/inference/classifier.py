import json
import torch
import timm

from torchvision import transforms
from PIL import Image


class DamageClassifier:

    def __init__(self, model_path="models/classifier/best_model_v2.pth",
                 classes_path="models/classifier/classes.json"):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load class list FIRST so num_classes always matches the checkpoint,
        # instead of hardcoding a number that can drift out of sync again.
        with open(classes_path) as f:
            self.classes = json.load(f)

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=len(self.classes)
        )

        state_dict = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def classify(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image).unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():

            outputs = self.model(image)

            probs = torch.softmax(outputs, dim=1)

            confidence, prediction = torch.max(probs, 1)

        return {
            "class": self.classes[prediction.item()],
            "confidence": round(confidence.item(), 4)
        }