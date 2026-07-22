import json
import torch
import timm

from torchvision import transforms
from PIL import Image


class DamageClassifier:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=7
        )

        state_dict = torch.load(
            "models/classifier/best_model_v2.pth",
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        with open("models/classifier/classes.json") as f:
            self.classes = json.load(f)

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