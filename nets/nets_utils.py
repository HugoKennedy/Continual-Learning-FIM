from typing import Literal
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ResNetForImageClassification

from tqdm import tqdm


def train(model, train_loader, val_loader, optimizer, device, num_epochs=10, save_path=None):
    """

    params：
    - model
    - train_loader
    - val_loader
    - optimizer
    - device: ('cuda' or 'cpu')
    - num_epochs

    return：
    - loss_history
    """

    #model.to(device)
    criterion = nn.CrossEntropyLoss()
    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)

        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        loss_history["train"].append(train_loss)
        acc_history["train"].append(train_acc)

        model.eval()

        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        loss_history["val"].append(val_loss)
        acc_history["val"].append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    print("complete！")
    return loss_history, acc_history


def evaluate(model, data_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def get_net(
        name: str,
        weights_path: str = None,
        num_classes: int = 10,
        gray_scale: bool = False,
        device="cuda:0",
        forWhat : Literal["transferability", "finetuning", "linear_probing", "chains_of_transfer"] = "finetuning"
    ) -> nn.Module:
    """
    Returns a neural network model based on the specified name.

    Args:
        name (str): The name of the model architecture.
        weights_path (str, optional): Path to pretrained weights to load. Defaults to None.
        num_classes (int, optional): Number of output classes. Defaults to 10.
        forWhat (str, optional): Purpose of the model, either "transferability" or "finetuning". Defaults to "finetuning".

    Returns:
        nn.Module: The instantiated neural network model.
    """
    match name:
        case "simplecnn":
            # SimpleCNN
            model = SimpleCNN(
                num_channels=1 if gray_scale else 3,
                num_classes=num_classes
            ).to(device)
            return model

        # HuggingFace Transformers models
        case "microsoft_resnet50":
            # ResNet-50
            print("[get_net] Creating Microsoft ResNet-50 model...")
            print("[get_net] weights_path:", weights_path)

            if forWhat in ["finetuning", "linear_probing"]:
                print(f"[get_net] Using pretrained weights for {forWhat}...")
                weights_path = "microsoft/resnet-50"

            if forWhat == "chains_of_transfer" and weights_path is None:
                # TODO use random initialization when training chains of transfer from scratch
                weights_path = "microsoft/resnet-50"

            processor = AutoImageProcessor.from_pretrained(
                "microsoft/resnet-50"
            )
            processor.do_rescale = False  # avoid double rescaling, as Avalanche dataloaders already do it
            # ask processor not to normalize since it expects 3 channels
            # TODO instead, allow Avalanche SplitMNIST benchmark to handle normalization
            #processor.do_normalize = False
            # if gray_scale and forWhat in ["finetuning", "linear_probing", "chains_of_transfer"]:  # BUG why verify forWhat?
            if gray_scale:
                processor.image_mean = [0.1307]  # for MNIST  TODO for other gray scale datasets
                processor.image_std = [0.3081]  # for MNIST  TODO for other gray scale datasets


            '''
            model = ResNetForImageClassification.from_pretrained(
                weights_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            ).to(device)
            '''
            model = ResNetWithProcessorForImageClassification.from_pretrained(
                weights_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                processor=processor,
                num_channels=1 if gray_scale else 3
            ).to(device)

            #print(model.processor)
            #print("#### model.config", model.config)

            # NOTE by defaults, all layers trainable by default

            if forWhat == "linear_probing":
                # freeze all layers except classifier and conv1 (adapted for gray scale if needed)
                print("[get_net] Freezing all layers except classifier and conv1 for linear probing...")
                for param in model.resnet.embedder.embedder.convolution.parameters():
                    param.requires_grad = True  # make sure conv1 parameters are trainable
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.classifier.parameters():
                    param.requires_grad = True

        case _:
            raise ValueError(f"Model {name} not recognized.")

    return model


class ResNetWithProcessorForImageClassification(ResNetForImageClassification):
    #BUG without the suffix ForImageClassification, Transformers library throws an error
    # because it uses this suffix to identify the correct loss function to use ...
    def __init__(self, config, processor=None):
        super().__init__(config)
        self.processor = processor

    def forward(self, pixel_values=None, labels=None):
        if self.processor is not None and pixel_values is not None:
            inputs = self.processor(
                images=pixel_values,
                return_tensors="pt"
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return super().forward(**inputs, labels=labels)


class SimpleNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, num_classes)
        
        # Initialize weights with reproducible seed
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    #def forward(self, x):
    def forward(self, pixel_values=None, labels=None, **kwargs):
        # Handle both positional and keyword argument input
        if pixel_values is None and len(kwargs) == 0:
            raise ValueError("pixel_values must be provided")

        x = pixel_values.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)

        # Transformers Trainer expects a scalar loss when labels are provided
        if labels is not None:
            target = labels
            # Normalize target shape/dtype for CE
            if target.dim() == 2 and target.size(-1) == 1:
                target = target.squeeze(-1)
            elif target.dim() == 2 and target.size(-1) == logits.size(-1):
                target = target.argmax(dim=-1)
            target = target.long()

            loss = torch.nn.functional.cross_entropy(logits, target)
            return {"loss": loss, "logits": logits}

        return logits


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv2d(num_channels, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)

        # Classifier is built lazily based on input spatial dims (supports CIFAR 32x32 and MNIST 28x28)
        self.fc1 = None
        self.fc2 = None

        # Initialize convolutional backbone weights
        self._init_conv_weights()

    def _init_conv_weights(self):
        """Initialize Conv2d weights with Kaiming (He) init."""
        for module in [self.conv1, self.conv2]:
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _init_classifier_weights(self):
        """Initialize Linear layers with Xavier init."""
        for module in [self.fc1, self.fc2]:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _build_classifier(self, in_features, device):
        """Create classifier layers once we know the flattened feature size."""
        self.fc1 = torch.nn.Linear(in_features, 128, device=device)
        self.fc2 = torch.nn.Linear(128, self.num_classes, device=device)
        self._init_classifier_weights()

    #def forward(self, x):
    def forward(self, pixel_values=None, labels=None, **kwargs):
        # Handle both positional and keyword argument input
        if pixel_values is None and len(kwargs) == 0:
            raise ValueError("pixel_values must be provided")

        x = pixel_values
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)

        # If inputs are larger (e.g., dSprites 64x64), shrink spatial dims to keep head size reasonable
        if x.size(-1) > 20 or x.size(-2) > 20:
            x = torch.nn.functional.adaptive_max_pool2d(x, output_size=14)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        # Build classifier on first forward pass to adapt to input resolution
        if self.fc1 is None or self.fc2 is None:
            self._build_classifier(x.size(1), device=x.device)

        features = self.fc1(x)
        if return_features := kwargs.get("return_features", False):
            return features  # allow extracting features only if needed

        x = torch.nn.functional.relu(features)
        x = self.dropout2(x)
        logits = self.fc2(x)

        # Transformers Trainer expects a scalar loss when labels are provided
        if labels is not None:
            target = labels
            # Normalize target shape/dtype for CE
            if target.dim() == 2 and target.size(-1) == 1:
                target = target.squeeze(-1)
            elif target.dim() == 2 and target.size(-1) == logits.size(-1):
                target = target.argmax(dim=-1)
            target = target.long()

            loss = torch.nn.functional.cross_entropy(logits, target)
            #return {"loss": loss, "logits": logits}
            return {"loss": loss, "logits": logits}

        return logits
