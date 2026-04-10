import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

try:
    from .label_info import CLASS_DETAILS, DEFAULT_CLASS_ORDER, describe_class
except ImportError:
    from label_info import CLASS_DETAILS, DEFAULT_CLASS_ORDER, describe_class

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, class_to_index, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["filepath"]).convert("RGB")
        tensor = self.transform(image)
        label = self.class_to_index[row["label"]]
        return tensor, label


def get_default_paths():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    csv_path = data_dir / "HAM10000_metadata.csv"
    return csv_path, data_dir


def build_model(num_classes, use_pretrained=True):
    weights = MobileNet_V2_Weights.DEFAULT if use_pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.to(DEVICE)
    return model


def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns:
        raise ValueError("Expected CSV to contain an image_id column.")
    if "dx" not in df.columns and "label" not in df.columns:
        raise ValueError("Expected CSV to contain a dx or label column.")

    label_col = "dx" if "dx" in df.columns else "label"
    df["filename"] = df["image_id"].astype(str) + ".jpg"
    df["label"] = df[label_col].astype(str)
    return df


def build_image_lookup(image_dir):
    jpg_paths = list(image_dir.rglob("*.jpg"))
    if not jpg_paths:
        raise FileNotFoundError(
            f"No JPG images were found inside {image_dir}. "
            "Expected HAM10000 images under folders like HAM10000_images_part_1 and HAM10000_images_part_2."
        )
    return {path.name: path.resolve() for path in jpg_paths}


def attach_image_paths(df, image_dir):
    image_lookup = build_image_lookup(image_dir)
    df["filepath"] = df["filename"].map(lambda name: str(image_lookup.get(name, "")))
    missing = df[df["filepath"] == ""]
    if not missing.empty:
        samples = missing["filename"].head(5).tolist()
        raise FileNotFoundError(
            "Some metadata entries do not have matching image files.\n"
            f"Examples: {samples}"
        )
    return df


def stratified_split(df, train_ratio=0.8):
    train_parts = []
    val_parts = []
    for _, group in df.groupby("label"):
        shuffled = group.sample(frac=1, random_state=RANDOM_SEED)
        split_index = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
        train_parts.append(shuffled.iloc[:split_index])
        val_parts.append(shuffled.iloc[split_index:])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return train_df, val_df


def create_data_loaders(train_df, val_df, batch_size=32):
    class_order = list(DEFAULT_CLASS_ORDER)
    class_to_index = {label: index for index, label in enumerate(class_order)}
    train_dataset = HAM10000Dataset(train_df, class_to_index, TRAIN_TRANSFORM)
    val_dataset = HAM10000Dataset(val_df, class_to_index, VAL_TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, class_order


def run_epoch(model, data_loader, criterion, epoch_number, total_epochs, phase, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    all_targets = []
    all_predictions = []

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch_number}/{total_epochs} [{phase}]",
        leave=False,
        dynamic_ncols=True,
    )

    for inputs, labels in progress_bar:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(torch.argmax(outputs, dim=1).detach().cpu().tolist())

        running_loss = total_loss / max(len(all_targets), 1)
        running_accuracy = accuracy_score(all_targets, all_predictions) if all_targets else 0.0
        progress_bar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_accuracy:.4f}")

    average_loss = total_loss / max(len(data_loader.dataset), 1)
    accuracy = accuracy_score(all_targets, all_predictions) if all_targets else 0.0
    return average_loss, accuracy, all_targets, all_predictions


def evaluate_model(model, val_loader, class_order):
    criterion = nn.CrossEntropyLoss()
    _, _, true_indices, predicted_indices = run_epoch(
        model,
        val_loader,
        criterion,
        epoch_number="final",
        total_epochs="eval",
        phase="evaluation",
        optimizer=None,
    )
    predicted_labels = [class_order[index] for index in predicted_indices]
    true_labels = [class_order[index] for index in true_indices]

    metrics = {
        "accuracy": round(float(accuracy_score(true_labels, predicted_labels)), 6),
        "precision_weighted": round(
            float(precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)), 6
        ),
        "recall_weighted": round(
            float(recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)), 6
        ),
        "f1_weighted": round(float(f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)), 6),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=class_order,
            target_names=[describe_class(code)["name"] for code in class_order],
            zero_division=0,
            output_dict=True,
        ),
    }
    return metrics


def save_class_metadata(class_order, output_path):
    payload = {
        "class_order": class_order,
        "classes": [CLASS_DETAILS.get(code, describe_class(code)) for code in class_order],
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_metrics(history, metrics, output_path):
    payload = {
        "history": {key: [round(float(value), 6) for value in values] for key, values in history.items()},
        "metrics": metrics,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def train(csv_path, image_dir, model_path, class_names_path, metrics_path, epochs=8, batch_size=32):
    print("Loading metadata from:", csv_path)
    metadata = load_metadata(csv_path)
    metadata = attach_image_paths(metadata, image_dir)
    metadata = metadata[metadata["label"].isin(DEFAULT_CLASS_ORDER)].copy()
    print("Found metadata for", len(metadata), "images.")
    print("Class distribution:")
    print(metadata["label"].value_counts().to_string())

    train_df, val_df = stratified_split(metadata)
    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))

    train_loader, val_loader, class_order = create_data_loaders(train_df, val_df, batch_size=batch_size)
    print("Training classes:", class_order)
    print("Training device:", DEVICE)

    model = build_model(len(class_order))
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_accuracy = -1.0
    best_state_dict = None
    stagnant_epochs = 0
    early_stopping_patience = 4

    for epoch in range(epochs):
        train_loss, train_accuracy, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            epoch_number=epoch + 1,
            total_epochs=epochs,
            phase="train",
            optimizer=optimizer,
        )
        val_loss, val_accuracy, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            epoch_number=epoch + 1,
            total_epochs=epochs,
            phase="val",
            optimizer=None,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- train_accuracy: {train_accuracy:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_order": class_order,
        "device_used_for_training": str(DEVICE),
    }

    print("Saving final model to:", model_path)
    torch.save(checkpoint, model_path)

    save_class_metadata(class_order, class_names_path)
    print("Saved class metadata to:", class_names_path)

    metrics = evaluate_model(model, val_loader, class_order)
    save_metrics(history, metrics, metrics_path)
    print("Saved metrics to:", metrics_path)
    print(json.dumps(metrics, indent=2))

    return model, history, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a HAM10000 skin lesion classification model.")
    parser.add_argument("--csv-path", default=None, help="Path to the HAM10000 metadata CSV file.")
    parser.add_argument("--image-dir", default=None, help="Root directory containing HAM10000 images.")
    parser.add_argument("--model-path", default=None, help="Output path for the trained model file.")
    parser.add_argument("--classes-path", default=None, help="Output path for the class metadata JSON file.")
    parser.add_argument("--metrics-path", default=None, help="Output path for the training metrics JSON file.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    return parser.parse_args()


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    args = parse_args()
    default_csv, default_image_dir = get_default_paths()
    base_dir = Path(__file__).resolve().parent

    csv_path = Path(args.csv_path) if args.csv_path else default_csv
    image_dir = Path(args.image_dir) if args.image_dir else default_image_dir
    model_path = Path(args.model_path) if args.model_path else base_dir / "skin_model.pth"
    class_names_path = Path(args.classes_path) if args.classes_path else base_dir / "class_names.json"
    metrics_path = Path(args.metrics_path) if args.metrics_path else base_dir / "training_metrics.json"

    if not csv_path.exists():
        print("Missing CSV metadata file:", csv_path)
        sys.exit(1)

    if not image_dir.is_dir():
        print("Missing image directory:", image_dir)
        sys.exit(1)

    train(
        csv_path=csv_path,
        image_dir=image_dir,
        model_path=model_path,
        class_names_path=class_names_path,
        metrics_path=metrics_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
