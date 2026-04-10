import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from .label_info import CLASS_DETAILS, DEFAULT_CLASS_ORDER, describe_class
except ImportError:
    from label_info import CLASS_DETAILS, DEFAULT_CLASS_ORDER, describe_class

RANDOM_SEED = 42


def get_default_paths():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    csv_path = data_dir / "HAM10000_metadata.csv"
    return csv_path, data_dir


def build_model(num_classes, input_shape=(224, 224, 3)):
    backbone = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape, pooling="avg")
    backbone.trainable = True

    x = backbone.output
    x = Dropout(0.35)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.25)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=backbone.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
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


def create_data_generators(train_df, val_df, target_size=(224, 224), batch_size=32):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    flow_args = {
        "x_col": "filepath",
        "y_col": "label",
        "target_size": target_size,
        "class_mode": "categorical",
        "batch_size": batch_size,
        "classes": DEFAULT_CLASS_ORDER,
    }

    train_flow = train_gen.flow_from_dataframe(
        train_df,
        shuffle=True,
        **flow_args,
    )
    val_flow = val_gen.flow_from_dataframe(
        val_df,
        shuffle=False,
        **flow_args,
    )
    return train_flow, val_flow


def evaluate_model(model, val_flow, class_order):
    probabilities = model.predict(val_flow, verbose=1)
    predicted_indices = np.argmax(probabilities, axis=1)
    true_indices = val_flow.classes
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
        "history": {key: [round(float(value), 6) for value in values] for key, values in history.history.items()},
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

    train_flow, val_flow = create_data_generators(train_df, val_df, batch_size=batch_size)
    class_order = list(train_flow.class_indices.keys())
    print("Training classes:", class_order)

    model = build_model(len(class_order))
    callbacks = [
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=epochs,
        callbacks=callbacks,
    )

    print("Saving final model to:", model_path)
    model.save(model_path)

    save_class_metadata(class_order, class_names_path)
    print("Saved class metadata to:", class_names_path)

    metrics = evaluate_model(model, val_flow, class_order)
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
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    args = parse_args()
    default_csv, default_image_dir = get_default_paths()
    base_dir = Path(__file__).resolve().parent

    csv_path = Path(args.csv_path) if args.csv_path else default_csv
    image_dir = Path(args.image_dir) if args.image_dir else default_image_dir
    model_path = Path(args.model_path) if args.model_path else base_dir / "skin_model.h5"
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
