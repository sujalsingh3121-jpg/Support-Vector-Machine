"""
TASK 3: Support Vector Machine (SVM) - Cat vs Dog Image Classification
--------------------------------------------------------------------------
Classifies images as "cat" or "dog" using an SVM trained on image features
(flattened pixel + HOG-style gradient features).

IMPORTANT NOTE ON DATA:
This environment has no internet access, so it can't download the real
Kaggle "Dogs vs. Cats" dataset. This script instead generates synthetic
grayscale images with class-distinguishing texture patterns (cats get a
smoother radial-gradient-like pattern, dogs get a noisier blotchy pattern)
so the full pipeline — feature extraction, train/test split, SVM training,
evaluation — runs end-to-end and produces real results.

TO USE THE REAL KAGGLE DATASET:
1. Download it from https://www.kaggle.com/c/dogs-vs-cats/data
2. Put cat images in ./data/cat/ and dog images in ./data/dog/
3. Replace `generate_synthetic_images()` below with `load_real_images()`
   (a working loader is included, commented out, using PIL + os.listdir).
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = 32  # 32x32 grayscale


def generate_synthetic_images(n_per_class=250, seed=1):
    """Generate synthetic grayscale 'cat' and 'dog' images with distinct textures."""
    rng = np.random.default_rng(seed)
    images, labels = [], []

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    center = IMG_SIZE / 2

    for _ in range(n_per_class):
        # "cat": smooth radial gradient + light noise
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        base = 255 - (dist / dist.max()) * 180
        img = base + rng.normal(0, 12, (IMG_SIZE, IMG_SIZE))
        images.append(img.clip(0, 255))
        labels.append(0)  # cat = 0

    for _ in range(n_per_class):
        # "dog": blotchy, higher-frequency noise pattern
        img = rng.normal(130, 55, (IMG_SIZE, IMG_SIZE))
        # add a few random blobs
        for _ in range(5):
            bx, by = rng.integers(0, IMG_SIZE, 2)
            r = rng.integers(3, 8)
            mask = (xx - bx) ** 2 + (yy - by) ** 2 <= r ** 2
            img[mask] += rng.normal(60, 15)
        images.append(img.clip(0, 255))
        labels.append(1)  # dog = 1

    return np.array(images), np.array(labels)


# def load_real_images(data_dir="data"):
#     """Loader for the real Kaggle dataset (requires Pillow)."""
#     from PIL import Image
#     import os
#     images, labels = [], []
#     for label_name, label_val in [("cat", 0), ("dog", 1)]:
#         folder = os.path.join(data_dir, label_name)
#         for fname in os.listdir(folder):
#             img = Image.open(os.path.join(folder, fname)).convert("L")
#             img = img.resize((IMG_SIZE, IMG_SIZE))
#             images.append(np.array(img))
#             labels.append(label_val)
#     return np.array(images), np.array(labels)


def extract_features(images):
    """Flatten pixels + add simple gradient-magnitude features (HOG-lite)."""
    n = images.shape[0]
    flat = images.reshape(n, -1)

    gx = np.gradient(images, axis=2)
    gy = np.gradient(images, axis=1)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2).reshape(n, -1)

    return np.hstack([flat, grad_mag])


def main():
    images, labels = generate_synthetic_images()
    print(f"Generated {len(images)} images ({IMG_SIZE}x{IMG_SIZE}), "
          f"{np.sum(labels == 0)} cats / {np.sum(labels == 1)} dogs\n")

    X = extract_features(images)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(X_train_s, y_train)

    y_pred = svm.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["cat", "dog"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    # Plot a few sample predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    test_images = images[len(X_train):][:10] if False else None
    # recover original test images for display
    idx_test = train_test_split(
        np.arange(len(images)), test_size=0.25, random_state=42, stratify=y
    )[1][:10]
    for ax, idx in zip(axes.flat, idx_test):
        ax.imshow(images[idx], cmap="gray")
        pred = svm.predict(scaler.transform(extract_features(images[idx:idx+1])))[0]
        true = labels[idx]
        label_map = {0: "cat", 1: "dog"}
        ax.set_title(f"true:{label_map[true]}\npred:{label_map[pred]}", fontsize=9)
        ax.axis("off")
    plt.suptitle("Task 3: SVM Cat vs Dog Predictions (synthetic sample images)")
    plt.tight_layout()
    plt.savefig("task3_sample_predictions.png", dpi=150)
    print("\nSaved sample predictions to task3_sample_predictions.png")


if __name__ == "__main__":
    main()
