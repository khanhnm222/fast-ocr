"""
Train CNN on MNIST (digits 0-9) or EMNIST byclass (alphanumeric, 62 classes).
Saves model + label map to saved_models/.

Usage:
    python train.py                          # MNIST digits only
    python train.py --dataset emnist         # EMNIST (digits + A-Z + a-z)
    python train.py --epochs 20 --batch-size 128
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model.cnn_model import build_model

# Label map for each dataset
MNIST_LABELS = [str(i) for i in range(10)]

# EMNIST byclass: 0-9 digits, 10-35 uppercase A-Z, 36-61 lowercase a-z
EMNIST_LABELS = (
    [str(i) for i in range(10)]
    + [chr(ord('A') + i) for i in range(26)]
    + [chr(ord('a') + i) for i in range(26)]
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_mnist() -> tuple:
    print("Downloading / loading MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)


def load_emnist() -> tuple:
    """
    Load EMNIST 'byclass' split (697 932 train, 116 323 test, 62 classes).
    The emnist package downloads data on the first call (~500 MB).
    Images are stored transposed vs. MNIST — fixed with a flip+transpose.
    """
    try:
        from emnist import extract_test_samples, extract_training_samples
    except ImportError:
        raise ImportError("Run `pip install emnist` to use the EMNIST dataset.")

    print("Downloading / loading EMNIST byclass (first run may take a few minutes)...")
    x_train, y_train = extract_training_samples('byclass')
    x_test, y_test = extract_test_samples('byclass')

    # Fix orientation: EMNIST images are stored with axes swapped vs. MNIST
    x_train = np.array([np.fliplr(img.T) for img in x_train])
    x_test = np.array([np.fliplr(img.T) for img in x_test])

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_augmentation_pipeline() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])


def train(dataset: str = 'mnist', epochs: int = 15, batch_size: int = 64):
    if dataset == 'emnist':
        (x_train, y_train), (x_test, y_test) = load_emnist()
        labels = EMNIST_LABELS
        num_classes = 62
        model_path = os.path.join('saved_models', 'emnist_cnn.keras')
        labels_path = os.path.join('saved_models', 'emnist_labels.json')
    else:
        (x_train, y_train), (x_test, y_test) = load_mnist()
        labels = MNIST_LABELS
        num_classes = 10
        model_path = os.path.join('saved_models', 'mnist_cnn.keras')
        labels_path = os.path.join('saved_models', 'mnist_labels.json')

    print(f"Dataset: {dataset.upper()} | Classes: {num_classes} | "
          f"Train: {len(x_train):,} | Test: {len(x_test):,}")

    model = build_model(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    augment = build_augmentation_pipeline()

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(min(len(x_train), 50_000))
        .batch(batch_size)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]

    print(f"\nTraining for up to {epochs} epochs...")
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")

    os.makedirs('saved_models', exist_ok=True)
    model.save(model_path)
    print(f"Model saved  -> {model_path}")

    with open(labels_path, 'w') as f:
        json.dump(labels, f)
    print(f"Labels saved -> {labels_path}")

    _plot_history(history, dataset)
    return model


def _plot_history(history, dataset: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{dataset.upper()} training history')
    axes[0].plot(history.history['accuracy'], label='train')
    axes[0].plot(history.history['val_accuracy'], label='val')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='train')
    axes[1].plot(history.history['val_loss'], label='val')
    axes[1].set_title('Loss')
    axes[1].legend()
    plt.tight_layout()
    out = f'saved_models/{dataset}_training_history.png'
    plt.savefig(out)
    print(f"Training history plot -> {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=['mnist', 'emnist'], default='mnist',
        help='mnist = digits only (fast). emnist = digits + A-Z + a-z (~500 MB download).',
    )
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    train(dataset=args.dataset, epochs=args.epochs, batch_size=args.batch_size)
 