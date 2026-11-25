"""
MobileNetV2 training pipeline for Hybrid Driver Drowsiness Detection.
Uses the existing NTHU LOSO loader and preserves subject-level splits.

Run from project root:
    python -m training.train_mobilenetv2 --fold 1
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import nthu_baseline as cfg
from scripts.nthu_data_loader import load_fold_datasets


def mobilenet_preprocess(image, label):
    # Existing loader outputs [0, 1]. MobileNetV2 preprocess_input expects [0, 255].
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
    return image, label


def prepare_for_mobilenet(dataset):
    return dataset.map(mobilenet_preprocess, num_parallel_calls=2).prefetch(1)


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], cfg.CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], cfg.CHANNELS), name="image")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(cfg.DROPOUT_RATE, name="dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(cfg.DROPOUT_RATE, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="drowsy_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_drowsiness")
    return model, base


def make_callbacks(output_dir, checkpoint_name, append_log=False):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / checkpoint_name),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=cfg.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv"), append=append_log),
    ]


def save_history(history_list, output_dir):
    merged = {}
    for history in history_list:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend([float(v) for v in values])
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    # Plot history if matplotlib is installed.
    try:
        import matplotlib.pyplot as plt

        if "loss" in merged:
            plt.figure()
            plt.plot(merged["loss"], label="Training loss")
            if "val_loss" in merged:
                plt.plot(merged["val_loss"], label="Validation loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("MobileNetV2 Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "loss_curve.png", dpi=160)
            plt.close()

        if "auc" in merged:
            plt.figure()
            plt.plot(merged["auc"], label="Training AUC")
            if "val_auc" in merged:
                plt.plot(merged["val_auc"], label="Validation AUC")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.title("MobileNetV2 AUC")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "auc_curve.png", dpi=160)
            plt.close()
    except ImportError:
        print("matplotlib not installed; skipping training plots.")


def evaluate_and_save(model, test_ds, output_dir):
    results = model.evaluate(test_ds, return_dict=True, verbose=1)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

    y_true = np.concatenate([labels.numpy().astype(int) for _, labels in test_ds], axis=0)
    y_prob = model.predict(test_ds, verbose=1).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

        report = classification_report(
            y_true, y_pred, target_names=["Not Drowsy", "Drowsy"],
            output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(y_true, y_pred).tolist()
        metrics = {
            "classification_report": report,
            "confusion_matrix": matrix,
            "threshold": 0.5,
        }
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc_from_predictions"] = float(roc_auc_score(y_true, y_prob))
        with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(matrix)
        plt.title("Test Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0, 1], ["Not Drowsy", "Drowsy"])
        plt.yticks([0, 1], ["Not Drowsy", "Drowsy"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, matrix[i][j], ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
        plt.close()

        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Test ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "roc_curve.png", dpi=160)
            plt.close()
        print("\nClassification report saved.")
    except ImportError:
        print("scikit-learn/matplotlib unavailable; raw model evaluation saved.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for NTHU drowsiness classification.")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional LOSO manifest path.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional experiment output root.")
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(cfg.RANDOM_SEED)
    output_root = (
        args.output_dir if args.output_dir is not None
        else Path(cfg.OUTPUT_ROOT)
    )
    output_dir = output_root / f"fold_{args.fold}" / "mobilenetv2"

    # Protect existing experiment results from accidental overwriting.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {output_dir}\n"
            "Choose a new --output-dir to protect existing checkpoints."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MOBILENETV2 DRIVER DROWSINESS TRAINING")
    print("=" * 72)
    print(f"Fold: {args.fold}")
    print(f"Output: {output_dir}")

    train_ds, val_ds, test_ds, metadata = load_fold_datasets(
        fold_number=args.fold,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        augment=True,
    )
    print("\nFold metadata:")
    print(json.dumps(metadata, indent=2))

    train_ds = prepare_for_mobilenet(train_ds)
    val_ds = prepare_for_mobilenet(val_ds)
    test_ds = prepare_for_mobilenet(test_ds)

    class_weights = {int(k): float(v) for k, v in metadata["class_weights"].items()}
    with (output_dir / "fold_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (output_dir / "class_weights.json").open("w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)

    model, base = build_model()
    callbacks = make_callbacks(output_dir, "phase1_best.keras", append_log=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()

    print("\nPHASE 1: Train classification head (backbone frozen)")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nPHASE 2: Fine-tune upper MobileNetV2 layers")
    base.trainable = True
    cutoff = max(0, len(base.layers) - args.fine_tune_layers)
    for layer in base.layers[:cutoff]:
        layer.trainable = False
    # Keep BatchNorm frozen for stable fine-tuning on a small subject split.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE * 0.01),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(output_dir, "phase2_best.keras", append_log=True),
        verbose=1,
    )

    save_history([history_head, history_finetune], output_dir)

    # Compare both phase checkpoints on validation data; choose the better checkpoint.
    phase_candidates = []
    for checkpoint_name in ["phase1_best.keras", "phase2_best.keras"]:
        checkpoint_path = output_dir / checkpoint_name
        if checkpoint_path.exists():
            candidate = tf.keras.models.load_model(checkpoint_path)
            candidate_metrics = candidate.evaluate(val_ds, return_dict=True, verbose=0)
            phase_candidates.append((float(candidate_metrics.get("auc", 0.0)), candidate, checkpoint_name))

    if phase_candidates:
        phase_candidates.sort(key=lambda item: item[0], reverse=True)
        best_auc, best_model, best_phase = phase_candidates[0]
        best_path = output_dir / "best_model.keras"
        best_model.save(best_path)
        with (output_dir / "selected_checkpoint.json").open("w", encoding="utf-8") as f:
            json.dump({"checkpoint": best_phase, "validation_auc": best_auc}, f, indent=2)
    else:
        best_model = model
        best_path = output_dir / "best_model.keras"
        best_model.save(best_path)

    print("\nFINAL TEST EVALUATION (held-out subject)")
    evaluate_and_save(best_model, test_ds, output_dir)

    print("\nTRAINING COMPLETE")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
