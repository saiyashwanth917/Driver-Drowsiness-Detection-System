import tensorflow as tf

from scripts.nthu_data_loader import load_fold_datasets


print("=" * 65)
print("NTHU DATA LOADER TEST")
print("=" * 65)

(
    train_ds,
    validation_ds,
    test_ds,
    metadata,
) = load_fold_datasets(
    fold_number=1,
    augment=False,
)

print("\nFold metadata:")
for key, value in metadata.items():
    print(f"{key}: {value}")

# Inspect one batch from each dataset.
for split_name, dataset in [
    ("Train", train_ds),
    ("Validation", validation_ds),
    ("Test", test_ds),
]:
    images, labels = next(iter(dataset))

    print(f"\n{split_name} batch")
    print("-" * 40)
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Min pixel:", float(tf.reduce_min(images)))
    print("Max pixel:", float(tf.reduce_max(images)))
    print("Labels:", labels.numpy()[:10])

    assert images.shape[1:] == (160, 160, 3)
    assert images.dtype == tf.float32
    assert float(tf.reduce_min(images)) >= 0.0
    assert float(tf.reduce_max(images)) <= 1.0

print("\nDATA LOADER TEST PASSED")