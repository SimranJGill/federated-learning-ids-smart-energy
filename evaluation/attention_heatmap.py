import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("saved_models/fl_model.h5")
X_test = np.load("saved_models/X_test.npy")
y_test = np.load("saved_models/y_test.npy")
class_names = np.load("saved_models/class_names.npy", allow_pickle=True)

# Uses the explicit name added in model/architecture.py
# (layers.Softmax(axis=1, name="attention_weights")).
# If you haven't updated architecture.py yet, this will raise a clear error
# instead of silently grabbing the wrong layer.
try:
    attn_layer = model.get_layer("attention_weights")
except ValueError:
    print("Layer 'attention_weights' not found — did you update model/architecture.py?")
    print("Falling back to searching for the first Softmax layer instead.")
    attn_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Softmax):
            attn_layer = layer
            break
    if attn_layer is None:
        raise ValueError("No Softmax layer found in the model at all.")

attention_model = tf.keras.Model(inputs=model.input, outputs=attn_layer.output)

os.makedirs("saved_models/attention_heatmaps", exist_ok=True)

for class_idx, class_name in enumerate(class_names):
    sample_idx = np.where(y_test == class_idx)[0]
    if len(sample_idx) == 0:
        print(f"No test samples for class {class_name}, skipping.")
        continue

    X_sample = X_test[sample_idx[:1]]  # shape (1, 20, 46)
    attn_weights = attention_model.predict(X_sample, verbose=0)  # shape (1, 20, 1)

    plt.figure(figsize=(8, 1.5))
    plt.imshow(attn_weights.reshape(1, -1), aspect="auto", cmap="viridis")
    plt.title(f"Attention heatmap — {class_name}")
    plt.xlabel("Time step (0-19)")
    plt.yticks([])
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(f"saved_models/attention_heatmaps/{class_name}.png", dpi=150)
    plt.close()
    print(f"Saved heatmap for {class_name}")

print("\nDone. Heatmaps saved to saved_models/attention_heatmaps/")
