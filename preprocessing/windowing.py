import numpy as np
import os

SAVE_DIR = "data/processed/"
WINDOW   = 20
STRIDE   = 5

def create_sequences(X, y, window=20, stride=5):
    Xs, ys = [], []
    total = (len(X) - window) // stride
    print(f"   Total sequences to create: ~{total}")
    
    for count, i in enumerate(range(0, len(X) - window, stride)):
        Xs.append(X[i : i + window])
        labels_in_window = y[i : i + window]
        ys.append(np.bincount(labels_in_window).argmax())
        
        # Print progress every 10000 sequences
        if (count + 1) % 10000 == 0:
            pct = (count + 1) / total * 100
            print(f"   Progress: {count+1}/{total} ({pct:.1f}%)")

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)

def process_dataset(name):
    print(f"\n📂 Processing {name}...")
    X = np.load(f"{SAVE_DIR}X_{name}.npy")
    y = np.load(f"{SAVE_DIR}y_{name}.npy")
    print(f"   Loaded X:{X.shape}  y:{y.shape}")

    # Stratified cap: keep ALL rows of rare classes (below the cap), only
    # downsample classes that have more than PER_CLASS_CAP rows. This
    # preserves the consecutive-row runs that majority-vote windowing
    # needs for rare classes like DoS, instead of uniformly thinning them
    # out along with everything else.
    PER_CLASS_CAP = 50_000
    classes, counts = np.unique(y, return_counts=True)
    keep_idx = []
    for c, cnt in zip(classes, counts):
        c_idx = np.where(y == c)[0]
        if cnt <= PER_CLASS_CAP:
            keep_idx.append(c_idx)  # keep everything — this is what saves DoS
        else:
            chosen = np.random.choice(c_idx, PER_CLASS_CAP, replace=False)
            keep_idx.append(chosen)
    keep_idx = np.sort(np.concatenate(keep_idx))
    X, y = X[keep_idx], y[keep_idx]
    print(f"   Stratified sample → kept {len(X)} rows "
          f"(cap={PER_CLASS_CAP}/class, rare classes kept whole)")
    for c, cnt in zip(*np.unique(y, return_counts=True)):
        print(f"      class {c}: {cnt} rows kept")

    Xs, ys = create_sequences(X, y, window=WINDOW, stride=STRIDE)
    print(f"   Sequences → X:{Xs.shape}  y:{ys.shape}")

    np.save(f"{SAVE_DIR}X_seq_{name}.npy", Xs)
    np.save(f"{SAVE_DIR}y_seq_{name}.npy", ys)
    print(f"   ✅ Saved X_seq_{name}.npy and y_seq_{name}.npy")
    return Xs, ys

if __name__ == "__main__":
    print("🔄 Creating sliding window sequences...")
    print(f"   Window size : {WINDOW}")
    print(f"   Stride      : {STRIDE}")

    Xs_cic,  ys_cic  = process_dataset("cic")
    Xs_edge, ys_edge = process_dataset("edge")

    print(f"\n✅ ALL DONE!")
    print(f"   CICIoT   sequences → {Xs_cic.shape}")
    print(f"   EdgeIIoT sequences → {Xs_edge.shape}")