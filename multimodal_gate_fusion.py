import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Concatenate,
    Multiply, Add, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.regularizers import l2

# ==============================
# CONFIG
# ==============================
DATA_DIR = r"C:\Users\bhuvi\OneDrive\cdc_project"
TRAIN_CSV = os.path.join(DATA_DIR, "train_processed.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_processed.csv")

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 8
GATE_REG = 0.005  # L2 regularization on gate to prevent extreme values

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ==============================
# 1. LOAD & PREPARE DATA
# ==============================
print("Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

y = train_df["price_log"].values.astype(np.float32)
y = np.nan_to_num(y, nan=np.nanmean(y))
y_mean, y_std = y.mean(), y.std()

# Spatial features + save center
center_lat, center_long = train_df["lat"].mean(), train_df["long"].mean()
os.makedirs("models", exist_ok=True)
joblib.dump({"center_lat": center_lat, "center_long": center_long}, "models/spatial_center.pkl")
print("✅ Spatial center saved")

def add_spatial(df):
    df = df.copy()
    df["dist_from_center"] = np.sqrt(
        (df["lat"] - center_lat)**2 + (df["long"] - center_long)**2
    )
    return df

train_df = add_spatial(train_df)
test_df  = add_spatial(test_df)

feature_cols = [c for c in train_df.columns if c not in ["price", "price_log", "image_path", "id"]]

# Stratified split
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=SEED,
    stratify=pd.qcut(y, q=10, duplicates="drop")
)

y_train, y_val = y[train_idx], y[val_idx]

# Tabular preprocessing
scaler = StandardScaler()
X_tab_train = scaler.fit_transform(train_df.iloc[train_idx][feature_cols]).astype(np.float32)
X_tab_val   = scaler.transform(train_df.iloc[val_idx][feature_cols]).astype(np.float32)
X_tab_train = np.nan_to_num(np.clip(X_tab_train, -5, 5), nan=0.0)
X_tab_val   = np.nan_to_num(np.clip(X_tab_val, -5, 5), nan=0.0)

train_paths = train_df.iloc[train_idx]["image_path"].values
val_paths   = train_df.iloc[val_idx]["image_path"].values

# ==============================
# 2. TF.DATA PIPELINE
# ==============================
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
])

def process_image(path, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    if augment:
        img = augmentation(img)
    return preprocess_input(img)

def create_dataset(paths, tabular, labels=None, shuffle=False, augment=False):
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices(((paths, tabular), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, tabular))

    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def map_with_label(inputs, label):
        path, tab = inputs
        img = process_image(path, augment)
        return (img, tab), label

    def map_no_label(path, tab):
        img = process_image(path, augment)
        return (img, tab)

    if labels is not None:
        ds = ds.map(map_with_label, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(map_no_label, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = create_dataset(train_paths, X_tab_train, y_train, shuffle=True, augment=True)
val_ds   = create_dataset(val_paths, X_tab_val, y_val, shuffle=False, augment=False)

# ==============================
# 3. GATED MULTIMODAL MODEL
# ==============================
def build_model(tab_dim):
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
    tab_input = Input(shape=(tab_dim,), name="tabular_input")

    # EfficientNet backbone (frozen initially)
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=img_input)
    base_model.trainable = False

    # Image branch
    x_img = GlobalAveragePooling2D()(base_model.output)
    x_img = Dense(512, activation="relu")(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = Dropout(0.5)(x_img)
    x_img = Dense(256, activation="relu")(x_img)
    x_img = Dropout(0.4)(x_img)  # Added dropout for consistency

    # Tabular branch
    x_tab = Dense(256, activation="relu")(tab_input)
    x_tab = BatchNormalization()(x_tab)
    x_tab = Dropout(0.5)(x_tab)
    x_tab = Dense(256, activation="relu")(x_tab)
    x_tab = BatchNormalization()(x_tab)
    x_tab = Dropout(0.4)(x_tab)

    # Gated Fusion
    gate_input = Concatenate()([x_img, x_tab])
    gate = Dense(
        256,
        activation="sigmoid",
        name="modality_gate",
        activity_regularizer=l2(GATE_REG)
    )(gate_input)

    img_gated = Multiply()([x_img, gate])
    tab_gated = Multiply()([x_tab, Lambda(lambda x: 1.0 - x)(gate)])

    fused = Add()([img_gated, tab_gated])

    # Final head
    x = Dense(256, activation="relu")(fused)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)

    output = Dense(
        1,
        activation="linear",
        bias_initializer=tf.keras.initializers.Constant(y_mean)
    )(x)

    model = Model(inputs=[img_input, tab_input], outputs=output)
    return model, base_model

model, base_model = build_model(len(feature_cols))
model.summary()

# ==============================
# 4. TRAINING PHASE 1 & 2
# ==============================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-7, verbose=1)
]

print("\n=== PHASE 1: Training Head + Gate ===")
model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss=Huber(delta=1.0), metrics=["mae"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE1, callbacks=callbacks, verbose=1)

print("\n=== PHASE 2: Fine-tuning EfficientNetB0 (Top 60 layers) ===")
base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False  # Always freeze BN during fine-tuning
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=Huber(delta=1.0), metrics=["mae"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE2, callbacks=callbacks, verbose=1)

# ==============================
# 5. GATE INSPECTION (CORRECT IMPLEMENTATION)
# ==============================
print("\n=== Analyzing Learned Modality Importance ===")
gate_model = Model(inputs=model.inputs, outputs=model.get_layer("modality_gate").output)

# Sample a batch from validation set
(sample_imgs, sample_tabs), _ = next(iter(val_ds))
gate_values = gate_model([sample_imgs, sample_tabs])

mean_gate = gate_values.numpy().mean()
img_contribution = mean_gate
tab_contribution = 1.0 - mean_gate

print(f"Average Gate Value (Image Weight)     : {img_contribution:.4f} → {img_contribution*100:.1f}%")
print(f"Average Tabular Weight                : {tab_contribution:.4f} → {tab_contribution*100:.1f}%")

if img_contribution > 0.55:
    print("🚀 Satellite images are significantly helping predictions!")
elif img_contribution > 0.4:
    print("✅ Images are contributing meaningfully alongside tabular data.")
else:
    print("ℹ️  Tabular features dominate — common in real estate, but images still add value.")

# ==============================
# 6. FINAL METRICS
# ==============================
print("\n=== Final Validation Metrics ===")
log_preds = model.predict(val_ds, verbose=0).ravel()
log_preds = np.clip(log_preds, y_mean - 5*y_std, y_mean + 5*y_std)

rmse_log = np.sqrt(mean_squared_error(y_val, log_preds))
r2_log = r2_score(y_val, log_preds)

price_true = np.expm1(y_val)
price_pred = np.expm1(log_preds)
rmse_price = np.sqrt(mean_squared_error(price_true, price_pred))
r2_price = r2_score(price_true, price_pred)
mape = np.mean(np.abs((price_true - price_pred) / (price_true + 1e-6))) * 100

print("="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Log RMSE     : {rmse_log:.4f}")
print(f"Log R²       : {r2_log:.4f}")
print(f"Price RMSE   : ${rmse_price:,.0f}")
print(f"Price R²     : {r2_price:.4f}")
print(f"Price MAPE   : {mape:.2f}%")
print("="*60)

# ==============================
# 7. SAVE ARTIFACTS
# ==============================
model.save("models/multimodal_gated_final.keras")
joblib.dump(scaler, "models/tabular_scaler.pkl")

print("\n✅ Training complete!")
print("   • Model saved: models/multimodal_gated_final.keras")
print("   • Scaler saved: models/tabular_scaler.pkl")
print("   • Spatial center saved: models/spatial_center.pkl")