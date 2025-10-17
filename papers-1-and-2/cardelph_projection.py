import json
import os
import torch
import open_clip  # Using open_clip for LAION models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap
import umap
from sklearn.decomposition import PCA

# --- Configuration ---
USE_COCO = True
USE_LAION = False  # New flag to visualize LAION sample
model_name = "ViT-L-14"
pretrained_name = "laion2b_s32b_b82k"  # The specific checkpoint from Hugging Face
DEBUG = False  # Set to True to visualize a few samples
method = "pca-norm"  # "pca" or "umap" or "pca-norm" or "pca-r2"

if USE_LAION:
    # We only visualize text embeddings for the LAION sample
    CACHE_FILE = f"clip_embeddings_cache_laion_{model_name.replace('/', '-')}.npz"
elif USE_COCO:
    CACHE_FILE = f"clip_embeddings_cache_coco_{model_name.replace('/', '-')}.npz"
else:
    CACHE_FILE = f"clip_embeddings_cache_{model_name.replace('/', '-')}.npz"


# Load OpenCLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained_name, device=device
)
tokenizer = open_clip.get_tokenizer(model_name)

# --- Load dataset ---
if USE_LAION:
    # A small hardcoded sample to represent the diverse LAION dataset
    data = []
    texts = [
        "A photo of a cat sitting on a keyboard",
        "A majestic mountain range at sunset, painted by a professional artist",
        "An abstract sculpture made of glass and metal",
        "A plate of sushi with chopsticks",
        "A wide shot of a bustling city street at night with neon lights",
        "A close-up of a human eye with detailed iris",
        "A watercolor painting of a sunflower in a field",
        "A digital art piece of a futuristic spaceship",
        "A dog wearing sunglasses and a funny hat",
    ]
    for i, text in enumerate(texts):
        data.append(
            {
                "image": None,  # No images to load for this sample
                "text": text,
                "label": f"laion_sample_{i}",
            }
        )
    DATASET_DIR = None  # Not used for this option
elif USE_COCO:
    DATASET_DIR = "coco/images/train2017"
    ANNO_FILE = "coco/annotations/captions_train2017.json"
    with open(ANNO_FILE, "r") as f:
        coco_data = json.load(f)
    image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}
    data = []
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
        if image_id in image_id_to_file:
            img_file = image_id_to_file[image_id]
            if os.path.exists(os.path.join(DATASET_DIR, img_file)):
                data.append(
                    {"image": img_file, "text": caption, "label": str(image_id)}
                )
    # display a few image caption pairs using matplotlib
    if DEBUG == True:
        for item in data[:5]:
            img = Image.open(os.path.join(DATASET_DIR, item["image"]))
            plt.imshow(img)
            plt.title(item["text"])
            plt.axis("off")
            plt.show()

else:
    DATASET_DIR = "ImageNet-AO-filtered"
    data = []
    for label in os.listdir(DATASET_DIR):
        label_dir = os.path.join(DATASET_DIR, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                if img_file.endswith((".png", ".jpg", ".jpeg")):
                    data.append(
                        {
                            "image": os.path.join(label, img_file),
                            "text": label.replace("_", " "),
                            "label": label,
                        }
                    )

df = pd.DataFrame(data)

# --- Load from cache if available ---
if os.path.exists(CACHE_FILE):
    cache = np.load(CACHE_FILE, allow_pickle=True)
    embeddings = cache["embeddings"]
    labels = cache["labels"].tolist()
    types = cache["types"].tolist()
    print(f"Loaded cached embeddings from {CACHE_FILE}")
else:
    print("Computing embeddings (no cache found)...")
    image_embeddings, text_embeddings, labels = [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx >= 200:
            break
        text, label = row["text"], row["label"]
        img_path = row["image"]

        # Only compute image embeddings if an image path exists
        if img_path:
            image = (
                preprocess(Image.open(os.path.join(DATASET_DIR, img_path)))
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                img_emb = model.encode_image(image).cpu().numpy().squeeze()
            image_embeddings.append(img_emb)

        # Text embedding
        text_token = tokenizer([text]).to(device)
        with torch.no_grad():
            txt_emb = model.encode_text(text_token).cpu().numpy().squeeze()
        text_embeddings.append(txt_emb)
        labels.append(label)

    embeddings = np.array(image_embeddings + text_embeddings)

    # Adjust labels and types based on what was computed
    if image_embeddings:
        labels = labels[: len(image_embeddings)] + labels
        types = ["image"] * len(image_embeddings) + ["text"] * len(text_embeddings)
    else:
        # For the LAION sample, there are only text embeddings
        labels = labels
        types = ["text"] * len(text_embeddings)

    np.savez(CACHE_FILE, embeddings=embeddings, labels=labels, types=types)
    print(f"Cached embeddings saved to {CACHE_FILE}")

if method == "pca":
    reducer = PCA(n_components=3)
    coords_3d = reducer.fit_transform(embeddings)

    # --- 3D Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Map labels to colors
    unique_labels = sorted(set(labels))
    cmap = get_cmap("tab20", len(unique_labels))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}
    # Scatter embeddings
    for i, (emb, lab, typ) in enumerate(zip(coords_3d, labels, types)):
        color = label_to_color[lab]
        marker = "o" if typ == "image" else "^"
        ax.scatter(
            emb[0],
            emb[1],
            emb[2],
            color=color,
            marker=marker,
            s=40,
            alpha=0.8,
            label=f"{lab} ({typ})",
        )

    # Draw lines between image-text pairs, only for datasets that have them
    if "image" in types and "text" in types:
        n = len(embeddings) // 2
        for i in range(n):
            img_coord = coords_3d[i]
            text_coord = coords_3d[i + n]
            ax.plot(
                [img_coord[0], text_coord[0]],
                [img_coord[1], text_coord[1]],
                [img_coord[2], text_coord[2]],
                color="gray",
                alpha=0.3,
                linewidth=0.7,
            )

    ax.set_title(f"CLIP ({model_name}) Embeddings in 3D using PCA")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

    plt.tight_layout()
    plt.show()

elif method == "pca-norm":
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    reducer = PCA(n_components=3)
    coords_3d = reducer.fit_transform(embeddings)

    # normalize to unit sphere
    norms = np.linalg.norm(coords_3d, axis=1, keepdims=True)
    coords_3d = coords_3d / norms

    # --- 3D Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Map labels to colors
    unique_labels = sorted(set(labels))
    cmap = get_cmap("tab20", len(unique_labels))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}
    # Scatter embeddings
    for i, (emb, lab, typ) in enumerate(zip(coords_3d, labels, types)):
        color = label_to_color[lab]
        marker = "o" if typ == "image" else "^"
        ax.scatter(
            emb[0],
            emb[1],
            emb[2],
            color=color,
            marker=marker,
            s=40,
            alpha=0.8,
            label=f"{lab} ({typ})",
        )

    # Draw lines between image-text pairs, only for datasets that have them
    if "image" in types and "text" in types:
        n = len(embeddings) // 2
        for i in range(n):
            img_coord = coords_3d[i]
            text_coord = coords_3d[i + n]
            ax.plot(
                [img_coord[0], text_coord[0]],
                [img_coord[1], text_coord[1]],
                [img_coord[2], text_coord[2]],
                color="gray",
                alpha=0.3,
                linewidth=0.7,
            )

    ax.set_title(f"CLIP ({model_name}) Embeddings in 3D using PCA")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

    plt.tight_layout()
    plt.show()

elif method == "pca-2d":
    # normalize embeddings first
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    reducer = PCA(n_components=2)
    coords_2d = reducer.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    cmap = get_cmap("tab20", len(unique_labels))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    # --- 2D Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=[label_to_color[lab] for lab in labels],
        s=40,
        alpha=0.8,
    )

    # Draw lines between image-text pairs, only for datasets that have them
    if "image" in types and "text" in types:
        n = len(embeddings) // 2
        for i in range(n):
            img_coord = coords_2d[i]
            text_coord = coords_2d[i + n]
            ax.plot(
                [img_coord[0], text_coord[0]],
                [img_coord[1], text_coord[1]],
                color="gray",
                alpha=0.3,
                linewidth=0.7,
            )

    ax.set_title(f"CLIP ({model_name}) Embeddings in 2D using PCA")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    plt.tight_layout()
    plt.show()

elif method == "umap":

    # --- Dimensionality Reduction using UMAP ---
    reducer = umap.UMAP(n_components=3, random_state=42)
    coords_3d = reducer.fit_transform(embeddings)

    # --- 3D Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Map labels to colors
    unique_labels = sorted(set(labels))
    cmap = get_cmap("tab20", len(unique_labels))
    label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    # Scatter embeddings
    for i, (emb, lab, typ) in enumerate(zip(coords_3d, labels, types)):
        color = label_to_color[lab]
        marker = "o" if typ == "image" else "^"
        ax.scatter(
            emb[0],
            emb[1],
            emb[2],
            color=color,
            marker=marker,
            s=40,
            alpha=0.8,
            label=f"{lab} ({typ})",
        )

    # Draw lines between image-text pairs, only for datasets that have them
    if "image" in types and "text" in types:
        n = len(embeddings) // 2
        for i in range(n):
            img_coord = coords_3d[i]
            text_coord = coords_3d[i + n]
            ax.plot(
                [img_coord[0], text_coord[0]],
                [img_coord[1], text_coord[1]],
                [img_coord[2], text_coord[2]],
                color="gray",
                alpha=0.3,
                linewidth=0.7,
            )

    ax.set_title(f"CLIP ({model_name}) Embeddings in 3D using UMAP")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")

    plt.tight_layout()
    plt.show()
