# Museum Artwork Guide

A coursework prototype exploring artwork-style classification and artwork retrieval for a fictional museum guide. The experiments compare a PyTorch feed-forward network with an ImageNet-pretrained ResNet-18, and inspect text-based retrieval approaches.

## Project contents

- `AI_Museum_Guide.ipynb` — dataset inspection, training, evaluation, and retrieval experiments.
- `make_wikiart_subset.py` — downloads and prepares a reproducible WikiArt image subset.
- `data/wikiart_subset/` — 1,500 images and metadata across five art styles.
- `outputs_official/` — created when the notebook runs: model checkpoints, plots, and the experiment summary (not stored on `main`).

## Dataset

The subset contains 300 images per class:

- Baroque
- Cubism
- Impressionism
- Pop Art
- Ukiyo-e

The notebook uses a stratified 70/15/15 split with seed 42, so every class is equally represented in each split:

| Split | Images | Per class |
|---|---|---|
| Train | 1,050 | 210 |
| Validation | 225 | 45 |
| Test | 225 | 45 |

The test set is used only once per model, for the final evaluation; checkpoints are chosen on validation loss. Images are organized in `ImageFolder`-compatible class directories; `metadata.csv` records labels and other available metadata.

The image files are from WikiArt. The dataset script notes that image licensing is unknown, so the images are for coursework use only and should not be reused or redistributed. See `make_wikiart_subset.py` and `data/wikiart_subset/subset_info.json` for dataset preparation details.

## Notebook structure

| Section | Checkpoint | What it does |
|---|---|---|
| 0–2 | — | Imports, seed, dataset inspection, stratified split |
| 3 | — | Shared training loop, early stopping, plotting and evaluation helpers |
| 4 | m1 + m2 | Dense feed-forward baseline (MLP), training curves, error diagnosis |
| 5 | m3 | ImageNet-pretrained ResNet-18: head-only training, then fine-tuning; comparison with the MLP |
| 6 | m4 | Text-to-artwork retrieval: TF–IDF vs sentence embeddings |

All image models share the same training function (`fit_model`): cross-entropy loss, Adam, the best checkpoint by validation loss is saved and reloaded, and training stops early after 5 epochs without improvement.

## Setup

Run commands from the `Project-HSM` directory. Create and activate a Python environment, then install the notebook dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy pandas matplotlib pillow scikit-learn torch torchvision

# optional, for the neural retrieval part of m4
python -m pip install sentence-transformers
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The first ResNet run downloads the ImageNet weights, so it needs internet access.

Then open the notebook and run all cells from top to bottom:

```bash
jupyter notebook AI_Museum_Guide.ipynb
```

The notebook fixes the random seed (`SEED = 42`) for Python, NumPy and PyTorch, so the split is identical on every run. Training results can still vary slightly between machines (CPU vs GPU). Plots, checkpoints and `experiment_summary.json` are written to `outputs_official/`.

## m1–m2: Feed-forward baseline (MLP)

**What it is.** The simplest neural network for the task, used as the baseline that ResNet-18 must beat. Each image is resized to 64×64, flattened into 3 × 64 × 64 = 12,288 pixel values, and passed through fully connected layers:

| Layer | Shape | Notes |
|---|---|---|
| Input | 12,288 | RGB pixels in [0, 1], no augmentation |
| Hidden 1 | 512 | ReLU, dropout 0.3 |
| Hidden 2 | 128 | ReLU, dropout 0.2 |
| Output | 5 | One logit per style |

About 6.36 M parameters in total, almost all of them in the first layer.

**Training.** Adam (learning rate 1e-3, weight decay 1e-4), cross-entropy loss, batch size 32, up to 20 epochs. The checkpoint with the lowest validation loss is kept (`mlp_best.pt`), with early stopping after 5 epochs without improvement.

**Results (seed 42).** The evaluated model is the epoch-19 checkpoint, the one with the lowest validation loss.

| Split | Accuracy |
|---|---|
| Random guess (5 classes) | 20.0 % |
| Validation (best checkpoint, epoch 19) | 34.2 % |
| **Test (225 images)** | **38.7 %** |

| Style | Precision | Recall | F1 |
|---|---|---|---|
| Baroque | 0.52 | 0.53 | 0.53 |
| Cubism | 0.67 | 0.04 | 0.08 |
| Impressionism | 0.27 | 0.07 | 0.11 |
| Pop Art | 0.46 | 0.38 | 0.41 |
| Ukiyo-e | 0.32 | 0.91 | 0.47 |

**Diagnosis (m2).**

- **Underfitting, not overfitting.** Training and validation accuracy stay close (≈35 % vs ≈37 %) and training loss stays high, so the model cannot even fit the training data well.
- **Cause: the input representation.** Flattening destroys spatial structure, so the MLP can only use global colour and brightness, not the shapes, edges and brushwork that define a style. It has enough parameters; the problem is what it can see.
- **Class behaviour.** Ukiyo-e acts as a "default" prediction (high recall, low precision). Baroque is recognised best, probably from its dark palette. Cubism and Impressionism, which depend on geometry and texture, are almost never detected.
- **Noisy validation curve.** Validation accuracy jumps between about 28 % and 37 % from epoch to epoch. With 225 validation images, one image moves accuracy by about 0.44 points, which is why checkpoints are selected by validation loss, a smoother signal, rather than accuracy.

**Takeaway.** Training longer or adding hidden units will not fix this. A model that keeps spatial structure is needed, which motivates the pretrained ResNet-18 in m3.

## m3: Transfer learning with ResNet-18

**Why.** The MLP diagnosis showed that the bottleneck is the input representation. ResNet-18 is a convolutional network pretrained on ImageNet, so it already detects edges, textures and shapes, which is exactly what art styles depend on.

**Setup.** Images are resized to 224×224 and normalised with ImageNet's channel statistics. The original 1,000-class output layer is replaced with a 5-class linear layer. Training happens in two stages:

| Stage | What is trained | Trainable parameters | Learning rate | Epochs |
|---|---|---|---|---|
| 1. Head only | New output layer; backbone frozen | 2,565 | 1e-3 | 8 |
| 2. Fine-tuning | `layer4` + output layer | 8,396,293 | 1e-4 (layer4), 1e-3 (head) | 10 max (stopped at 7) |

The pretrained layer gets a 10× smaller learning rate so fine-tuning adjusts its features without destroying them.

**Results.**

- Head only: 74.7 % validation accuracy, already about double the MLP, while training only 2,565 parameters.
- Fine-tuning: best validation loss at epoch 2 (79.1 % validation accuracy). After that, training accuracy climbs to 99.8 % while validation loss rises, which is clear overfitting. Early stopping ended training at epoch 7 and the epoch-2 checkpoint was restored.
- **Test accuracy: 80.0 %** (180/225).

| Style | Precision | Recall | F1 |
|---|---|---|---|
| Baroque | 0.80 | 0.87 | 0.83 |
| Cubism | 0.87 | 0.73 | 0.80 |
| Impressionism | 0.80 | 0.73 | 0.77 |
| Pop Art | 0.69 | 0.78 | 0.73 |
| Ukiyo-e | 0.87 | 0.89 | 0.88 |

Unlike the MLP, every class is recognised reasonably well, including Cubism and Impressionism.

## Model comparison

| Model | Input | Trainable parameters | Test accuracy | Main issue |
|---|---|---|---|---|
| Random guess | — | — | 20.0 % | — |
| MLP (m1) | 64×64, flattened | 6.36 M | 38.7 % | Underfitting: cannot see spatial structure |
| ResNet-18 fine-tuned (m3) | 224×224 | 8.40 M (of 11.18 M) | **80.0 %** | Overfits quickly when fine-tuned; handled by early stopping |

Pretrained convolutional features more than double the MLP's accuracy with a similar number of trainable parameters, so the gain comes from the architecture and pretraining, not from model size. This is a single run on one split and seed, so it is strong evidence for this dataset, not a general proof.

## m4: Question-to-artwork retrieval

**Goal.** Let a visitor type a question (for example "a stormy sea") and get matching artworks.

**Approach.** Each image gets a short text record built from `metadata.csv` (style, genre, artist), matched for all 1,500 images. The metadata is thin: 82 % of images have "Unknown Genre" and 93 % "Unknown Artist", and the known genres are only still life, genre painting, sketch and landscape. Two retrieval methods are compared using cosine similarity:

- **TF–IDF (classical):** word and bigram counts weighted by rarity.
- **Sentence embeddings (neural):** pretrained `all-MiniLM-L6-v2` model, which captures meaning rather than exact words.

**Findings on the example queries.**

| Query | TF–IDF | Sentence embeddings |
|---|---|---|
| "a stormy sea" | All scores 0; ranking arbitrary | Baroque records; 2 of the top 5 are landscapes (score ≈ 0.19) |
| "portraits of children" | All scores 0 | Impressionism records with identical text (≈ 0.37), so their order is arbitrary |
| "geometric abstract painting" | All scores 0 | Cubism genre paintings (≈ 0.52) |

TF–IDF fails because the queries share no words with the metadata (even "painting" does not match, because the genre is stored as the single token `genre_painting`). Sentence embeddings return plausible styles (geometric → Cubism), because they match meaning. But both methods only see style, genre and artist, and many records are "Unknown Genre Unknown Artist", so neither can actually find a stormy sea or a child's portrait. There are no ground-truth relevance labels, so retrieval is assessed qualitatively only.

**Next step.** Real retrieval needs content descriptions (titles, captions), or an image–text model such as CLIP that compares the query directly with the image.

## Limitations

- Small dataset (1,500 images, 5 styles) and a single split and seed; results may vary by a few points on other splits.
- 225 validation and test images per model, so one image changes accuracy by about 0.44 points.
- Results come from one CPU run; no data augmentation was used for either model.
- Retrieval metadata contains no content descriptions, so content questions cannot be answered yet.
