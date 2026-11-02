# Museum Artwork Guide

A coursework prototype exploring artwork-style classification and artwork retrieval for a fictional museum guide. The experiments compare a PyTorch feed-forward network with an ImageNet-pretrained ResNet-18, and inspect text-based retrieval approaches.

## Project contents

- `AI_Museum_Guide.ipynb` — dataset inspection, training, evaluation, and retrieval experiments.
- `make_wikiart_subset.py` — downloads and prepares a reproducible WikiArt image subset.
- `data/wikiart_subset/` — 1,500 images and metadata across five art styles.
- `outputs_official/` — saved model checkpoints, plots, and experiment summaries.
- `Project_requirements.txt` — project brief.

## Dataset

The subset contains 300 images per class:

- Baroque
- Cubism
- Impressionism
- Pop Art
- Ukiyo-e

The notebook uses a stratified split with seed 42: 1,050 training images, 225 validation images, and 225 test images. Images are organized in `ImageFolder`-compatible class directories; `metadata.csv` records labels and other available metadata.

The image files are from WikiArt. The dataset script notes that image licensing is unknown: use the images for coursework, and do not publish or commit them. See `make_wikiart_subset.py` and `data/wikiart_subset/subset_info.json` for dataset preparation details.

## Setup

Run commands from the `Project-HSM` directory. Create and activate a Python environment, then install the notebook dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy pandas matplotlib pillow scikit-learn torch torchvision