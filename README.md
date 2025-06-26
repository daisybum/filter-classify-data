# Filter & Classify Weather Image Folders

This utility scans *date/time-named* sub-folders, predicts the prevailing weather condition from their JPEG images, and copies the entire folder into the corresponding class directory.

| Label Index | Class Name               |
|-------------|--------------------------|
| 0           | Hazy                     |
| 1           | Normal                   |
| 2           | raining                  |
| 3           | rainy but not raining    |
| 4           | snowing                  |
| 5           | snowy but not snowing    |
| 6           | unclear                  |

The classifier is based on **Swin-Transformer V2** and supports GPU inference on CUDA 12.4.

---

## Quick Start

### 1 · Create a virtual environment
```bash
# Windows / macOS / Linux (Python ≥ 3.10)
python -m venv .venv

# Activate
#   Windows PowerShell
.venv\Scripts\Activate.ps1
#   macOS / Linux
source .venv/bin/activate
```

### 2 · Install dependencies
```bash
pip install -r requirements.txt
```

(The `--extra-index-url` at the top of `requirements.txt` tells pip where to download the official CUDA 12.4 wheels for PyTorch.)

### 3 · Add your trained model checkpoint
Place your `.pth` file under:
```
weather_classification/models/
```
Only the first matching checkpoint is loaded.

### 4 · Configure paths
Create `config.json` (anywhere) with at least:
```json
{
  "src": "~/path/to/source/root",
  "dst": "~/path/to/destination/root",
  "overwrite": false
}
```
* `src` – root directory that contains one-level-deep sub-folders (e.g. `20250101_1200/`)
* `dst` – destination root where class directories will be created
* `overwrite` – if `true`, existing destination folders will be replaced

### 5 · Run
```bash
python main.py --config config.json
```
A progress bar (powered by **tqdm**) indicates how many source folders have been processed.

---

## Project Layout (simplified)
```
filter-classify-data/
├── main.py                  # CLI script: scanning, classification, copy
├── requirements.txt         # Python dependencies (CUDA 12.4 wheels)
├── README.md                # → you are here
└── src/
    └── classifier.py        # Lazy-loaded Swin-V2 inference helper
```

## Notes
* CPU inference works too; GPU is automatically selected when available.
* The model is loaded **once** thanks to `functools.lru_cache`, then reused.
* Classification accuracy depends entirely on the checkpoint you provide.

---

## License
MIT (see `LICENSE` file)
