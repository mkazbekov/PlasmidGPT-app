# PlasmidGPT Studio

A local, private browser app for designing and analyzing plasmid DNA with **PlasmidGPT**, a generative language model pretrained on 153k engineered plasmid sequences from [Addgene](https://www.addgene.org/). No command line, no cloud, no account — clone it, run one script, and you're generating sequences in your browser.

![PlasmidGPT Studio](https://github.com/user-attachments/assets/fc75bf4f-972c-4e3e-913e-499f01ab41ba)

> **Upstream attribution:** This repository is an interface-focused adaptation of the original [lingxusb/PlasmidGPT](https://github.com/lingxusb/PlasmidGPT) project. The underlying model, scientific methods, and original scripts/notebooks come from that project. See [UPSTREAM.md](UPSTREAM.md) for full provenance and licensing details.

## What it does

PlasmidGPT Studio has four work areas, all running on your own machine:

- **Generate** — start from a short DNA seed (or upload a FASTA file) and generate candidate plasmid sequences. Download the results as FASTA, or send a sequence straight into Predict.
- **Predict** — paste or upload a sequence and predict its most likely **lab of origin**, **host species**, or **vector type**, ranked with confidence scores.
- **Embeddings** — convert one or more sequences into 768-value numerical embeddings for downstream analysis (clustering, similarity search, custom ML).
- **Help** — an in-app quick guide and troubleshooting reference.

Your sequences never leave your computer. The app uses your GPU automatically when one is available, and falls back to CPU otherwise.

## Install and run

You need [Git](https://git-scm.com/downloads) with [Git LFS](https://git-lfs.com/) (the pretrained model is ~489 MB and is stored via LFS) and Python 3.10+.

```bash
git lfs install
git clone https://github.com/mkazbekov/PlasmidGPT-app.git
cd PlasmidGPT-app
```

Then start the app:

| Platform | Command |
| --- | --- |
| Windows | Double-click **`Start PlasmidGPT.cmd`** |
| macOS / Linux | `chmod +x start.sh && ./start.sh` |

The first run creates a local Python environment and installs dependencies automatically — this takes a few minutes and happens only once. Your browser then opens PlasmidGPT Studio. Keep the launcher window open while using the app; closing it stops the local server.

> Downloaded a ZIP instead of cloning with Git? Git LFS files (the pretrained model) won't be included, and the app will tell you what's missing on startup. Clone with `git lfs install` first instead.

## Advanced usage

The original command-line scripts (`generate.py`, `embeddings.py`, `prediction.py`) and the reproducibility notebooks remain fully available for scripted or batch workflows, using the virtual environment created by the launcher above. See **[HOW_TO_RUN.txt](HOW_TO_RUN.txt)** for the complete command reference, argument tables, FASTA format notes, and troubleshooting.

<details>
<summary>Reproducibility notebooks</summary>

Jupyter notebooks that reproduce the analyses from the PlasmidGPT paper live in [`reproducibility/`](reproducibility). Required data can be downloaded from Hugging Face:

```bash
wget https://huggingface.co/lingxusb/PlasmidGPT/resolve/main/reproducibility.zip
unzip reproducibility.zip
```

| Notebook | Description |
| --- | --- |
| `0_embedding_calculation.ipynb` | Calculate sequence embeddings using PlasmidGPT |
| `1_embedding_visualization.ipynb` | Visualize embeddings with t-SNE, analyze lab-specific patterns and plasmid diversity |
| `2_prediction_benchmarking.ipynb` | Benchmark lab prediction using PlasmidGPT embeddings with cluster-based cross-validation |
| `3_CNN_benchmarking.ipynb` | Benchmark lab prediction using a CNN baseline with one-hot encoded sequences |
| `4_generation_analysis.ipynb` | Compare part organization (synteny, co-occurrence) between real and generated plasmids |
| `5_finetune_analysis.ipynb` | Visualization figures for the fine-tuned plasmid generation analysis |

</details>

<details>
<summary>Trained model files</summary>

The trained model and tokenizer are available on [Hugging Face](https://huggingface.co/lingxusb/PlasmidGPT/tree/main):

- `pretrained_model.pt` — the pretrained PlasmidGPT model
- `addgene_trained_dna_tokenizer.json` — the BPE tokenizer trained on Addgene plasmid sequences

Both are already bundled in this repository via Git LFS, so a normal `git clone` (with `git lfs install`) is all you need.

</details>

## Testing

```bash
python -m unittest discover tests
```

## Important notes

PlasmidGPT generates computational predictions and candidate sequences. Review outputs with appropriate biological validation and biosafety procedures before using them experimentally.

## Reference

- [PlasmidGPT: a generative framework for plasmid design and annotation (Preprint)](https://www.biorxiv.org/content/10.1101/2024.09.30.615762v1)
- [PlasmidGPT: A generative framework for plasmid analysis and generation (Sci Adv 2026)](https://www.science.org/doi/10.1126/sciadv.aee6916)

## License

The repository code is MIT licensed (see [LICENSE](LICENSE)). The pretrained model is distributed under **CC BY-NC 4.0**, which restricts commercial use — see [UPSTREAM.md](UPSTREAM.md) for details.
