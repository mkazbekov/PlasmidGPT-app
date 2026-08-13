"""Shared, UI-independent PlasmidGPT inference helpers.

The original command-line scripts remain available.  This module centralizes
their behavior so the local browser app can load the large foundation model
once, validate input consistently, and support every bundled classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from Bio import SeqIO
from transformers import GenerationConfig, PreTrainedTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "pretrained_model"
PREDICTION_MODEL_DIR = PROJECT_ROOT / "prediction_models"
ALLOWED_BASES = frozenset("ATCGN")


@dataclass(frozen=True)
class PredictionTask:
    label: str
    description: str
    model_path: Path
    labels_path: Path


PREDICTION_TASKS: dict[str, PredictionTask] = {
    "Lab of origin": PredictionTask(
        label="Lab of origin",
        description="Which depositing lab is most similar to this sequence?",
        model_path=PREDICTION_MODEL_DIR / "embedding_prediction_labs.pth",
        labels_path=PREDICTION_MODEL_DIR / "lab_list.txt",
    ),
    "Host species": PredictionTask(
        label="Host species",
        description="Which host species is most associated with this sequence?",
        model_path=PREDICTION_MODEL_DIR / "embedding_prediction_Species.pth",
        labels_path=PREDICTION_MODEL_DIR / "species_list.txt",
    ),
    "Vector type": PredictionTask(
        label="Vector type",
        description="Which vector category is most similar to this sequence?",
        model_path=PREDICTION_MODEL_DIR / "embedding_prediction_Vector type.pth",
        labels_path=PREDICTION_MODEL_DIR / "vector_list.txt",
    ),
}


class SimpleNN(nn.Module):
    """The small classifier architecture used by the bundled predictors."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(values)))


def preferred_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_summary() -> tuple[str, str]:
    """Return a short label and explanation for the active inference device."""

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "GPU ready", name
    return "CPU mode", "No CUDA GPU detected; inference will be slower."


def installation_issues() -> list[str]:
    required = [
        DEFAULT_MODEL_DIR / "pretrained_model.pt",
        DEFAULT_MODEL_DIR / "addgene_trained_dna_tokenizer.json",
    ]
    for task in PREDICTION_TASKS.values():
        required.extend([task.model_path, task.labels_path])
    return [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]


def validate_sequence(sequence: str) -> str:
    """Normalize DNA and reject empty input or unsupported characters."""

    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("Enter a DNA sequence or upload a FASTA file.")
    invalid = sorted(set(normalized) - ALLOWED_BASES)
    if invalid:
        shown = ", ".join(invalid[:8])
        raise ValueError(
            f"DNA can contain only A, T, C, G, and N. Remove: {shown}"
        )
    return normalized


def parse_sequence_input(text: str | bytes) -> list[tuple[str, str]]:
    """Parse either a FASTA document or plain DNA into named sequences."""

    if isinstance(text, bytes):
        try:
            decoded = text.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("The uploaded file must be a plain-text FASTA file.") from exc
    else:
        decoded = text

    decoded = decoded.strip()
    if not decoded:
        raise ValueError("Enter a DNA sequence or upload a FASTA file.")

    if decoded.lstrip().startswith(">"):
        records = list(SeqIO.parse(StringIO(decoded), "fasta"))
        if not records:
            raise ValueError("No sequences were found in the FASTA file.")
        parsed: list[tuple[str, str]] = []
        for index, record in enumerate(records, start=1):
            name = record.id.strip() or f"sequence_{index}"
            parsed.append((name, validate_sequence(str(record.seq))))
        return parsed

    return [("sequence_1", validate_sequence(decoded))]


def sequence_stats(sequence: str) -> dict[str, float | int]:
    sequence = validate_sequence(sequence)
    known_bases = sum(sequence.count(base) for base in "ATCG")
    gc_bases = sequence.count("G") + sequence.count("C")
    gc_percent = (gc_bases / known_bases * 100.0) if known_bases else 0.0
    return {
        "length": len(sequence),
        "gc_percent": gc_percent,
        "ambiguous_bases": sequence.count("N"),
    }


def fasta_bytes(
    sequences: Iterable[tuple[str, str]],
    *,
    line_width: int = 80,
) -> bytes:
    lines: list[str] = []
    for index, (name, sequence) in enumerate(sequences, start=1):
        safe_name = "_".join(name.strip().split()) or f"sequence_{index}"
        normalized = validate_sequence(sequence)
        lines.append(f">{safe_name}")
        lines.extend(
            normalized[offset : offset + line_width]
            for offset in range(0, len(normalized), line_width)
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def load_foundation_model(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    device: torch.device | None = None,
):
    """Load the pretrained model and tokenizer onto the preferred device."""

    model_dir = Path(model_dir)
    model_path = model_dir / "pretrained_model.pt"
    tokenizer_path = model_dir / "addgene_trained_dna_tokenizer.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    active_device = device or preferred_device()
    # The checkpoint pickles the full GPT2LMHeadModel object (not just a
    # state_dict), so it needs weights_only=False to unpickle under
    # PyTorch >=2.6's stricter default. This is the project's own bundled,
    # trusted checkpoint.
    model = torch.load(model_path, map_location=active_device, weights_only=False)
    model.eval()
    model.to(active_device)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[PROMPT]", "[PROMPT2]"]}
    )
    return model, tokenizer, active_device


def generate_sequences(
    model,
    tokenizer: PreTrainedTokenizerFast,
    start_sequence: str,
    *,
    num_sequences: int = 1,
    max_new_tokens: int = 160,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> list[str]:
    """Generate DNA continuations with reliable masks and bounded retries."""

    start_sequence = validate_sequence(start_sequence)
    if not 1 <= num_sequences <= 10:
        raise ValueError("Choose between 1 and 10 sequences.")
    if not 32 <= max_new_tokens <= 768:
        raise ValueError("Generation length must be between 32 and 768 tokens.")
    if not 0.1 <= temperature <= 2.0:
        raise ValueError("Creativity must be between 0.1 and 2.0.")

    active_device = device or next(model.parameters()).device
    encoded = tokenizer.encode(start_sequence, return_tensors="pt").to(active_device)
    prompt_tokens = torch.tensor(
        [3] * 10 + [2], dtype=torch.long, device=active_device
    ).unsqueeze(0)
    input_ids = torch.cat((prompt_tokens, encoded), dim=1)

    context_limit = int(
        getattr(model.config, "n_positions", None)
        or getattr(model.config, "max_position_embeddings", 2048)
    )
    available = context_limit - input_ids.shape[1]
    if available < 32:
        raise ValueError(
            "This starting sequence is too long for generation. Use a shorter "
            "design region (the model can use at most 2,048 tokens)."
        )
    actual_new_tokens = min(max_new_tokens, available)
    attention_mask = torch.ones_like(input_ids, device=active_device)
    generated: list[str] = []
    generation_config = GenerationConfig.from_model_config(model.config)
    # The checkpoint inherited GPT-2's 50,256 BOS/EOS IDs, but its actual
    # vocabulary has only 30,002 entries.  Open-ended generation does not need
    # either token, so clear the invalid inherited values and use a valid pad.
    generation_config.bos_token_id = None
    generation_config.eos_token_id = None
    generation_config.pad_token_id = 0
    with torch.inference_mode():
        for _ in range(num_sequences):
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=actual_new_tokens,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                generation_config=generation_config,
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            cleaned = "".join(decoded.split()).upper()
            cleaned = "".join(base for base in cleaned if base in ALLOWED_BASES)
            if not cleaned:
                raise RuntimeError("The model returned an empty sequence. Try again.")
            generated.append(cleaned)
    return generated


def calculate_embeddings(
    model,
    tokenizer: PreTrainedTokenizerFast,
    sequences: Sequence[str],
    *,
    device: torch.device | None = None,
    max_length: int = 2048,
) -> np.ndarray:
    """Return one mean-pooled, 768-value embedding per DNA sequence."""

    if not sequences:
        raise ValueError("No sequences were provided.")
    active_device = device or next(model.parameters()).device
    embeddings: list[np.ndarray] = []
    for sequence in sequences:
        normalized = validate_sequence(sequence)
        input_ids = tokenizer.encode(
            normalized,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(active_device)
        with torch.inference_mode():
            outputs = model(input_ids, output_hidden_states=True)
            pooled = outputs.hidden_states[-1].mean(dim=1).squeeze(0)
        embeddings.append(pooled.detach().cpu().numpy())
    return np.asarray(embeddings, dtype=np.float32)


def load_prediction_model(
    task_name: str,
    device: torch.device | None = None,
) -> tuple[SimpleNN, np.ndarray, torch.device]:
    if task_name not in PREDICTION_TASKS:
        raise ValueError(f"Unknown prediction task: {task_name}")
    task = PREDICTION_TASKS[task_name]
    active_device = device or preferred_device()
    labels = np.asarray(
        [line.strip() for line in task.labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    )
    state = torch.load(task.model_path, map_location=active_device)
    output_classes = int(state["fc2.weight"].shape[0])
    if len(labels) != output_classes:
        raise ValueError(
            f"{task.label} has {len(labels)} labels but its model has "
            f"{output_classes} outputs."
        )
    classifier = SimpleNN(input_dim=768, num_classes=output_classes).to(active_device)
    classifier.load_state_dict(state)
    classifier.eval()
    return classifier, labels, active_device


def predict_attributes(
    embeddings: np.ndarray,
    classifier: SimpleNN,
    labels: np.ndarray,
    *,
    top_n: int = 5,
    device: torch.device | None = None,
) -> list[list[tuple[str, float]]]:
    """Predict the top labels for each embedding."""

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 768:
        raise ValueError("Embeddings must contain 768 values per sequence.")
    top_n = max(1, min(int(top_n), len(labels)))
    active_device = device or next(classifier.parameters()).device
    tensor = torch.from_numpy(values).to(active_device)
    with torch.inference_mode():
        probabilities = torch.softmax(classifier(tensor), dim=1)
        top_probabilities, top_indices = torch.topk(probabilities, k=top_n, dim=1)

    results: list[list[tuple[str, float]]] = []
    for row_indices, row_probabilities in zip(
        top_indices.cpu().numpy(), top_probabilities.cpu().numpy()
    ):
        results.append(
            [
                (str(labels[index]), float(probability))
                for index, probability in zip(row_indices, row_probabilities)
            ]
        )
    return results
