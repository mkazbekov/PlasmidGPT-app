"""One-click local browser interface for PlasmidGPT."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

from plasmidgpt_app import (
    PREDICTION_TASKS,
    calculate_embeddings,
    device_summary,
    fasta_bytes,
    generate_sequences,
    installation_issues,
    load_foundation_model,
    load_prediction_model,
    parse_sequence_input,
    predict_attributes,
    sequence_stats,
)


SAMPLE_SEQUENCE = "ATCGATCGATCG"


st.set_page_config(
    page_title="PlasmidGPT Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #112622;
      --muted: #59706a;
      --forest: #075e54;
      --mint: #c8f7dc;
      --lime: #d9ff73;
      --cream: #f6f4eb;
      --line: #dce8e3;
      --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
    }
    .stApp { background: linear-gradient(180deg, #f8fbf8 0%, #f3f6f1 100%); }
    [data-testid="stHeader"] { background: rgba(248,251,248,.88); }
    [data-testid="stSidebar"] { background: #102b27; }
    [data-testid="stSidebar"] * { color: #f4fff8; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #bcd4cb; }
    .block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    .hero {
      border: 1px solid #d8e6df;
      border-radius: 26px;
      padding: 2.15rem 2.35rem;
      margin-bottom: 1.5rem;
      background:
        radial-gradient(circle at 89% 18%, rgba(217,255,115,.78), transparent 24%),
        linear-gradient(135deg, #e5f8ed 0%, #f8fbf5 58%, #f5f2df 100%);
      box-shadow: 0 18px 55px rgba(26, 70, 58, .08);
    }
    .eyebrow { color: var(--forest); font-size: .75rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
    .hero h1 { margin: .45rem 0 .55rem; font-size: clamp(2.1rem, 4vw, 3.7rem); line-height: .98; }
    .hero p { color: #3f5f56; font-size: 1.06rem; max-width: 720px; margin: 0; }
    .privacy { display: inline-flex; margin-top: 1rem; padding: .36rem .72rem; border-radius: 999px; background: #fff; color: #1e5b4d; font-size: .78rem; font-weight: 700; border: 1px solid #d6e8df; }
    .section-intro { color: var(--muted); margin-top: -.35rem; margin-bottom: 1.2rem; }
    .result-card {
      border: 1px solid var(--line); background: white; border-radius: 18px; padding: 1rem 1.1rem; margin: .6rem 0;
      opacity: 1; transform: translateY(0);
      transition: opacity 220ms var(--ease-out), transform 220ms var(--ease-out);
      @starting-style { opacity: 0; transform: translateY(6px); }
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) .result-card { transition-delay: 60ms; }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(3) .result-card { transition-delay: 120ms; }
    .result-title { color: var(--ink); font-weight: 750; }
    .result-meta { color: var(--muted); font-size: .82rem; }
    .sequence-preview { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .78rem; overflow-wrap: anywhere; color: #17493d; background: #f0f7f3; border-radius: 10px; padding: .75rem; margin-top: .65rem; }
    .callout { background: #edf8f2; border-left: 4px solid #16806c; padding: .9rem 1rem; border-radius: 0 12px 12px 0; color: #315c52; }
    .stButton > button, .stDownloadButton > button {
      border-radius: 999px; font-weight: 700; min-height: 2.7rem;
      transition: transform 130ms var(--ease-out);
    }
    .stButton > button:active, .stDownloadButton > button:active { transform: scale(0.97); }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background: var(--forest); border-color: var(--forest); }
    [data-testid="stMetric"] {
      background: #fff; border: 1px solid var(--line); padding: .8rem 1rem; border-radius: 15px;
      opacity: 1; transition: opacity 220ms var(--ease-out);
      @starting-style { opacity: 0; }
    }
    [data-testid="stFileUploaderDropzone"] {
      background: #f8fbf9; border-color: #bfd6cc;
      transition: border-color 150ms var(--ease-out), background-color 150ms var(--ease-out);
    }
    @media (hover: hover) and (pointer: fine) {
      [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--forest); background: #eef8f2; }
    }
    [data-testid="stAlert"] {
      opacity: 1;
      transition: opacity 180ms var(--ease-out);
      @starting-style { opacity: 0; }
    }
    @media (prefers-reduced-motion: reduce) {
      .result-card, [data-testid="stMetric"], [data-testid="stAlert"] { transition-duration: 80ms; }
      .result-card { transition-property: opacity; transform: none !important; }
      .result-card { @starting-style { transform: none; } }
    }
    code { color: #155a4a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def foundation_assets():
    return load_foundation_model()


@st.cache_resource(show_spinner=False)
def prediction_assets(task_name: str):
    return load_prediction_model(task_name)


def input_sequences(text: str, uploaded_file) -> list[tuple[str, str]]:
    if uploaded_file is not None:
        return parse_sequence_input(uploaded_file.getvalue())
    return parse_sequence_input(text)


def set_sample(target_key: str) -> None:
    st.session_state[target_key] = SAMPLE_SEQUENCE


def send_generated_to_prediction() -> None:
    generated = st.session_state.get("generated_sequences", [])
    if generated:
        st.session_state["prediction_dna"] = generated[0][1]
        st.session_state["section"] = "Predict"


def render_status() -> None:
    status, detail = device_summary()
    st.sidebar.markdown("### PlasmidGPT")
    st.sidebar.caption("Local sequence design & analysis")
    st.sidebar.markdown("---")
    if status == "GPU ready":
        st.sidebar.success(f"{status}\n\n{detail}")
    else:
        st.sidebar.warning(f"{status}\n\n{detail}")
    st.sidebar.caption("Your sequences stay on this computer.")


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Local AI workspace</div>
          <h1>PlasmidGPT Studio</h1>
          <p>Generate candidate plasmid DNA, predict sequence attributes, and export model embeddings—without commands or file paths.</p>
          <span class="privacy">● Runs privately on this computer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_generated_results() -> None:
    results = st.session_state.get("generated_sequences")
    if not results:
        return
    st.markdown("### Your generated sequences")
    columns = st.columns(min(3, len(results)))
    for index, (name, sequence) in enumerate(results):
        stats = sequence_stats(sequence)
        preview = sequence if len(sequence) <= 240 else f"{sequence[:240]}…"
        with columns[index % len(columns)]:
            st.markdown(
                f"""
                <div class="result-card">
                  <div class="result-title">Sequence {index + 1}</div>
                  <div class="result-meta">{stats['length']:,} bases · {stats['gc_percent']:.1f}% GC</div>
                  <div class="sequence-preview">{preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    download = fasta_bytes(results)
    action_left, action_right, _ = st.columns([1.25, 1.25, 2])
    with action_left:
        st.download_button(
            "Download FASTA",
            data=download,
            file_name="plasmidgpt_generated.fasta",
            mime="text/plain",
            type="primary",
            width="stretch",
        )
    with action_right:
        st.button(
            "Predict first sequence",
            on_click=send_generated_to_prediction,
            width="stretch",
        )


def render_generate() -> None:
    st.markdown("## Generate DNA")
    st.markdown(
        '<p class="section-intro">Start with a short DNA sequence or upload a FASTA file. Beginner-friendly settings are already selected.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.text_area(
            "Starting DNA",
            key="generation_dna",
            height=180,
            placeholder="Paste A, T, C, G, or N…",
            help="Whitespace and line wrapping are ignored.",
        )
        sample_col, hint_col = st.columns([1, 2])
        with sample_col:
            st.button(
                "Use an example",
                on_click=set_sample,
                args=("generation_dna",),
                width="stretch",
            )
        with hint_col:
            st.caption("A short seed is enough. The model continues it.")
    with right:
        uploaded = st.file_uploader(
            "Or upload FASTA",
            type=["fasta", "fa", "fas", "txt"],
            key="generation_upload",
        )
        st.markdown(
            '<div class="callout"><b>First run:</b> loading the model can take 10–20 seconds. Later runs reuse it.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Advanced options"):
        option_a, option_b, option_c = st.columns(3)
        with option_a:
            count = st.slider("Number of sequences", 1, 5, 1)
        with option_b:
            length = st.slider(
                "Generation length",
                64,
                512,
                160,
                step=32,
                help="Model tokens to add. DNA base count will differ.",
            )
        with option_c:
            temperature = st.slider(
                "Creativity",
                0.5,
                1.5,
                1.0,
                step=0.1,
                help="Lower is more conservative; higher is more varied.",
            )

    if st.button("Generate sequences", type="primary", width="stretch"):
        try:
            sequences = input_sequences(st.session_state.get("generation_dna", ""), uploaded)
            _, starting_sequence = sequences[0]
            with st.spinner("Loading PlasmidGPT and generating DNA…"):
                model, tokenizer, device = foundation_assets()
                generated = generate_sequences(
                    model,
                    tokenizer,
                    starting_sequence,
                    num_sequences=count,
                    max_new_tokens=length,
                    temperature=temperature,
                    device=device,
                )
            st.session_state["generated_sequences"] = [
                (f"PlasmidGPT_generate_{index}", sequence)
                for index, sequence in enumerate(generated, start=1)
            ]
            st.success("Generation complete.")
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            st.error(str(exc))
    render_generated_results()


def render_prediction_results(task_name: str, names: list[str], predictions) -> None:
    st.markdown("### Prediction results")
    flat_rows: list[dict[str, object]] = []
    for sequence_index, (name, rows) in enumerate(zip(names, predictions), start=1):
        st.markdown(f"**{name or f'Sequence {sequence_index}'}**")
        frame = pd.DataFrame(rows, columns=[task_name, "Probability"])
        frame["Confidence"] = frame["Probability"].map(lambda value: f"{value:.1%}")
        display = frame[[task_name, "Confidence"]].copy()
        display.index = np.arange(1, len(display) + 1)
        st.dataframe(display, width="stretch")
        chart = frame.set_index(task_name)[["Probability"]]
        st.bar_chart(chart, horizontal=True, color="#0b7a66")
        for rank, (label, probability) in enumerate(rows, start=1):
            flat_rows.append(
                {
                    "sequence": name,
                    "rank": rank,
                    "prediction": label,
                    "probability": probability,
                }
            )
    export = pd.DataFrame(flat_rows).to_csv(index=False, sep="\t").encode("utf-8")
    st.download_button(
        "Download predictions",
        data=export,
        file_name="plasmidgpt_predictions.tsv",
        mime="text/tab-separated-values",
        type="primary",
    )


def render_predict() -> None:
    st.markdown("## Predict sequence attributes")
    st.markdown(
        '<p class="section-intro">Paste DNA or upload FASTA, choose what to predict, and get ranked results.</p>',
        unsafe_allow_html=True,
    )
    task_name = st.selectbox(
        "What would you like to predict?", list(PREDICTION_TASKS), index=0
    )
    st.caption(PREDICTION_TASKS[task_name].description)
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.text_area(
            "DNA sequence",
            key="prediction_dna",
            height=220,
            placeholder="Paste DNA here…",
        )
        st.button(
            "Use an example",
            on_click=set_sample,
            args=("prediction_dna",),
            key="prediction_sample",
        )
    with right:
        uploaded = st.file_uploader(
            "Or upload FASTA",
            type=["fasta", "fa", "fas", "txt"],
            key="prediction_upload",
        )
        top_n = st.slider("Show top results", 3, 15, 5)
        st.info("Long sequences are safely limited to the model's 2,048-token context.")

    if st.button("Run prediction", type="primary", width="stretch"):
        try:
            records = input_sequences(st.session_state.get("prediction_dna", ""), uploaded)
            names = [name for name, _ in records]
            sequences = [sequence for _, sequence in records]
            with st.spinner("Calculating embeddings and ranking predictions…"):
                model, tokenizer, device = foundation_assets()
                embeddings = calculate_embeddings(
                    model, tokenizer, sequences, device=device
                )
                classifier, labels, _ = prediction_assets(task_name)
                predictions = predict_attributes(
                    embeddings,
                    classifier,
                    labels,
                    top_n=top_n,
                    device=device,
                )
            st.session_state["prediction_result"] = (
                task_name,
                names,
                predictions,
            )
            st.success("Prediction complete.")
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            st.error(str(exc))

    saved = st.session_state.get("prediction_result")
    if saved:
        render_prediction_results(*saved)


def render_embeddings() -> None:
    st.markdown("## Export embeddings")
    st.markdown(
        '<p class="section-intro">Convert one or more DNA sequences into 768-value numerical representations for downstream analysis.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.text_area(
            "DNA or FASTA text",
            key="embedding_dna",
            height=220,
            placeholder="Paste a sequence or a multi-record FASTA document…",
        )
        st.button(
            "Use an example",
            on_click=set_sample,
            args=("embedding_dna",),
            key="embedding_sample",
        )
    with right:
        uploaded = st.file_uploader(
            "Or upload FASTA",
            type=["fasta", "fa", "fas", "txt"],
            key="embedding_upload",
        )
        st.markdown(
            '<div class="callout">Each input sequence becomes one row with 768 numerical values.</div>',
            unsafe_allow_html=True,
        )

    if st.button("Create embeddings", type="primary", width="stretch"):
        try:
            records = input_sequences(st.session_state.get("embedding_dna", ""), uploaded)
            with st.spinner("Creating embeddings…"):
                model, tokenizer, device = foundation_assets()
                embeddings = calculate_embeddings(
                    model,
                    tokenizer,
                    [sequence for _, sequence in records],
                    device=device,
                )
            output = StringIO()
            np.savetxt(output, embeddings, fmt="%.6f")
            st.session_state["embedding_result"] = (
                output.getvalue().encode("utf-8"),
                embeddings.shape,
            )
            st.success(
                f"Created {embeddings.shape[0]} embedding row(s), each with {embeddings.shape[1]} values."
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            st.error(str(exc))

    result = st.session_state.get("embedding_result")
    if result:
        data, shape = result
        metric_a, metric_b, _ = st.columns([1, 1, 2])
        metric_a.metric("Sequences", shape[0])
        metric_b.metric("Values each", shape[1])
        st.download_button(
            "Download embeddings",
            data=data,
            file_name="plasmidgpt_embeddings.txt",
            mime="text/plain",
            type="primary",
        )


def render_help() -> None:
    st.markdown("## Quick guide")
    st.markdown(
        """
        ### Generate
        Paste a short DNA seed, keep the defaults, and select **Generate sequences**.
        Download the result as FASTA or send the first sequence directly to prediction.

        ### Predict
        Choose **Lab of origin**, **Host species**, or **Vector type**. Paste one
        sequence or upload a multi-sequence FASTA file, then select **Run prediction**.

        ### Embeddings
        Add one or more sequences and select **Create embeddings**. The downloaded
        text file has one row of 768 values per sequence.
        """
    )
    st.warning(
        "PlasmidGPT produces computational candidates and predictions. Review all "
        "outputs with appropriate biological validation and biosafety procedures "
        "before experimental use."
    )
    with st.expander("Troubleshooting"):
        st.markdown(
            """
            - Keep the launcher window open while using the app.
            - The first model action takes longer because the model is loading.
            - If GPU memory is low, generate fewer or shorter sequences.
            - FASTA files must be plain text and sequences may use A, T, C, G, or N.
            """
        )
    st.caption(
        "Repository code: MIT license · Pretrained model: CC BY-NC 4.0 "
        "(non-commercial restriction applies)."
    )


issues = installation_issues()
render_status()
if issues:
    st.error(
        "PlasmidGPT cannot start because required files are missing: "
        + ", ".join(issues)
    )
    st.stop()

if "section" not in st.session_state:
    st.session_state["section"] = "Generate"
render_header()
section = st.sidebar.radio(
    "Workspace",
    ["Generate", "Predict", "Embeddings", "Help"],
    key="section",
)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: model loading happens once per session.")

if section == "Generate":
    render_generate()
elif section == "Predict":
    render_predict()
elif section == "Embeddings":
    render_embeddings()
else:
    render_help()
