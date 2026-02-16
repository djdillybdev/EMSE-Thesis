# /// script
# dependencies = [
#     "gradio>=5.0",
#     "fasttext",
#     "numpy<2",
#     "huggingface_hub",
#     "lingua-language-detector",
#     "pymupdf",
#     "pandas",
# ]
# ///

import json

import gradio as gr
import pandas as pd

from detect_foreign_words import (
    LANG_NAMES,
    STRIP_PUNCT_RE,
    SUPPORTED_MODELS,
    detect_language,
    load_input_text,
    load_model,
)

# Dropdown choices: sorted ISO codes
LANG_CODE_CHOICES = sorted(LANG_NAMES.keys())


def build_lang_key(records):
    """Build a markdown string mapping ISO codes to language names for detected languages."""
    if not records:
        return ""
    codes = sorted({r["detected_lang"] for r in records} | {r["document_lang"] for r in records})
    parts = [f"**{code}** = {LANG_NAMES.get(code, code)}" for code in codes]
    return "**Language Key:** " + " · ".join(parts)


def run_detection(doc_model_name, word_model_name, file, output_path, threshold, model_cache, progress=gr.Progress()):
    if file is None:
        raise gr.Error("Please upload a file first.")

    if not output_path.strip():
        output_path = "foreign_words.jsonl"

    if model_cache is None:
        model_cache = {}

    if doc_model_name not in model_cache:
        progress(0, desc=f"Loading doc model: {doc_model_name}")
        model_cache[doc_model_name] = load_model(doc_model_name)

    if word_model_name not in model_cache:
        progress(0, desc=f"Loading word model: {word_model_name}")
        model_cache[word_model_name] = load_model(word_model_name)

    doc_model_type, doc_model = model_cache[doc_model_name]
    word_model_type, word_model = model_cache[word_model_name]

    file_path = file.name if hasattr(file, "name") else str(file)
    text = load_input_text(file_path, "auto")

    progress(0, desc="Detecting document language...")
    doc_lang, doc_confidence = detect_language(text, doc_model_type, doc_model)
    if doc_lang is None:
        doc_lang = "unknown"

    clean_text = text.replace("\n", " ")
    words = clean_text.split()
    total_words = len(words)

    all_records = []
    if doc_lang != "unknown":
        for i, word in enumerate(words):
            if i % 50 == 0:
                progress(i / total_words, desc=f"Analyzing words ({i}/{total_words})")

            normalized_word = STRIP_PUNCT_RE.sub("", word)
            if not normalized_word or normalized_word.isdigit():
                continue

            detected_lang, confidence = detect_language(normalized_word, word_model_type, word_model)
            if detected_lang is None:
                detected_lang = "unknown"
            if threshold > 0 and confidence < threshold:
                continue

            is_foreign = detected_lang != doc_lang
            all_records.append({
                "word": word,
                "normalized_word": normalized_word,
                "index": i,
                "detected_lang": detected_lang,
                "confidence": round(float(confidence), 4),
                "document_lang": doc_lang,
                "is_foreign": is_foreign,
            })

    progress(1.0, desc="Done")

    foreign_count = sum(1 for r in all_records if r["is_foreign"])
    info = (
        f"**Document Language:** {doc_lang} &nbsp;|&nbsp; "
        f"**Confidence:** {float(doc_confidence):.4f}\n\n"
        f"**Total words:** {total_words} &nbsp;|&nbsp; "
        f"**Foreign words found:** {foreign_count}"
    )

    original_langs = {r["index"]: r["detected_lang"] for r in all_records}

    return info, all_records, model_cache, original_langs, clean_text, doc_lang


def build_dataframe(all_records, show_all, min_length):
    """Build a display dataframe from all_records based on current filters."""
    if not all_records:
        return pd.DataFrame(columns=["word", "normalized_word", "index", "detected_lang", "confidence", "document_lang", "is_foreign"])

    records = all_records
    if not show_all:
        records = [r for r in records if r["is_foreign"]]
    if min_length > 1:
        records = [r for r in records if len(r["normalized_word"]) >= min_length]

    if not records:
        return pd.DataFrame(columns=["word", "normalized_word", "index", "detected_lang", "confidence", "document_lang", "is_foreign"])

    return pd.DataFrame(records)


def build_highlights(all_records, full_text, doc_lang, show_all, min_length, enabled_langs=None):
    """Build HighlightedText data from all_records and full document text."""
    if not full_text or not all_records:
        return []

    words = full_text.split()
    # Build lookup: word index -> record
    record_map = {}
    for r in all_records:
        should_show = show_all or r["is_foreign"]
        long_enough = len(r["normalized_word"]) >= min_length
        if should_show and long_enough:
            record_map[r["index"]] = r

    highlights = []
    for i, word in enumerate(words):
        if i in record_map:
            r = record_map[i]
            # If enabled_langs filter is active, only highlight enabled languages
            if enabled_langs is not None and r["detected_lang"] not in enabled_langs:
                highlights.append((word + " ", None))
            else:
                highlights.append((word + " ", r["detected_lang"]))
        else:
            highlights.append((word + " ", None))

    return highlights


def get_foreign_languages(all_records):
    """Extract unique foreign language codes from records."""
    return sorted({r["detected_lang"] for r in all_records if r["is_foreign"]})


def build_app():
    with gr.Blocks(title="Foreign Word Detection Tool") as app:
        gr.Markdown("# Foreign Word Detection Tool")

        # -- State --
        model_cache = gr.State(value={})
        original_langs = gr.State(value={})
        all_records_state = gr.State(value=[])
        full_text_state = gr.State(value="")
        doc_lang_state = gr.State(value="")
        selected_row_idx = gr.State(value=None)  # index into all_records

        # -- Model / File inputs --
        with gr.Row():
            doc_model = gr.Dropdown(choices=SUPPORTED_MODELS, value="fasttext-lid.176", label="Document Model")
            word_model = gr.Dropdown(choices=SUPPORTED_MODELS, value="fasttext-lid.176", label="Word Model")

        with gr.Row():
            file_input = gr.File(label="Upload Document (.txt / .pdf)", file_types=[".txt", ".pdf"])
            output_path = gr.Textbox(value="foreign_words.jsonl", label="Output Path")

        threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Confidence Threshold (0 = no filter)")

        run_btn = gr.Button("Run Detection", variant="primary")

        doc_info = gr.Markdown(value="*Upload a file and click Run Detection*")

        # -- Toggles --
        with gr.Row():
            show_all_cb = gr.Checkbox(value=False, label="Show all words")
            min_length_num = gr.Number(value=3, label="Min word length", minimum=1, maximum=50, precision=0)

        lang_key = gr.Markdown(value="")

        # -- Tabs --
        with gr.Tabs():
            with gr.Tab("Table View"):
                with gr.Row():
                    with gr.Column(scale=3):
                        results_df = gr.Dataframe(
                            interactive=False,
                            label="Detection Results",
                            wrap=True,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Correction Panel")
                        tv_correction_word = gr.Textbox(label="Selected Word", interactive=False)
                        tv_correction_dropdown = gr.Dropdown(choices=LANG_CODE_CHOICES, label="Correct Language", interactive=True)
                        tv_apply_btn = gr.Button("Apply Correction", variant="secondary")

            with gr.Tab("Document View"):
                with gr.Row():
                    with gr.Column(scale=3):
                        lang_filter = gr.CheckboxGroup(
                            choices=[], value=[], label="Highlighted Languages (uncheck to hide)",
                        )
                        highlighted_text = gr.HighlightedText(
                            label="Document with language annotations",
                            combine_adjacent=False,
                            show_legend=True,
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Correction Panel")
                        dv_correction_word = gr.Textbox(label="Selected Word", interactive=False)
                        dv_correction_dropdown = gr.Dropdown(choices=LANG_CODE_CHOICES, label="Correct Language", interactive=True)
                        dv_apply_btn = gr.Button("Apply Correction", variant="secondary")

        # -- Save --
        with gr.Row():
            save_btn = gr.Button("Save Results")
            save_status = gr.Markdown()

        # =====================================================================
        # Event handlers
        # =====================================================================

        def on_run(doc_model_name, word_model_name, file, output_path_val, thresh, cache, show_all, min_len):
            info, records, cache, orig_langs, full_text, doc_lang = run_detection(
                doc_model_name, word_model_name, file, output_path_val, thresh, cache
            )
            foreign_langs = get_foreign_languages(records)
            lang_choices = [f"{LANG_NAMES.get(c, c)} ({c})" for c in foreign_langs]
            df = build_dataframe(records, show_all, min_len)
            highlights = build_highlights(records, full_text, doc_lang, show_all, min_len, enabled_langs=set(foreign_langs))
            return (
                info, df, cache, orig_langs, records, full_text, doc_lang,
                highlights,
                gr.update(choices=lang_choices, value=lang_choices),  # lang_filter: all checked
                build_lang_key(records),
                # Reset both correction panels
                "", gr.update(value=None),  # table view
                "", gr.update(value=None),  # document view
                None,  # selected_row_idx
            )

        run_btn.click(
            fn=on_run,
            inputs=[doc_model, word_model, file_input, output_path, threshold, model_cache, show_all_cb, min_length_num],
            outputs=[
                doc_info, results_df, model_cache, original_langs,
                all_records_state, full_text_state, doc_lang_state,
                highlighted_text,
                lang_filter,
                lang_key,
                tv_correction_word, tv_correction_dropdown,
                dv_correction_word, dv_correction_dropdown,
                selected_row_idx,
            ],
        )

        # -- Helper to parse lang codes from filter checkbox labels --
        def _filter_to_codes(filter_values):
            """Extract ISO codes from checkbox labels like 'French (fr)'."""
            codes = set()
            for val in filter_values:
                # Extract code from "Name (code)" format
                if "(" in val and val.endswith(")"):
                    code = val.rsplit("(", 1)[1].rstrip(")")
                    codes.add(code)
                else:
                    codes.add(val)
            return codes

        # -- Toggle handlers: re-render dataframe + highlights --
        def on_toggle(records, full_text, doc_lang, show_all, min_len, filter_vals):
            df = build_dataframe(records, show_all, min_len)
            enabled = _filter_to_codes(filter_vals) if filter_vals else None
            highlights = build_highlights(records, full_text, doc_lang, show_all, min_len, enabled_langs=enabled)
            return df, highlights

        for toggle_input in [show_all_cb, min_length_num]:
            toggle_input.change(
                fn=on_toggle,
                inputs=[all_records_state, full_text_state, doc_lang_state, show_all_cb, min_length_num, lang_filter],
                outputs=[results_df, highlighted_text],
            )

        # -- Language filter change: re-render highlights only --
        def on_lang_filter(records, full_text, doc_lang, show_all, min_len, filter_vals):
            enabled = _filter_to_codes(filter_vals) if filter_vals else set()
            highlights = build_highlights(records, full_text, doc_lang, show_all, min_len, enabled_langs=enabled)
            return highlights

        lang_filter.change(
            fn=on_lang_filter,
            inputs=[all_records_state, full_text_state, doc_lang_state, show_all_cb, min_length_num, lang_filter],
            outputs=[highlighted_text],
        )

        # -- Row selection in dataframe -> update BOTH correction panels --
        def on_row_select(evt: gr.SelectData, records, show_all, min_len):
            row_idx = evt.index[0]
            visible = records if show_all else [r for r in records if r["is_foreign"]]
            if min_len > 1:
                visible = [r for r in visible if len(r["normalized_word"]) >= min_len]
            if row_idx >= len(visible):
                return "", gr.update(), "", gr.update(), None
            record = visible[row_idx]
            word_display = f"{record['word']} (index {record['index']})"
            current_lang = record["detected_lang"]
            return word_display, gr.update(value=current_lang), word_display, gr.update(value=current_lang), record["index"]

        results_df.select(
            fn=on_row_select,
            inputs=[all_records_state, show_all_cb, min_length_num],
            outputs=[tv_correction_word, tv_correction_dropdown, dv_correction_word, dv_correction_dropdown, selected_row_idx],
        )

        # -- Highlight click in Document View -> update BOTH correction panels --
        def on_highlight_select(evt: gr.SelectData, records, show_all, min_len):
            span_idx = evt.index
            record = None
            for r in records:
                if r["index"] == span_idx:
                    record = r
                    break
            if record is None:
                return "", gr.update(), "", gr.update(), None
            word_display = f"{record['word']} (index {record['index']})"
            current_lang = record["detected_lang"]
            return word_display, gr.update(value=current_lang), word_display, gr.update(value=current_lang), record["index"]

        highlighted_text.select(
            fn=on_highlight_select,
            inputs=[all_records_state, show_all_cb, min_length_num],
            outputs=[tv_correction_word, tv_correction_dropdown, dv_correction_word, dv_correction_dropdown, selected_row_idx],
        )

        # -- Apply correction (shared logic for both panels) --
        def on_apply(records, sel_idx, lang_choice, show_all, min_len, full_text, doc_lang, filter_vals):
            if sel_idx is None or lang_choice is None:
                raise gr.Error("Select a word first, then choose a language.")
            new_code = lang_choice
            updated_records = []
            target_word = None
            for r in records:
                if r["index"] == sel_idx:
                    target_word = r["normalized_word"].lower()
                    break
            for r in records:
                if target_word is not None and r["normalized_word"].lower() == target_word:
                    r = dict(r)
                    r["detected_lang"] = new_code
                    r["is_foreign"] = new_code != r["document_lang"]
                updated_records.append(r)
            # Update lang filter choices (languages may have changed)
            foreign_langs = get_foreign_languages(updated_records)
            lang_choices = [f"{LANG_NAMES.get(c, c)} ({c})" for c in foreign_langs]
            # Keep currently checked languages that still exist
            enabled = _filter_to_codes(filter_vals) if filter_vals else set()
            new_filter_value = [lc for lc in lang_choices if any(c in enabled for c in [lc.rsplit("(", 1)[1].rstrip(")")])]
            enabled_after = _filter_to_codes(new_filter_value) if new_filter_value else set()
            df = build_dataframe(updated_records, show_all, min_len)
            highlights = build_highlights(updated_records, full_text, doc_lang, show_all, min_len, enabled_langs=enabled_after)
            return updated_records, df, highlights, gr.update(choices=lang_choices, value=new_filter_value), build_lang_key(updated_records)

        tv_apply_btn.click(
            fn=on_apply,
            inputs=[all_records_state, selected_row_idx, tv_correction_dropdown, show_all_cb, min_length_num, full_text_state, doc_lang_state, lang_filter],
            outputs=[all_records_state, results_df, highlighted_text, lang_filter, lang_key],
        )
        dv_apply_btn.click(
            fn=on_apply,
            inputs=[all_records_state, selected_row_idx, dv_correction_dropdown, show_all_cb, min_length_num, full_text_state, doc_lang_state, lang_filter],
            outputs=[all_records_state, results_df, highlighted_text, lang_filter, lang_key],
        )

        # -- Save --
        def save_with_corrections(records, output_path_val, orig_langs_val):
            if not records:
                raise gr.Error("No results to save. Run detection first.")

            if not output_path_val.strip():
                output_path_val = "foreign_words.jsonl"

            # Only save foreign words (or all if user wants — save all for completeness)
            with open(output_path_val, "w", encoding="utf-8") as f:
                for record in records:
                    idx = record["index"]
                    original_lang = orig_langs_val.get(idx, record["detected_lang"])
                    # Use string key fallback since State may serialize int keys as strings
                    if original_lang == record["detected_lang"]:
                        original_lang = orig_langs_val.get(str(idx), record["detected_lang"])
                    corrected = record["detected_lang"] != original_lang
                    out = {
                        "word": record["word"],
                        "normalized_word": record["normalized_word"],
                        "index": idx,
                        "detected_lang": original_lang,
                        "confidence": float(record["confidence"]),
                        "document_lang": record["document_lang"],
                        "is_foreign": record["is_foreign"],
                        "corrected_lang": record["detected_lang"],
                        "corrected": corrected,
                    }
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

            return f"Saved {len(records)} records to `{output_path_val}`"

        save_btn.click(
            fn=save_with_corrections,
            inputs=[all_records_state, output_path, original_langs],
            outputs=[save_status],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
