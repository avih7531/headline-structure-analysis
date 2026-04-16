"""
Real-time headline style workbench (Textual TUI).

Type a headline and get live structural predictions + style statistics
on every character change.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
# pylint: disable=import-error
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Static

# Ensure project root importability.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.model.headline_style_profiler import profile_record
from scripts.pipeline.parse_headlines import load_spacy_model, parse_headline


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}


class HeadlineInput(Input):
    """Input widget with a local Ctrl+X clear shortcut."""

    def key_ctrl_x(self) -> None:
        """Clear the current input buffer when Ctrl+X is pressed."""
        self.value = ""
        self.focus()


@dataclass
class CorpusBaselines:
    """Corpus-level reference metrics used for live headline comparison."""

    avg_tokens: float
    avg_density: float
    avg_entities: float
    actor_entity_first_pct: float


@dataclass
class BenchmarkStats:
    """Evaluation metrics surfaced in the UI footer/status line."""

    validation_macro_f1: float
    structure_test_accuracy: float
    corpus_size: int


def _load_corpus_baselines(path: str = "data/headlines_parsed.json") -> CorpusBaselines:
    """Load baseline stats from parsed corpus for delta-based UI feedback."""
    if not os.path.exists(path):
        return CorpusBaselines(avg_tokens=0.0, avg_density=0.0, avg_entities=0.0, actor_entity_first_pct=0.0)

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    df = pd.DataFrame(data)
    if df.empty:
        return CorpusBaselines(avg_tokens=0.0, avg_density=0.0, avg_entities=0.0, actor_entity_first_pct=0.0)

    densities = []
    actor_entity_first = 0
    for _, row in df.iterrows():
        tokens = [t for t in row.get("tokens", []) if not t.get("is_punct", False)]
        profile = profile_record(row.to_dict())
        if profile.get("lead_frame") == "actor_entity_first":
            actor_entity_first += 1
        content = sum(1 for t in tokens if t.get("pos") in CONTENT_POS)
        densities.append(content / len(tokens) if tokens else 0.0)

    avg_entities = sum(len(r.get("entities", [])) for _, r in df.iterrows()) / len(df)
    return CorpusBaselines(
        avg_tokens=float(df["num_tokens"].mean()),
        avg_density=float(sum(densities) / len(densities)),
        avg_entities=float(avg_entities),
        actor_entity_first_pct=float(actor_entity_first / len(df) * 100),
    )


def _load_benchmark_stats(
    parsed_path: str = "data/headlines_parsed.json",
    structure_eval_path: str = "data/evaluation/classifier_eval.json",
) -> BenchmarkStats:
    """Load benchmark metrics from evaluation outputs for TUI display."""
    corpus_size = 0
    if os.path.exists(parsed_path):
        with open(parsed_path, "r", encoding="utf-8") as handle:
            corpus_size = len(json.load(handle))

    validation_macro_f1 = 0.0
    structure_test_accuracy = 0.0
    if os.path.exists(structure_eval_path):
        with open(structure_eval_path, "r", encoding="utf-8") as handle:
            eval_report = json.load(handle)
        validation_macro_f1 = float(eval_report.get("dev", {}).get("rule_based", {}).get("macro_f1", 0.0))
        structure_test_accuracy = float(eval_report.get("test", {}).get("rule_based", {}).get("accuracy", 0.0))

    return BenchmarkStats(
        validation_macro_f1=validation_macro_f1,
        structure_test_accuracy=structure_test_accuracy,
        corpus_size=corpus_size,
    )


def _headline_stats(record: Dict) -> Dict[str, float]:
    """Compute real-time lexical/NER stats for a single parsed headline record."""
    tokens = [t for t in record.get("tokens", []) if not t.get("is_punct", False)]
    content = sum(1 for t in tokens if t.get("pos") in CONTENT_POS)
    density = (content / len(tokens)) if tokens else 0.0
    entities = len(record.get("entities", []))
    proper_nouns = sum(1 for t in tokens if t.get("pos") == "PROPN")
    propn_ratio = (proper_nouns / len(tokens)) if tokens else 0.0
    return {
        "num_tokens": record.get("num_tokens", 0),
        "density": density,
        "entities": entities,
        "propn_ratio": propn_ratio,
    }


def _clamp01(value: float) -> float:
    """Clamp value into [0, 1] range."""
    return max(0.0, min(1.0, value))


def _compute_confidences(profile: Dict, stats: Dict, parsed: Dict, headline: str) -> Dict[str, float]:
    """Heuristic confidence scores for rule-based predictions."""
    tokens = parsed.get("tokens", [])
    deps = {t.get("dep", "") for t in tokens}
    root_pos = parsed.get("root_pos", "")

    # Structure confidence.
    structure = profile["predicted_structure"]
    if structure == "question_form":
        structure_conf = 0.96 if "?" in headline else 0.88
    elif structure == "passive_clause":
        structure_conf = 0.93 if ("nsubjpass" in deps or "auxpass" in deps) else 0.84
    elif structure == "coordination":
        cue_count = int(" and " in headline.lower()) + int(" but " in headline.lower()) + int(";" in headline)
        structure_conf = _clamp01(0.74 + 0.08 * cue_count)
    elif structure == "noun_phrase_fragment":
        structure_conf = 0.89 if root_pos in {"NOUN", "PROPN", "ADJ"} else 0.8
    elif structure == "simple_clause":
        has_subject = "nsubj" in deps
        has_object = "dobj" in deps or "obj" in deps
        structure_conf = 0.83 + (0.08 if has_subject else 0.0) + (0.05 if has_object else 0.0)
    else:
        structure_conf = 0.58
    structure_conf = _clamp01(structure_conf)

    # Lead frame confidence.
    lead_frame = profile["lead_frame"]
    first = next((t for t in tokens if not t.get("is_punct", False)), {})
    first_pos = first.get("pos", "")
    lead_expect = {
        "actor_entity_first": {"PROPN", "PRON", "NOUN"},
        "event_first": {"NOUN"},
        "action_first": {"VERB", "AUX"},
        "context_first": {"ADV", "ADJ", "NUM", "DET", "ADP"},
    }
    lead_conf = 0.78 if first_pos in lead_expect.get(lead_frame, set()) else 0.6
    if lead_frame == "event_first" and str(first.get("text", "")).lower().endswith("ing"):
        lead_conf += 0.12
    lead_conf = _clamp01(lead_conf)

    # Agency confidence.
    agency = profile["agency_style"]
    if agency == "active_or_nonpassive":
        agency_conf = 0.9 if profile["predicted_structure"] != "passive_clause" else 0.65
    elif agency == "passive_with_agent":
        agency_conf = 0.92 if ("agent" in deps or "by " in headline.lower()) else 0.7
    else:
        agency_conf = 0.87

    # Density confidence (distance from threshold boundaries).
    density = stats["density"]
    if density >= 0.70:
        margin = density - 0.70
    elif density >= 0.50:
        margin = min(density - 0.50, 0.70 - density)
    else:
        margin = 0.50 - density
    density_conf = _clamp01(0.65 + min(margin * 2.5, 0.3))

    # Rhetorical confidence from cue strength.
    rhetorical = profile["rhetorical_mode"]
    low = headline.lower()
    if rhetorical == "question_hook":
        rhetorical_conf = 0.95 if "?" in headline else 0.86
    elif rhetorical == "live_or_alert":
        cue_hits = sum(int(cue in low) for cue in ("live", "breaking", "updates", "update"))
        rhetorical_conf = _clamp01(0.74 + 0.08 * cue_hits)
    elif rhetorical == "analysis_explainer":
        cue_hits = sum(int(cue in low) for cue in ("analysis", "explainer", "what to know", "why ", "how "))
        rhetorical_conf = _clamp01(0.7 + 0.08 * cue_hits)
    else:
        rhetorical_conf = 0.82

    return {
        "structure": structure_conf,
        "lead_frame": lead_conf,
        "agency_style": _clamp01(agency_conf),
        "density_band": density_conf,
        "rhetorical_mode": _clamp01(rhetorical_conf),
    }


def _parse_evidence(parsed: Dict) -> Tuple[str, str, str]:
    """Build left-to-right phrase/dependency evidence strings covering all tokens."""
    tokens = [t for t in parsed.get("tokens", []) if not t.get("is_punct", False)]
    if not tokens:
        return "NP: - | VP: - | NP: -", "ROOT: - | nsubj: - | dobj: -", "tokens: -"

    subj_tokens = [t.get("text", "") for t in tokens if t.get("dep") in {"nsubj", "nsubjpass", "csubj"}]
    obj_tokens = [t.get("text", "") for t in tokens if t.get("dep") in {"dobj", "obj", "pobj"}][:4]
    verb_tokens = [t.get("text", "") for t in tokens if t.get("pos") in {"VERB", "AUX"}][:3]

    np1 = " ".join(subj_tokens) if subj_tokens else str(tokens[0].get("text", ""))
    vp = " ".join(verb_tokens) if verb_tokens else str(parsed.get("root", "-"))
    np2 = " ".join(obj_tokens) if obj_tokens else "-"
    # Avoid [] tokens because Rich markup treats them as tags.
    template = f"NP: {np1} | VP: {vp} | NP: {np2}"

    root = parsed.get("root", "-")
    nsubj = subj_tokens[0] if subj_tokens else "-"
    dobj = obj_tokens[0] if obj_tokens else "-"
    deps = f"ROOT: {root} | nsubj: {nsubj} | dobj: {dobj}"

    # Full token map for transparency over all words/phrases.
    token_flow_parts = []
    for tok in tokens:
        text = str(tok.get("text", ""))
        pos = str(tok.get("pos", ""))
        dep = str(tok.get("dep", ""))
        token_flow_parts.append(f"{text}/{pos}:{dep}")
    token_flow = " | ".join(token_flow_parts) if token_flow_parts else "-"

    return template, deps, token_flow


class HeadlineLiveApp(App):
    """Interactive Textual app for live headline profiling and diagnostics."""

    CSS = """
    Screen {
      layout: vertical;
      background: #242933;
      color: #E5E9F0;
    }
    #main {
      height: 1fr;
    }
    #left, #right {
      width: 1fr;
      padding: 1 2;
      height: 1fr;
    }
    Input {
      margin: 1 2;
      background: #434C5E;
      color: #E5E9F0;
      border: round #5E81AC;
    }
    .panel {
      background: #434C5E;
      color: #E5E9F0;
      border: round #a97ea1;
      padding: 1;
      margin-bottom: 1;
    }
    #prediction_panel {
      border: round #88C0D0;
    }
    #stats_panel {
      border: round #5E81AC;
    }
    #comparison_panel {
      border: round #a97ea1;
      height: 1fr;
    }
    #evidence_panel {
      border: round #5E81AC;
      margin: 0 2 1 2;
    }
    #warning_panel {
      background: #242933;
      border: round #b74e58;
      height: 8;
    }
    #prediction_panel, #stats_panel {
      height: 1fr;
    }
    #stack_line {
      margin: 0 2 1 2;
      padding: 0 1;
      color: #E5E9F0;
      background: #434C5E;
      border: round #4C566A;
    }
    Header, Footer {
      background: #434C5E;
      color: #e279a1;
    }
    """

    TITLE = "Headline Style Workbench"
    SUB_TITLE = "Real-time structure + style prediction with parse evidence"
    BINDINGS = [("ctrl+x", "clear_headline", "Clear headline input")]

    def __init__(self) -> None:
        """Initialize application state and precomputed corpus baselines."""
        super().__init__()
        self.nlp = None
        self.baselines = _load_corpus_baselines()
        self.benchmarks = _load_benchmark_stats()

    def compose(self) -> ComposeResult:
        """Compose static layout widgets for the live workbench."""
        yield Header()
        yield HeadlineInput(placeholder="Type a headline... updates on every keystroke", id="headline_input")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Static("Waiting for input...", classes="panel", id="prediction_panel")
                yield Static("Waiting for input...", classes="panel", id="stats_panel")
            with Vertical(id="right"):
                yield Static("Waiting for input...", classes="panel", id="comparison_panel")
                yield Static("Waiting for input...", classes="panel", id="warning_panel")
        yield Static("Waiting for input...", classes="panel", id="evidence_panel")
        yield Static("", id="stack_line")
        yield Footer()

    def on_mount(self) -> None:
        """Load NLP resources once, then render the empty-state panels."""
        self.nlp = load_spacy_model()
        self._render_empty_state()

    def _render_empty_state(self) -> None:
        """Render placeholder panel content before user input is available."""
        self.query_one("#prediction_panel", Static).update(
            "[b #ffffff on #88C0D0] Predictions [/b #ffffff on #88C0D0]\nType a headline to start."
        )
        self.query_one("#stats_panel", Static).update(
            "[b #ffffff on #5E81AC] Headline Stats [/b #ffffff on #5E81AC]\n-"
        )
        self.query_one("#comparison_panel", Static).update(
            "[b #ffffff on #a97ea1] Corpus Comparison [/b #ffffff on #a97ea1]\n-"
        )
        self.query_one("#evidence_panel", Static).update(
            "[b #ffffff on #5E81AC] Parse Evidence [/b #ffffff on #5E81AC]\n-"
        )
        self.query_one("#warning_panel", Static).update(
            "[b #ffffff on #b74e58] Live Warnings [/b #ffffff on #b74e58]\n"
            "Type a headline to see instant alerts."
        )
        self._render_stack_line()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Re-parse and refresh all panels whenever input text changes."""
        text = event.value
        if not text.strip():
            self._render_empty_state()
            return

        parsed = parse_headline(text, self.nlp)
        record = {"headline": text, **parsed}
        profile = profile_record(record)
        stats = _headline_stats(record)
        conf = _compute_confidences(profile, stats, parsed, text)
        template, dep_evidence, token_flow = _parse_evidence(parsed)

        self._render_predictions(profile, conf)
        self._render_stats(stats, parsed)
        self._render_comparison(profile, stats)
        self._render_evidence(template, dep_evidence, token_flow)
        self._render_warnings(profile, stats)
        self._render_stack_line()

    def _render_predictions(self, profile: Dict, conf: Dict) -> None:
        """Render structured style predictions into the left prediction panel."""
        panel = self.query_one("#prediction_panel", Static)
        panel.update(
            "[b #ffffff on #88C0D0] Predictions [/b #ffffff on #88C0D0]\n"
            f"- structure: [#e7c173]{profile['predicted_structure']} ({conf['structure']:.2f})[/#e7c173]\n"
            f"- lead_frame: [#88C0D0]{profile['lead_frame']} ({conf['lead_frame']:.2f})[/#88C0D0]\n"
            f"- agency_style: [#97b67c]{profile['agency_style']} ({conf['agency_style']:.2f})[/#97b67c]\n"
            f"- density_band: [#e7c173]{profile['density_band']} ({conf['density_band']:.2f})[/#e7c173]\n"
            f"- rhetorical_mode: [#e279a1]{profile['rhetorical_mode']} ({conf['rhetorical_mode']:.2f})[/#e279a1]\n"
            f"- signature: [#a97ea1]{profile['style_signature']}[/#a97ea1]"
        )

    def _render_stats(self, stats: Dict, parsed: Dict) -> None:
        """Render lexical and entity statistics for the current headline."""
        top_entities = ", ".join(e["label"] for e in parsed.get("entities", [])[:4]) or "none"
        panel = self.query_one("#stats_panel", Static)
        panel.update(
            "[b #ffffff on #5E81AC] Headline Stats [/b #ffffff on #5E81AC]\n"
            f"- tokens: [#e7c173]{stats['num_tokens']}[/#e7c173]\n"
            f"- content-word ratio: [#e7c173]{stats['density']*100:.1f}%[/#e7c173]\n"
            f"- entities: [#88C0D0]{stats['entities']} ({top_entities})[/#88C0D0]\n"
            f"- proper-noun ratio: [#e7c173]{stats['propn_ratio']*100:.1f}%[/#e7c173]\n"
            f"- root: [#97b67c]{parsed.get('root')} ({parsed.get('root_pos')})[/#97b67c]"
        )

    def _render_comparison(self, profile: Dict, stats: Dict) -> None:
        """Render baseline deltas and high-level editorial interpretation."""
        token_delta = stats["num_tokens"] - self.baselines.avg_tokens
        density_delta = (stats["density"] - self.baselines.avg_density) * 100
        ent_delta = stats["entities"] - self.baselines.avg_entities

        panel = self.query_one("#comparison_panel", Static)
        panel.update(
            "[b #ffffff on #a97ea1] Corpus Comparison [/b #ffffff on #a97ea1]\n"
            f"- avg tokens baseline: [#e7c173]{self.baselines.avg_tokens:.1f}[/#e7c173] (delta [#e7c173]{token_delta:+.1f}[/#e7c173])\n"
            f"- avg density baseline: [#e7c173]{self.baselines.avg_density*100:.1f}%[/#e7c173] "
            f"(delta [#e7c173]{density_delta:+.1f} pp[/#e7c173])\n"
            f"- avg entities baseline: [#e7c173]{self.baselines.avg_entities:.2f}[/#e7c173] "
            f"(delta [#e7c173]{ent_delta:+.2f}[/#e7c173])\n"
            f"- actor/entity-first baseline: [#e7c173]{self.baselines.actor_entity_first_pct:.1f}%[/#e7c173]\n\n"
            f"[b #ffffff on #5E81AC] Instant editorial read [/b #ffffff on #5E81AC]\n"
            f"- opening: [#88C0D0]{profile['lead_frame']}[/#88C0D0]\n"
            f"- tone mode: [#e279a1]{profile['rhetorical_mode']}[/#e279a1]\n"
            f"- agency: [#97b67c]{profile['agency_style']}[/#97b67c]"
        )

    def _render_evidence(self, template: str, dep_evidence: str, token_flow: str) -> None:
        """Render compact left-to-right parse evidence with full token coverage."""
        panel = self.query_one("#evidence_panel", Static)
        panel.update(
            "[b #ffffff on #5E81AC] Parse Evidence [/b #ffffff on #5E81AC]\n"
            f"- phrase flow: {template}\n"
            f"- dependency mini-view: {dep_evidence}\n"
            f"- token map: {token_flow}"
        )

    def _render_warnings(self, profile: Dict, stats: Dict) -> None:
        """Render rule-driven caution/success badges from live headline signals."""
        warnings: list[str] = []

        token_delta = stats["num_tokens"] - self.baselines.avg_tokens
        density_delta = stats["density"] - self.baselines.avg_density
        entity_delta = stats["entities"] - self.baselines.avg_entities

        if token_delta > 4:
            warnings.append("- [#ffffff on #b74e58] ! [/#ffffff on #b74e58] Longer than corpus norm; may reduce scan speed.")
        elif token_delta < -4:
            warnings.append("- [#ffffff on #e7c173] ! [/#ffffff on #e7c173] Much shorter than norm; may omit key context.")

        if density_delta < -0.12:
            warnings.append("- [#ffffff on #e7c173] ! [/#ffffff on #e7c173] Low information density vs baseline.")
        elif density_delta > 0.12:
            warnings.append("- [#ffffff on #97b67c] + [/#ffffff on #97b67c] Very high density (compact wording).")

        if entity_delta < -1 and stats["entities"] == 0:
            warnings.append("- [#ffffff on #e7c173] ! [/#ffffff on #e7c173] No named entities; credibility anchoring may be weaker.")

        if profile["agency_style"] == "passive_agent_omitted":
            warnings.append("- [#ffffff on #b74e58] ! [/#ffffff on #b74e58] Passive framing hides actor (agent omitted).")

        if profile["rhetorical_mode"] == "live_or_alert":
            warnings.append("- [#ffffff on #e7c173] ! [/#ffffff on #e7c173] Live/alert mode detected; verify urgency is justified.")

        if profile["predicted_structure"] == "other":
            warnings.append("- [#ffffff on #e7c173] ! [/#ffffff on #e7c173] Structure is ambiguous; headline may parse inconsistently.")

        if not warnings:
            warnings.append("- [#ffffff on #97b67c] + [/#ffffff on #97b67c] No major warnings relative to corpus baseline.")

        # Keep warnings panel compact and predictable.
        warnings = warnings[:4]
        panel = self.query_one("#warning_panel", Static)
        panel.update("[b #ffffff on #b74e58] Live Warnings [/b #ffffff on #b74e58]\n" + "\n".join(warnings))

    def _render_stack_line(self) -> None:
        """Render technical stack and benchmark footer line."""
        line = self.query_one("#stack_line", Static)
        line.update(
            "spaCy dependency parse + rule-based classifier + corpus z-score comparison"
            f" | Validation F1: {self.benchmarks.validation_macro_f1:.3f}"
            f" | Structure accuracy: {self.benchmarks.structure_test_accuracy*100:.1f}%"
            f" | Corpus size: {self.benchmarks.corpus_size:,} headlines"
        )

    def action_clear_headline(self) -> None:
        """Clear headline input quickly via Ctrl+X key binding."""
        input_widget = self.query_one("#headline_input", Input)
        input_widget.value = ""
        input_widget.focus()


if __name__ == "__main__":
    HeadlineLiveApp().run()
