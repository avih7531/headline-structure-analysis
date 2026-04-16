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
from typing import Dict

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


@dataclass
class CorpusBaselines:
    avg_tokens: float
    avg_density: float
    avg_entities: float
    actor_first_pct: float


def _load_corpus_baselines(path: str = "data/headlines_parsed.json") -> CorpusBaselines:
    if not os.path.exists(path):
        return CorpusBaselines(avg_tokens=0.0, avg_density=0.0, avg_entities=0.0, actor_first_pct=0.0)

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    df = pd.DataFrame(data)
    if df.empty:
        return CorpusBaselines(avg_tokens=0.0, avg_density=0.0, avg_entities=0.0, actor_first_pct=0.0)

    densities = []
    actor_first = 0
    for _, row in df.iterrows():
        tokens = [t for t in row.get("tokens", []) if not t.get("is_punct", False)]
        if tokens and tokens[0].get("pos") in {"NOUN", "PROPN", "PRON"}:
            actor_first += 1
        content = sum(1 for t in tokens if t.get("pos") in CONTENT_POS)
        densities.append(content / len(tokens) if tokens else 0.0)

    avg_entities = sum(len(r.get("entities", [])) for _, r in df.iterrows()) / len(df)
    return CorpusBaselines(
        avg_tokens=float(df["num_tokens"].mean()),
        avg_density=float(sum(densities) / len(densities)),
        avg_entities=float(avg_entities),
        actor_first_pct=float(actor_first / len(df) * 100),
    )


def _headline_stats(record: Dict) -> Dict[str, float]:
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


class HeadlineLiveApp(App):
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
    }
    #warning_panel {
      background: #242933;
      border: round #b74e58;
    }
    Header, Footer {
      background: #434C5E;
      color: #e279a1;
    }
    """

    TITLE = "Headline Style Workbench"
    SUB_TITLE = "Real-time structure + style prediction"

    def __init__(self) -> None:
        super().__init__()
        self.nlp = None
        self.baselines = _load_corpus_baselines()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Type a headline... updates on every keystroke", id="headline_input")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Static("Waiting for input...", classes="panel", id="prediction_panel")
                yield Static("Waiting for input...", classes="panel", id="stats_panel")
            with Vertical(id="right"):
                yield Static("Waiting for input...", classes="panel", id="comparison_panel")
                yield Static("Waiting for input...", classes="panel", id="warning_panel")
        yield Footer()

    def on_mount(self) -> None:
        self.nlp = load_spacy_model()
        self._render_empty_state()

    def _render_empty_state(self) -> None:
        self.query_one("#prediction_panel", Static).update(
            "[b #ffffff on #88C0D0] Predictions [/b #ffffff on #88C0D0]\nType a headline to start."
        )
        self.query_one("#stats_panel", Static).update(
            "[b #ffffff on #5E81AC] Headline Stats [/b #ffffff on #5E81AC]\n-"
        )
        self.query_one("#comparison_panel", Static).update(
            "[b #ffffff on #a97ea1] Corpus Comparison [/b #ffffff on #a97ea1]\n-"
        )
        self.query_one("#warning_panel", Static).update(
            "[b #ffffff on #b74e58] Live Warnings [/b #ffffff on #b74e58]\n"
            "Type a headline to see instant alerts."
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        text = event.value
        if not text.strip():
            self._render_empty_state()
            return

        parsed = parse_headline(text, self.nlp)
        record = {"headline": text, **parsed}
        profile = profile_record(record)
        stats = _headline_stats(record)

        self._render_predictions(profile)
        self._render_stats(stats, parsed)
        self._render_comparison(profile, stats)
        self._render_warnings(profile, stats)

    def _render_predictions(self, profile: Dict) -> None:
        panel = self.query_one("#prediction_panel", Static)
        panel.update(
            "[b #ffffff on #88C0D0] Predictions [/b #ffffff on #88C0D0]\n"
            f"- structure: [#e7c173]{profile['predicted_structure']}[/#e7c173]\n"
            f"- lead_frame: [#88C0D0]{profile['lead_frame']}[/#88C0D0]\n"
            f"- agency_style: [#97b67c]{profile['agency_style']}[/#97b67c]\n"
            f"- density_band: [#e7c173]{profile['density_band']}[/#e7c173]\n"
            f"- rhetorical_mode: [#e279a1]{profile['rhetorical_mode']}[/#e279a1]\n"
            f"- signature: [#a97ea1]{profile['style_signature']}[/#a97ea1]"
        )

    def _render_stats(self, stats: Dict, parsed: Dict) -> None:
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
            f"- actor-first baseline: [#e7c173]{self.baselines.actor_first_pct:.1f}%[/#e7c173]\n\n"
            f"[b #ffffff on #5E81AC] Instant editorial read [/b #ffffff on #5E81AC]\n"
            f"- opening: [#88C0D0]{profile['lead_frame']}[/#88C0D0]\n"
            f"- tone mode: [#e279a1]{profile['rhetorical_mode']}[/#e279a1]\n"
            f"- agency: [#97b67c]{profile['agency_style']}[/#97b67c]"
        )

    def _render_warnings(self, profile: Dict, stats: Dict) -> None:
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

        panel = self.query_one("#warning_panel", Static)
        panel.update("[b #ffffff on #b74e58] Live Warnings [/b #ffffff on #b74e58]\n" + "\n".join(warnings))


if __name__ == "__main__":
    HeadlineLiveApp().run()
