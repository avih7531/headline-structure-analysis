# Structural Patterns in News Headlines: A Computational Linguistic Analysis Using NLP

**Victor Derani, Avi Herman, Kylie Lin**

## 1. Abstract:
We analyze the syntactic and structural patterns of news headlines using NLP. Our pipeline collects headlines from RSS feeds, parses them with spaCy, and feeds the parses into a transparent rule-based classifier. The classifier assigns each headline a primary structural label (question, passive, coordination, noun-phrase fragment, simple clause, or other) plus a four-dimensional style profile (lead frame, agency style, density band, rhetorical mode). Our claim is simple: headlines are not free-form prose but reproducible syntactic products whose form is recoverable, structurally constrained, and stylistically decomposable.

On a manually annotated gold-standard set of 464 headlines, the classifier reaches **0.806 accuracy** and **0.675 macro F1** on the held-out test split. That beats a majority baseline by +16.3 accuracy points (macro F1 0.130 → 0.675) and a random baseline by +34.7 points. A 5-seed split-stability sweep places the structure test macro F1 mean at **0.568 ± 0.076**, making the size of "split luck" explicit. Style dimensions decompose cleanly: framing and density are nearly perfectly recoverable (test macro F1 1.000 / 1.000), while agency and rhetorical mode are harder (0.428 / 0.820).

The takeaway: headline writing follows recoverable rules, and interpretable, parse-grounded methods capture them well — no opaque neural models required. The framework supports summarization, style auditing, and media-framing analysis, and is exposed through an interactive terminal sandbox that updates structure, agency, and rhetorical mode in real time as a user rewrites a headline.

## 2. Introduction:
News headlines are among the most widely read forms of text. Millions of people use them as a primary information source — often to form opinions or decide what to click on — and they do this without reading the underlying article. Despite that reach, headlines are extreme in form: they drop function words, restructure sentences, and compress an event into roughly a dozen words, all while balancing clarity, informativeness, and engagement.

That makes them an interesting computational object. From a linguistic-engineering point of view, a headline is information compression at the limit. From an applied point of view, the same compressed form drives summarization quality, search-result snippets, and — when it slips — misinformation and bias. If we can describe headline form mechanically, we can also flag it mechanically.

The idea that grammatical regularities can be recovered from raw text corpora is itself old. Atwell and Drakos (1987) showed that a first-order Markov model over part-of-speech tags — the engine of their CLAWS tagger on the LOB corpus — can acquire a grammatical classification system from unrestricted English without hand-written grammar rules. Our project sits in that lineage but uses explicit, parse-grounded rules over modern dependency parses, applied specifically to the compressed register of news headlines.

These motivations led us to three concrete research questions:

* What grammatical structures are most common in news headlines, and how do headlines achieve high information density?
* What syntactic templates dominate headline construction, and are these patterns consistent across different news domains?
* Can a transparent, rule-based system recover those structures with reliable out-of-sample performance?

### 2.1 Hypothesis and Falsification Criteria
We adopt an explicit, falsifiable hypothesis up front so that the rest of the paper can be read as a direct test of it:

> **H1:** News headlines follow stable, reusable syntactic templates that can be predicted from POS and dependency signals at rates significantly above trivial baselines.

To make H1 falsifiable rather than rhetorical, we define the conditions under which it should be rejected:

| Criterion | Failure Condition |
| :--- | :--- |
| Structural predictability | Test macro F1 collapses toward majority/random baseline behavior |
| Template consistency | No dominant recurring structures appear in the corpus-level distribution |
| Robustness | Large instability across random train/dev/test splits (large seed-sweep variance) |
| Interpretability | Predictions cannot be traced back to explicit parse evidence |

The remainder of the paper checks each of these conditions in turn.

## 3. Solution:
The pipeline has five stages: **Collect, Parse, Analyze, Classify, Evaluate.** Each section below covers one stage end-to-end.

### 3.1 Collect
A Python script pulls headlines from Google News RSS feeds covering both domestic and world news. For each RSS item, we keep the raw title and then lightly normalize it: fix Unicode and escape-sequence quirks, split on dashes and pipes, and filter out boilerplate keywords. The output is a clean headline plus its source and timestamp, ready for parsing.

### 3.2 Parse
We use spaCy to extract tokens, POS tags, dependency trees, and named entities. The parsing stage is designed to be intentionally robust, because off-the-shelf parsers are sensitive to domain shift: Eggleston and O'Connor (2022) showed that parsers fine-tuned on standard Universal Dependencies treebanks (GUM, EWT) lose substantial accuracy when applied to social-media English, with an additional disparity between Mainstream American English and African-American English. Their drift axis is dialectal social media; ours is compressed, title-cased news headlines. The warning generalizes either way: domain shift hurts parser quality, so we build defenses against it.

Concretely, we prefer models in this order: `en_core_web_trf`, then `en_core_web_lg`, `en_core_web_md`, and `en_core_web_sm`. Before parsing, we normalize punctuation and spacing to remove easy noise (smart quotes, escaped sequences, irregular whitespace).

We then run a two-pass parse strategy. The first pass parses the headline as written and scores parse quality. If the score is poor — for example, no clear verb in a long headline or ambiguous casing on what should be a proper noun — we run a fallback parse with light re-casing and lexicon overrides for common headline ambiguities, then keep the better-scoring of the two parses. This costs us a small constant factor in compute and buys us much better behavior on the headlines that off-the-shelf parsers most often mishandle.

### 3.3 Analyze
The Analyze stage derives higher-level features from each parse: collapsed POS patterns, macro phrase structures (NP, VP, MOD), opening and ending patterns, voice, and a coarse headline type. On top of those features, a suite of analysis functions reports descriptive statistics across the corpus — entity-type counts, content-word density, voice distribution, common dependency templates, length patterns, and the most frequent root verbs and openings, among others. A `generate_insights()` step rolls these into short corpus-level summaries.

This stage describes; it does not assign labels. To turn these descriptive patterns into predictions, we transition to Classify.

### 3.4 Classify
The Classify stage takes the parsed features (POS tags, dependency relations, entities, derived signals) and assigns each headline one primary structural label. The label set is:

1.  **question_form**: interrogative punctuation or opening pattern
2.  **passive_clause**: passive cues (e.g., `nsubjpass`, auxiliary/passive constructions)
3.  **coordination**: multi-clause coordination (`cc`/`conj` and coordination signals)
4.  **noun_phrase_fragment**: nominal fragment with no finite clause
5.  **simple_clause**: canonical finite subject-verb clause
6.  **other**: residual category for out-of-pattern forms

Rules fire in a fixed priority order — question, passive, coordination, NP fragment, simple clause, other — to keep output deterministic. Each rule is a small set of explicit linguistic conditions over POS, dependencies, punctuation, and token order; ambiguous or short headlines fall through to `other`.

Structure alone is too coarse to capture headline variation, so we layer on four style dimensions:

1.  **lead_frame**: actor/entity-first, action-first, context-first, etc.
2.  **agency_style**: active vs. passive (with/without agent)
3.  **density_band**: low/medium/high (based on content-word ratio)
4.  **rhetorical_mode**: straight-report, analysis-explainer, question-hook, live/alert

Each dimension is driven by a distinct signal: entity position for `lead_frame`, passive cues for `agency_style`, content-word ratio for `density_band`, and a mix of structural and lexical cues for `rhetorical_mode`.

The `rhetorical_mode` dimension is not invented from scratch. Bonyadi and Samuel (2013), surveying the headline-function tradition in their contrastive study of editorial headlines, frame headlines as serving three simultaneous roles: summaries of the main event, attention-getting devices, and "relevance optimizers" between story and reader. Our four modes are a parse-grounded operationalization of those competing communicative functions.

The output of Classify is a multi-dimensional label per headline, written as a compact "headline signature" of the form `structure | lead_frame | rhetorical_mode`. This labeled dataset is what we evaluate next.

#### Manual tagging in practice
Two representative manual decisions, to make the annotation protocol concrete and to show why human-in-the-loop labeling is needed at all. The *gold* label requires reading a headline the way a competent English speaker would, which a purely surface-pattern matcher can't always do.

1. Headline: *"Nine killed in second Turkish school shooting in two days"*
    - Manual `gold_label`: **passive_clause**
    - Manual `gold_rhetorical_mode`: **straight_report**
    - Rationale: This is a compressed passive event frame (`NUM + VBN`) where the auxiliary "were" is elided and the agent is omitted entirely. A naive surface match might call this a noun phrase fragment, but the underlying clause is unambiguously passive.
2. Headline: *"Iran Update Special Report, April 14, 2026 Institute for the Study of War"*
    - Manual `gold_label`: **noun_phrase_fragment**
    - Manual `gold_rhetorical_mode`: **analysis_explainer**
    - Rationale: There is no finite main clause; the entire string functions as a report-style nominal title. Its discourse function is explanatory and bulletin-style rather than narrating a single event.

These two examples bracket the easy and hard ends of the annotation space. They also motivate why we evaluate against human gold labels rather than self-agreement: the human decision encodes *why* a headline takes the form it does, and that's what we need to test the model against.

### 3.5 Evaluate
The Evaluate stage compares our predicted labels against a manually annotated gold-standard set. The gold dataset contains **464** hand-labeled headlines covering both structure and style, split **60 / 20 / 20** into Train, Dev, and Test. Train develops the rules. Dev tunes them. Test is touched only at the very end. Keeping Test isolated during rule iteration is what lets us claim the final number isn't a tuning artifact.

We score the system with accuracy, per-class precision/recall/F1, and macro F1. Accuracy is fine when classes are balanced; macro F1 is the metric that exposes whether minority classes are also being handled, which matters here because `simple_clause` dominates.

To put the numbers in context, we also report two trivial baselines:

- **Majority baseline** — always predicts the most common class. If our system isn't beating this on macro F1, it's just exploiting class imbalance.
- **Random baseline** — assigns labels randomly. If we're not beating this, there's no learned structure at all.

Finally, single-split numbers can be misleading: a lucky test split can flatter a system, an unlucky one can sink it. We re-run the entire evaluation across **5 random seeds (13, 42, 87, 123, 202)** and report the mean and standard deviation, so "split luck" becomes a measurable quantity rather than a hand-wave.

Beyond this offline evaluation, we also built an interactive terminal sandbox on top of the same model that shows predictions, confidences, and parse evidence in real time. We defer its full description and screenshot to Section 5.3.

## 4. Proof/Evaluation:
Before turning to the numbers, the metrics we use:

$$\text{accuracy} = \frac{\#\text{correct}}{\#\text{total}} \qquad \text{precision} = \frac{TP}{TP+FP} \qquad \text{recall} = \frac{TP}{TP+FN} \qquad F1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

For a class $c$, **precision** measures how often we're right when we predict $c$ (low false-positive cost), and **recall** measures how often we catch real $c$'s (low false-negative cost). F1 punishes optimizing one at the expense of the other. **Macro F1** is the unweighted average of per-class F1; it is what surfaces minority-class problems that accuracy hides.

To avoid reading too much into one random train/dev/test split, we also run a 5-seed sweep (13, 42, 87, 123, 202) and report the mean and sample standard deviation across those runs:

$$\overline{m} = \frac{1}{K} \sum_{i=1}^{K} m_{i} \qquad s = \sqrt{\frac{1}{K-1} \sum_{i=1}^{K} (m_{i} - \overline{m})^{2}}$$

The mean estimates expected performance under random split variation; $s$ quantifies how sensitive that estimate is to which split we happened to draw.

### 4.1 Corpus-Level Structural Findings
We organize results into four parts: corpus-level structural findings (this section), structure-classifier performance (4.2), style-dimension performance (4.3), and stability analysis (4.4). The first question is the descriptive one: what do news headlines actually look like structurally?

| Metric | Value |
| :--- | :--- |
| Average length | 13.7 tokens |
| Content-word ratio | 73.2% |
| Headlines with verbs | 92.1% |
| Active/non-passive rate | 94.6% |
| Headlines with named entities | 95.7% |
| Avg entities per headline | 2.4 |
| Actor/entity-first openings | 81.6% |

The picture in Table 1 is consistent and tight. The average headline is ~14 tokens, 73% of them content words. 92% contain a verb, so even though they're compressed, they remain *sentences about an action*, not captions. 95% contain at least one named entity (2.4 on average), 95% are active voice, and 82% lead with the main actor or entity. The default newsroom template emerges almost mechanically: **a compressed, active, entity-led clause built around a single event verb.** That alone is evidence for H1's *template consistency* criterion.

This isn't a corpus-specific accident. Robert (2020) applied Halliday's Systemic Functional Grammar to fourteen anti-corruption headlines from two Nigerian dailies and reports that *material processes* — clauses that grammaticalize doing rather than being or thinking — are the dominant process type. The convergence is real even though the lenses differ: she measures transitivity, we measure voice and clause shape, but both characterizations describe action-oriented, agent-led headline syntax as the genre default. The template lives in the genre, not in any single outlet.

![Structure label distribution](../images/structure_label_distribution.png)

**Figure 1** — distribution of predicted structure labels across the corpus. The dominant class is the simple finite clause, with noun-phrase fragments as a clear secondary mode and the rest (questions, passives, coordinations, `other`) as a long tail. The shape itself is a finding: headline writing concentrates on two or three reusable templates, and the rare classes are rare because they fail the brevity constraint that defines the genre.

### 4.2 Structure Classifier Performance
The next question is whether the model can actually predict that template, not just describe it.

| Evaluation Slice | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| All labeled (n=465) | 0.772 | 0.601 |
| Dev (n=90) | — | 0.561 |
| Test (n=98) | 0.806 | 0.675 |

**Table 2** — aggregate structure metrics. The pattern across slices is consistent: macro F1 sits below accuracy because the dataset is imbalanced (`simple_clause` dominates). On the held-out test split — the only number we tune *nothing* against — the classifier reaches **0.806 accuracy** and **0.675 macro F1**. The dev row deliberately omits accuracy because, on that imbalanced slice, accuracy isn't an informative tuning signal.

A test number on its own is hard to read. We compare against two trivial baselines:

| Model | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| Rule-based classifier | 0.806 | 0.675 |
| Majority baseline | 0.643 | 0.130 |
| Random baseline | 0.459 | 0.159 |

**Table 3** — baseline comparison on the held-out test split. Our system beats the majority baseline by **+16.3 accuracy points** and the random baseline by **+34.7 points**. The macro F1 jump is the more telling number: from 0.130 (majority) and 0.159 (random) to 0.675 — more than four times the baseline. Accuracy alone is friendly to a majority predictor on an imbalanced dataset; macro F1 is not, and we clear it decisively. The model is doing real work, not exploiting class frequency.

| Label | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| question_form | 0.960 | 0.960 | 0.960 |
| passive_clause | 0.478 | 0.344 | 0.400 |
| coordination | 0.722 | 0.448 | 0.553 |
| noun_phrase_fragment | 0.578 | 0.627 | 0.602 |
| simple_clause | 0.882 | 0.865 | 0.874 |
| other | 0.138 | 0.500 | 0.216 |

**Table 4** — per-class performance. Strong on the dominant classes (`simple_clause`, `question_form`), weaker on ambiguous ones (`other`, `passive_clause` minority variants). The system captures the newsroom syntactic backbone but stays conservative on edge constructions where label boundaries are inherently fuzzy.

A normal-approximation 95% confidence interval on the test accuracy 0.806 (n = 98) is:

$$\text{CI}_{0.95}(\text{Accuracy}) \approx [0.728,\ 0.884]$$

Both baselines (0.643, 0.459) fall well outside this interval, so the gain is not a sampling artifact. The interval alone is enough to reject H1's *structural predictability* failure condition.

### 4.3 Style-Dimension Performance
Beyond the structural label, can the model also recover stylistic characteristics? We evaluate the four style dimensions twice: across the entire labeled slice, and on the held-out test split alone, so "consistent on training data" and "generalizes" don't get conflated.

| Dimension | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| lead_frame | 0.989 | 0.986 |
| agency_style | 0.935 | 0.622 |
| density_band | 0.994 | 0.995 |
| rhetorical_mode | 0.914 | 0.785 |

**Table 5** — All-slice (n=465). `lead_frame` and `density_band` are nearly perfectly recoverable: headline openings and information density follow stable, surface-level conventions our rules pick up almost every time. `agency_style` is high in accuracy but much lower in macro F1, a textbook symptom of class imbalance — the dominant active class is easy, the rare compressed passives (with elided auxiliaries) are not. `rhetorical_mode` lands in the middle: reliable on the dominant `straight_report` but more uncertain on `analysis_explainer` and edge-case live-alerts, where syntax alone doesn't fully disambiguate discourse function.

| Dimension | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| lead_frame | 1.000 | 1.000 |
| agency_style | 0.908 | 0.428 |
| density_band | 1.000 | 1.000 |
| rhetorical_mode | 0.908 | 0.820 |

**Table 6** — Held-out test (n=98). `lead_frame` and `density_band` saturate the score, consistent with their being deterministic functions of surface features (entity position and content-word ratio). `rhetorical_mode` actually rises slightly compared to the all-slice number, so the rule generalizes rather than overfits. `agency_style` drops noticeably (macro F1 0.622 → 0.428) — in a small held-out slice, a few misclassified rare passives crater the minority-class score. We discuss this in Section 5.2.

![Model performance summary](../images/model_performance_summary.png)

**Figure 2** — model performance summary. Accuracy and macro F1 across the structure classifier and four style dimensions, both all-slice and held-out test. Structure performance is moderate but well above baseline; framing and density saturate; agency and rhetorical mode sit in between, where syntax alone is only partially sufficient.

![Performance heatmap](../images/performance_heatmap.png)

**Figure 3** — performance heatmap. The same picture in colour: form-grounded properties (where the headline starts, how dense it is) are dark green; semantically loaded ones (agency, rhetoric) are lighter; structural classification sits in between. This is the graphical answer to "how recoverable is each dimension of headline form?"

### 4.4 Stability Analysis
Single-split numbers are fragile. A lucky split flatters the model; an unlucky one undersells it. To put a number on that fragility, we re-ran the full structure and style evaluation across 5 random splits (seeds 13, 42, 87, 123, 202) and report the mean and standard deviation.

| Metric | Mean | Std. Dev. |
| :--- | :--- | :--- |
| Structure test accuracy | 0.740 | 0.051 |
| Structure test macro F1 | 0.568 | 0.076 |
| Structure dev macro F1 | 0.608 | 0.073 |
| Style test rhetorical_mode macro F1 | 0.771 | 0.063 |
| Style test agency_style macro F1 | 0.611 | 0.168 |

**Table 7**: 5-seed split-stability summary. The mean estimates expected performance under random split variation; the standard deviation quantifies sensitivity to split choice.

![Seed stability summary](../images/seed_stability_summary.png)

**Figure 4** — per-seed structure performance with mean and ±1 standard deviation bands. The spread is what matters: test macro F1 ranges from **0.481 to 0.675** (spread 0.194) and test accuracy from **0.663 to 0.806** (spread 0.143). Same model, same data — change only which 98 headlines you hold out, and the system can look anywhere in that band. The seed-42 number we lead with in Section 4.2 happens to land at the favorable end; reporting it alone would overstate expected performance.

Approximate 95% intervals on the seed means are:

$$\bar{m}_{\text{Accuracy}} \in [0.695,\ 0.785] \qquad \bar{m}_{\text{Macro F1}} \in [0.502,\ 0.634]$$

A fair one-line description of the system, then, is "test accuracy in the high 0.7s and macro F1 in the high 0.5s to low 0.6s, with the upper end of either requiring a favorable split." This is exactly the kind of self-skeptical reporting H1's *robustness* criterion was designed to enforce, and the model passes it: the variance is real, but every seed sits well above both baselines.

### 4.5 Worked Decision-Path Examples
A central claim of this paper is that our predictions are **traceable**: every label can be explained by saying which rule fired, and why. Three worked examples, each hitting a different point in the priority order.

**Example A — question-form path.** Headline: *"Are we witnessing the death of expertise?"*

Decision path:
1. The headline contains a `?` and starts with an interrogative.
2. The `question_form` rule has the highest priority and fires immediately.
3. Downstream style outputs are conditioned on the structural label.

Predicted profile:
- structure: `question_form`
- lead_frame: `action_first`
- agency_style: `active_or_nonpassive`
- density_band: `medium_density`
- rhetorical_mode: `question_hook`

**Example B — passive compressed-event path.** Headline: *"Nine killed in second Turkish school shooting in two days"*

Decision path:
1. Not interrogative, so `question_form` does not fire.
2. The passive-fragment cue (`NUM + VBN`, here "Nine killed") is detected.
3. Returns `passive_clause` before any coordination or fragment fallback rule has a chance to fire.

Predicted profile:
- structure: `passive_clause`
- lead_frame: `context_first`
- agency_style: `passive_agent_omitted`
- density_band: `high_density`
- rhetorical_mode: `straight_report`

**Example C — nominal report-title path.** Headline: *"Iran Update Special Report, April 14, 2026 Institute for the Study of War"*

Decision path:
1. No interrogative cue.
2. No passive event predicate.
3. No finite subject-verb clause is detectable.
4. Falls into the nominal fragment rule (`noun_phrase_fragment`).

Predicted profile:
- structure: `noun_phrase_fragment`
- lead_frame: `actor_entity_first`
- agency_style: `active_or_nonpassive`
- density_band: `high_density`
- rhetorical_mode: `analysis_explainer`

These three examples aren't just illustrations — they are how we check H1's *interpretability* criterion. Every label traces back to a specific rule, a specific priority position, and a specific surface feature. That is the property that makes the system auditable in a way a black-box neural model is not.

## 5. Discussion:
### 5.1 Strengths of Solution
The thesis of this paper is simple: headlines are predictable syntactic products, not free-form prose, and the rules backed it up. The classifier captured the dominant newsroom structures well — especially `simple_clause` and `question_form` — which is exactly what we'd expect if headline syntax really does follow reusable patterns. Adding the four style dimensions on top moved us from a single label to a richer signature that simultaneously tracks framing, agency, density, and rhetorical intent.

The second strength is interpretability. Every prediction grounds in explicit parse features (POS tags, dependencies, entity positions), so any classification can be traced back to the linguistic evidence that produced it. The multi-seed evaluation reinforces that rigor by showing split variance instead of hiding it in a single point estimate.

Some style dimensions proved easier than expected. `lead_frame` and `density_band` saturate test macro F1 at 1.000, which says these aspects of headline construction are governed by stable surface conventions rather than subjective judgment. The harder dimensions (`rhetorical_mode`, rare passive variants of `agency_style`) are exactly where syntax-only signals run out and discourse-level cues take over — also expected. Either way, the central claim holds: headline construction is structurally constrained, stylistically patterned, and recoverable through interpretable, linguistically grounded rules.

### 5.2 Why Some Classes Remain Difficult
Being honest about where the system fails is part of the argument. The weak categories — the residual `other` bucket, parts of `coordination`, and some compressed passive variants — share a few underlying causes:

- **Sparse support**: classes like `other` and rare passive forms have so few gold examples in the held-out test set that a single misclassification swings their F1 dramatically. The macro F1 metric is what surfaces this; the accuracy metric can hide it entirely.
- **Boundary overlap between labels**: `coordination` and `simple_clause` are not always cleanly separable when a headline contains a comma list and a finite verb. The priority order resolves the conflict, but the resolution is not always the human's preferred reading.
- **Punctuation-heavy multi-clause headline variants**: headlines that combine semicolons, em-dashes, and embedded clauses (live-blog-style) sit between coordination and noun-phrase fragment, and the rule set has to make a hard call.
- **Residual annotation ambiguity in edge forms**: even our human annotators sometimes had to choose between two defensible labels, and the model inherits that ambiguity directly.

The takeaway is not that the model is broken on these classes, but that they are inherently noisier targets than the dominant ones, and the *macro* score rightly punishes us for that noise.

### 5.3 Practical Sandbox: The TUI Workbench
The interpretability claim is most concrete in the interactive terminal sandbox we built on top of the model.

![Interactive TUI workbench](../images/tui_image.svg)

**Figure 5** — the real-time TUI workbench. As the user types, the interface continuously shows the model's predictions with confidence estimates, the parse evidence supporting each decision (phrase flow, dependency mini-view, token map), warnings for edge-case constructions, and a benchmark footer anchoring everything in corpus size and validation metrics.

The workbench compresses the feedback loop from *minutes* (run the script, compare to gold, edit) to *seconds* (type, watch structure / agency / rhetorical mode shift live). Because every prediction is displayed alongside its parse evidence, an editor can immediately see *why* the system reads a candidate as `passive_clause` or `analysis_explainer` — and decide whether the system is wrong or whether the wording is. The same setup also flags potentially problematic constructions (unclear agency, excessive compression, misleading rhetorical hooks) before publication.

Empirical work supports this kind of interface. Banerjee and Urminsky (2021) analyze thousands of A/B-tested Upworthy headlines and show that informational, cognitive, linguistic, and affective textual cues — mapped via NLP tools to interpretable constructs — materially shift click-through. Their result is exactly the case for a feature-level editorial lever: instead of asking a model whether a candidate will perform, the editor sees which structural and stylistic dimensions it activates and edits with full visibility into why the system reads it that way. That is what makes interpretable NLP useful for newsroom workflows.

### 5.4 Limitations
The results come with five honest limitations that constrain the strength of any general claim:

1. **Rule-bound model family**: rule-based architectures are highly interpretable but inherently less flexible than fully learned models. Patterns that fall outside the rule set are systematically missed rather than gracefully approximated.
2. **Class imbalance**: dominant templates like `simple_clause` make accuracy easy and macro F1 hard. The minority classes drive the macro number disproportionately, so small annotation differences have outsized effects.
3. **Annotation subjectivity**: edge-case labels (especially within `rhetorical_mode` and `coordination`) are inherently debatable. Two competent annotators can produce defensible but different gold labels.
4. **English- and source-bias**: the corpus is built from a small set of English-language Google News RSS feeds, which limits external generalization to other languages, registers, and outlet styles.
5. **No external lexicon expansion**: we deliberately excluded curated event-noun and named-entity lexicons in this phase to keep the rules transparent. This makes some rare entity- or event-driven disambiguation harder than it could be, but it preserves interpretability.

None of these limitations invalidate H1; they bound it. The claim we defend is that headlines follow stable, recoverable templates, and that a transparent rule-based system captures them well above baseline — not that our particular rule set is the final word on headline structure.

## 6. Related Work:
Three strands of prior work bear on a parse-grounded, interpretable analysis of headline form: descriptive linguistic studies of headlines as a genre, empirical studies of how headline language affects readers, and the methodological tradition of recovering grammatical structure from raw text.

**Descriptive linguistics of headlines.** Bonyadi and Samuel (2013) run a contrastive textual study of editorial headlines from *The New York Times* and *Tehran Times*, surveying the headline-function tradition and analyzing presupposition and rhetorical devices. Their framing of headlines as simultaneously summary devices, attention-getters, and "relevance optimizers" is the conceptual scaffolding we operationalize as `rhetorical_mode` (§3.4). Robert (2020) applies Halliday's Systemic Functional Grammar to a small purposive sample of anti-corruption headlines from two Nigerian newspapers and reports that material processes — clauses of *doing* — dominate. Our 94.6% active-voice rate, dominant `simple_clause`, and 81.6% actor/entity-first openings (§4.1) describe the same action-oriented template through a different lens (voice and clause shape rather than process type), and the convergence across corpora and frameworks suggests the template lives in the genre rather than any one outlet.

**Empirical engagement studies.** Banerjee and Urminsky (2021) analyze A/B-tested Upworthy headlines, using NLP-extracted features to map textual cues to psychological constructs and measure their impact on click-through. Their finding — that interpretable linguistic features of a headline have measurable downstream consequences — is the practical case for the editorial lever we expose in the TUI workbench (§5.3). Their work treats headline language as an *input* to engagement; ours treats it as an *output* of stable syntactic templates, and offers an interpretable surface for seeing which templates a candidate activates.

**Parser robustness and corpus-driven grammar.** Eggleston and O'Connor (2022) study cross-dialect dependency parsing for social-media English, training a state-of-the-art parser on Tweebank v2 and showing that parsers fine-tuned on more conventional UD treebanks (GUM, EWT) lose substantial accuracy on social-media text — with an additional MAE/AAE dialect disparity. Their drift axis is dialectal social media; ours is headline compression; the warning that off-the-shelf parsers are domain-sensitive generalizes either way, and it informs the defensive parsing pipeline in §3.2. Atwell and Drakos (1987) sit further back in the lineage: they argued that a first-order Markov model over POS tags — the engine of their CLAWS tagger on the LOB corpus — can acquire a grammatical classification system from unrestricted English without hand-written grammar rules. Our work continues that corpus-driven program with explicit, parse-grounded rules over modern dependency parses rather than Markov transition probabilities.

In short, prior work has described the form of headlines (Bonyadi and Samuel, 2013; Robert, 2020), measured the engagement consequences of headline language (Banerjee and Urminsky, 2021), or built infrastructure for analyzing noisy or compressed text at scale (Eggleston and O'Connor, 2022; Atwell and Drakos, 1987). Our contribution sits at the intersection: a transparent, parse-grounded classifier for the structural and stylistic form of news headlines, evaluated rigorously against manually annotated gold labels and exposed through an interactive sandbox.

## 7. Conclusion
News headlines follow consistent structural patterns despite their brevity. The pipeline described here — collect, parse, analyze, classify, evaluate — recovers those patterns with a rule-based system grounded in syntactic features, beats trivial baselines on every seed, and stays fully interpretable in the process.

The four-dimensional style profile turns a single structural label into a richer signature that simultaneously tracks framing, agency, density, and rhetorical intent. The TUI workbench shows that this interpretability has practical value: an editor can iterate phrasing and watch structure, agency, and rhetorical mode shift in real time. The framework supports concrete applications in summarization, editorial tooling, and media-framing analysis. Future work may extend it with hybrid neural-rule models, larger and more diverse datasets, and multilingual analysis to test how universal the headline template really is.

## 8. Bibliography
Atwell, Eric Steven and Nicos Frixou Drakos. 1987. Pattern Recognition Applied to the  
    Acquisition of a Grammatical Classification System From Unrestricted English Text.  
    In *Proceedings of the Third Conference of the European Chapter of the Association  
    for Computational Linguistics (EACL 1987)*, pages 56–62, Copenhagen, Denmark.  
    Association for Computational Linguistics.

Banerjee, Akshina and Oleg Urminsky. 2021. The Language That Drives Engagement:  
    A Systematic Large-Scale Analysis of Headline Experiments. SSRN Working Paper.  
    Available at https://ssrn.com/abstract=3770366.

Bonyadi, Alireza and Moses Samuel. 2013. Headlines in Newspaper Editorials:  
    A Contrastive Study. *SAGE Open*, 3(2):1–10.  
    DOI: 10.1177/2158244013494863.

Eggleston, Chloe and Brendan O'Connor. 2022. Cross-Dialect Social Media Dependency  
    Parsing for Social Scientific Entity Attribute Analysis. In *Proceedings of the Eighth  
    Workshop on Noisy User-generated Text (W-NUT 2022)*, pages 38–50, Gyeongju,  
    Republic of Korea. Association for Computational Linguistics.

Robert, Esther. 2020. Language Use in Selected Nigerian Newspaper Headlines.  
    *Journal of Arts and Humanities*, 9(1):91–103.  
    DOI: 10.18533/journal.v9i1.1800.

> **Bottom line:** News headlines are not free-form snippets; they are reproducible syntactic products. Interpretable, parse-grounded NLP can measure that structure directly, evaluate it rigorously, and put it in front of the people who write headlines.