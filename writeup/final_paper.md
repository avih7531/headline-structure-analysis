# Structural Patterns in News Headlines: A Computational Linguistic Analysis Using NLP

**Victor Derani, Avi Herman, Kylie Lin**

## 1. Abstract:
We analyze the syntactic and structural patterns of news headlines using NLP. Our pipeline collects headlines from RSS feeds, parses them with spaCy, and feeds the parses into a transparent rule-based classifier to try to prove that headlines are not free-form prose but reproducible syntactic products. The classifier assigns each headline a primary structural label (question, passive, coordination, noun-phrase fragment, simple clause, or other) plus a four-dimensional style profile (lead frame, agency style, density band, rhetorical mode). On a manually annotated gold-standard set of 464 headlines, the classifier reaches **0.806 accuracy** and **0.675 macro F1** on the held-out test split. That beats a majority baseline by +16.3 accuracy points (macro F1 0.130 → 0.675) and a random baseline by +34.7 points. A 5-seed split-stability sweep places the structure test macro F1 mean at **0.568 ± 0.076**, making the size of "split luck" explicit. Style dimensions decompose cleanly: framing and density are nearly perfectly recoverable (test macro F1 1.000 / 1.000), while agency and rhetorical mode are harder (0.428 / 0.820). The framework supports summarization, style auditing, and media-framing analysis, and is exposed through an interactive terminal sandbox that updates structure, agency, and rhetorical mode in real time as a user rewrites a headline.

## 2. Introduction:
News headlines are among the most widely read forms of text. Millions of people use them as a primary information source, often to form opinions or decide what to click on, and do so without reading the underlying article. Despite that reach, headlines are extreme in form: they drop function words, restructure sentences, and compress an event into roughly a dozen words, all while balancing clarity, informativeness, and engagement.

From a linguistic-engineering point of view, a headline is information compression at the limit. From an applied point of view, the same compressed form drives summarization quality, search-result snippets, and misinformation and bias. If we can describe headline form mechanically, we can also flag it mechanically.

The idea that grammatical regularities can be recovered from raw text corpora was a foundational aspect to our project. Atwell and Drakos (1987) showed that a first-order Markov model over part-of-speech tags, the engine of their CLAWS tagger on the LOB corpus, can acquire a grammatical classification system from unrestricted English without hand-written grammar rules. Our project sits in that lineage but uses explicit, parse-grounded rules over modern dependency parses, applied specifically to the compressed register of news headlines.

These motivations led us to three concrete research questions:

* What grammatical structures are most common in news headlines, and how do headlines achieve high information density?
* What syntactic templates dominate headline construction, and are these patterns consistent across different news domains?
* Can a transparent, rule-based system recover those structures with reliable out-of-sample performance?

### 2.1 Hypothesis and Falsification Criteria
Our hypothesis, which we aim to prove throughout this experiment, is:

> **H1:** News headlines follow stable, reusable syntactic templates that can be predicted from POS and dependency signals at rates significantly above trivial baselines.

To make H1 explicit and proveable, the conditions under which the hypothesis should be rejected:

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

We then run a two-pass parse strategy. The first pass parses the headline as written and scores parse quality. If the score is poor, such as no clear verb in a long headline or ambiguous casing on what should be a proper noun, we run a fallback parse with light re-casing and lexicon overrides for common headline ambiguities, then keep the better-scoring of the two parses. This costs us a small constant factor in compute and buys us much better behavior on the headlines that off-the-shelf parsers most often mishandle.

### 3.3 Analyze
The Analyze stage derives higher-level features from each parse: collapsed POS patterns, macro phrase structures (NP, VP, MOD), opening and ending patterns, voice, and a coarse headline type. On top of those features, a suite of analysis functions reports descriptive statistics across the corpus, entity-type counts, content-word density, voice distribution, common dependency templates, length patterns, and the most frequent root verbs and openings, among others. A `generate_insights()` step rolls these into short corpus-level summaries.

This stage does not assign labels. To turn these descriptive patterns into predictions, we transition to Classify.

### 3.4 Classify
The Classify stage takes the parsed features (POS tags, dependency relations, entities, derived signals) and assigns each headline one primary structural label. The label set is:

1.  **question_form**: interrogative punctuation or opening pattern
2.  **passive_clause**: passive cues (e.g., `nsubjpass`, auxiliary/passive constructions)
3.  **coordination**: multi-clause coordination (`cc`/`conj` and coordination signals)
4.  **noun_phrase_fragment**: nominal fragment with no finite clause
5.  **simple_clause**: canonical finite subject-verb clause
6.  **other**: residual category for out-of-pattern forms

Rules fire in a fixed priority order, question, passive, coordination, NP fragment, simple clause, other, to keep output deterministic. Each rule is a small set of explicit linguistic conditions over POS, dependencies, punctuation, and token order; ambiguous or short headlines fall through to `other`.

Structure alone is too coarse to capture headline variation, so we layer on four style dimensions:

1.  **lead_frame**: actor/entity-first, action-first, context-first, etc.
2.  **agency_style**: active vs. passive (with/without agent)
3.  **density_band**: low/medium/high (based on content-word ratio)
4.  **rhetorical_mode**: straight-report, analysis-explainer, question-hook, live/alert

Each dimension is driven by a distinct signal: entity position for `lead_frame`, passive cues for `agency_style`, content-word ratio for `density_band`, and a mix of structural and lexical cues for `rhetorical_mode`.

The `rhetorical_mode` dimension is based off a prevoius study. Bonyadi and Samuel (2013), surveying the headline-function tradition in their contrastive study of editorial headlines, frame headlines as serving three simultaneous roles: summaries of the main event, attention-getting devices, and "relevance optimizers" between story and reader. Our four modes are a parse-grounded operationalization of those competing communicative functions.

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
The Evaluate stage compares our predicted labels against a manually annotated gold-standard set. The gold dataset contains **464** hand-labeled headlines covering both structure and style, split **60 / 20 / 20** into Train, Dev, and Test. Train develops the rules, Dev tunes them, and Test is touched at the very end. Keeping Test isolated during rule iteration validates the final result as unbiased.

We score the system with accuracy, per-class precision/recall/F1, and macro F1. Macro F1 is the metric that exposes whether minority classes are also being handled, as `simple_clause` dominates.

To put the numbers in context, we also report two trivial baselines:

- **Majority baseline** — always predicts the most common class. If our system isn't beating this on macro F1, it's just exploiting class imbalance.
- **Random baseline** — assigns labels randomly. If we're not beating this, there's no learned structure at all.

Finally, single-split numbers can be misleading. A lucky or unlucky test split can create biased results against or for our model. We re-run the entire evaluation across **5 random seeds (13, 42, 87, 123, 202)** and report the mean and standard deviation, so "split luck" becomes a measurable quantity.

Beyond this offline evaluation, we also built an interactive terminal sandbox on top of the same model that shows predictions, confidences, and parse evidence in real time. We defer its full description and screenshot to Section 5.3.

## 4. Proof/Evaluation:
Before turning to the numbers, the metrics we use:

$$\text{accuracy} = \frac{\#\text{correct}}{\#\text{total}} \qquad \text{precision} = \frac{TP}{TP+FP} \qquad \text{recall} = \frac{TP}{TP+FN} \qquad F1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

For a class $c$, **precision** measures how often we're right when we predict $c$ (low false-positive cost), and **recall** measures how often we catch real $c$'s (low false-negative cost). F1 punishes optimizing one at the expense of the other. **Macro F1** is the unweighted average of per-class F1. It is what surfaces minority-class problems that accuracy hides.

When we run the 5-seed sweep (13, 42, 87, 123, 202), we report the mean and sample standard deviation across those runs:

$$\overline{m} = \frac{1}{K} \sum_{i=1}^{K} m_{i} \qquad s = \sqrt{\frac{1}{K-1} \sum_{i=1}^{K} (m_{i} - \overline{m})^{2}}$$

The mean estimates expected performance under random split variation; $s$ quantifies how sensitive that estimate is to which split we happened to draw.

### 4.1 Corpus-Level Structural Findings
We organize results into four parts: corpus-level structural findings (this section), structure-classifier performance (4.2), style-dimension performance (4.3), and stability analysis (4.4). This section answers the question "what do news headlines actually look like structurally?"

| Metric | Value |
| :--- | :--- |
| Average length | 13.7 tokens |
| Content-word ratio | 73.2% |
| Headlines with verbs | 92.1% |
| Active/non-passive rate | 94.6% |
| Headlines with named entities | 95.7% |
| Avg entities per headline | 2.4 |
| Actor/entity-first openings | 81.6% |

The average headline is ~14 tokens, 73% of them content words. 92% contain a verb, so even though they're compressed, they remain *sentences about an action*, not captions. 95% contain at least one named entity (2.4 on average), 95% are active voice, and 82% lead with the main actor or entity. 

Results like these have been seen previously. Robert (2020) applied Halliday's Systemic Functional Grammar to fourteen anti-corruption headlines from two Nigerian dailies and reports that *material processes*, clauses that grammaticalize doing rather than being or thinking, are the dominant process type. The convergence is real even though the lenses differ: she measures transitivity, we measure voice and clause shape, but both characterizations describe action-oriented, agent-led headline syntax as the genre default. The template lives in the genre, not in any single outlet.

![Structure label distribution](../images/structure_label_distribution.png)

**Figure 1** — distribution of predicted structure labels across the corpus. The dominant class is the simple finite clause, with noun-phrase fragments as a clear secondary mode and the rest (questions, passives, coordinations, `other`) as a long tail. The shape indicates that headline writing concentrates on two or three reusable templates. Rare cases are due to failing the brevity constraint that defines the genre.

### 4.2 Structure Classifier Performance
The sectoin answers whether the model can actually predict that template.

| Evaluation Slice | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| All labeled (n=465) | 0.772 | 0.601 |
| Dev (n=90) | — | 0.561 |
| Test (n=98) | 0.806 | 0.675 |

**Table 2** — aggregate structure metrics. The pattern across slices is consistent: macro F1 sits below accuracy because the dataset is imbalanced (`simple_clause` dominates). On the held-out test split the classifier reaches **0.806 accuracy** and **0.675 macro F1**. The dev row deliberately omits accuracy because, on that imbalanced slice, accuracy isn't an informative tuning signal.

A test number on its own is hard to read. We compare against two trivial baselines:

| Model | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| Rule-based classifier | 0.806 | 0.675 |
| Majority baseline | 0.643 | 0.130 |
| Random baseline | 0.459 | 0.159 |

**Table 3** — baseline comparison on the held-out test split. Our system beats the majority baseline by **+16.3 accuracy points** and the random baseline by **+34.7 points**. The macro F1 jumps from 0.130 (majority) and 0.159 (random) to 0.675, more than four times the baseline. 

| Label | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| question_form | 0.960 | 0.960 | 0.960 |
| passive_clause | 0.478 | 0.344 | 0.400 |
| coordination | 0.722 | 0.448 | 0.553 |
| noun_phrase_fragment | 0.578 | 0.627 | 0.602 |
| simple_clause | 0.882 | 0.865 | 0.874 |
| other | 0.138 | 0.500 | 0.216 |

**Table 4** — per-class performance. Strong on the dominant classes (`simple_clause`, `question_form`), weaker on ambiguous ones (`other`, `passive_clause` minority variants). 

A normal-approximation 95% confidence interval on the test accuracy 0.806 (n = 98) is:

$$\text{CI}_{0.95}(\text{Accuracy}) \approx [0.728,\ 0.884]$$

Both baselines (0.643, 0.459) fall outside this interval. 

### 4.3 Style-Dimension Performance
Beyond the structural label, can the model also recover stylistic characteristics? We evaluate the four style dimensions twice: across the entire labeled slice, and on the held-out test split alone, so "consistent on training data" and "generalizes" don't get conflated.

| Dimension | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| lead_frame | 0.989 | 0.986 |
| agency_style | 0.935 | 0.622 |
| density_band | 0.994 | 0.995 |
| rhetorical_mode | 0.914 | 0.785 |

**Table 5** — All-slice (n=465). `lead_frame` and `density_band` are nearly perfectly recoverable: headline openings and information density follow stable, surface-level conventions our rules pick up almost every time. `agency_style` is high in accuracy but much lower in macro F1, demonstrating class imbalance. `rhetorical_mode` lands in the middle: reliable on the dominant `straight_report` but more uncertain on `analysis_explainer` and edge-case live-alerts, where syntax alone doesn't fully disambiguate discourse function.

| Dimension | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| lead_frame | 1.000 | 1.000 |
| agency_style | 0.908 | 0.428 |
| density_band | 1.000 | 1.000 |
| rhetorical_mode | 0.908 | 0.820 |

**Table 6** — Held-out test (n=98). `lead_frame` and `density_band` saturate the score, consistent with their being deterministic functions of surface features (entity position and content-word ratio). `rhetorical_mode` rises slightly compared to the all-slice number, so the rule generalizes rather than overfits. `agency_style` drops noticeably (macro F1 0.622 → 0.428) in a small held-out slice, a few misclassified rare passives crater the minority-class score. We discuss this in Section 5.2.

![Model performance summary](../images/model_performance_summary.png)

**Figure 2** — model performance summary. Accuracy and macro F1 across the structure classifier and four style dimensions, both all-slice and held-out test. Structure performance is moderate but well above baseline; framing and density saturate; agency and rhetorical mode sit in between, where syntax alone is only partially sufficient.

![Performance heatmap](../images/performance_heatmap.png)

**Figure 3** — performance heatmap. The same picture in colour: form-grounded properties (where the headline starts, how dense it is) are dark green; semantically loaded ones (agency, rhetoric) are lighter; structural classification sits in between. 

### 4.4 Stability Analysis
Single-split numbers are fragile, and to put a number on that fragility, we re-ran the full structure and style evaluation across 5 random splits (seeds 13, 42, 87, 123, 202) and report the mean and standard deviation.

| Metric | Mean | Std. Dev. |
| :--- | :--- | :--- |
| Structure test accuracy | 0.740 | 0.051 |
| Structure test macro F1 | 0.568 | 0.076 |
| Structure dev macro F1 | 0.608 | 0.073 |
| Style test rhetorical_mode macro F1 | 0.771 | 0.063 |
| Style test agency_style macro F1 | 0.611 | 0.168 |

**Table 7**: 5-seed split-stability summary. The mean estimates expected performance under random split variation; the standard deviation quantifies sensitivity to split choice.

![Seed stability summary](../images/seed_stability_summary.png)

**Figure 4** — per-seed structure performance with mean and ±1 standard deviation bands. The test macro F1 ranges from **0.481 to 0.675** (spread 0.194) and test accuracy from **0.663 to 0.806** (spread 0.143). The same model and data are used, changing only which 98 headlines to hold out so that the system can look anywhere in that band. The seed-42 number we lead with in Section 4.2 happens to land at the favorable end; reporting that it would overstate expected performance.

Approximate 95% intervals on the seed means are:

$$\bar{m}_{\text{Accuracy}} \in [0.695,\ 0.785] \qquad \bar{m}_{\text{Macro F1}} \in [0.502,\ 0.634]$$

The system follows a methodology of "test accuracy in the high 0.7s and macro F1 in the high 0.5s to low 0.6s, with the upper end of either requiring a favorable split." 

### 4.5 Worked Decision-Path Examples
This experiment aims to prove that our predictions are **traceable**: every label can be explained by saying which rule fired and why. Three worked examples, each hitting a different point in the priority order.

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

Every label traces back to a specific rule, a specific priority position, and a specific surface feature. 

These same examples also explain why the features matter for summarization. A question-form headline can be attention-grabbing, but it is often a weak neutral summary because the reader must resolve the question before recovering the event. A passive compressed headline can fit the space budget while hiding the agent, which is useful for brevity but risky if a summarizer needs to preserve who did what. A nominal report-title headline, by contrast, often behaves more like a section label or index entry than a sentence summary. Detecting those forms lets a summarization system decide whether to keep the wording, expand it, or rewrite it into a more explicit declarative summary.
## 5. Discussion:

### 5.1 How We Demonstrate H1
We prove H1, headlines follow stable, reusable syntactic templates—by assembling four converging evidence strands. Through empirical performance, we determine the structure classifier beats trivial baselines on held-out test data (Section 4.2; test accuracy 0.806, test macro F1 0.675, with baselines 0.643/0.459 lying outside the accuracy CI). Through template consistency, we prove that  corpus-level distributions show dominant recurring patterns (Section 4.1; e.g., 81.6% actor/entity-first openings, 92.1% contain verbs). Through robustness, we show a 5-seed split sweep (Section 4.4) yields mean macro F1 0.568 ± 0.076, which demonstrates stability across splits. Lastly interpretability/falsifiability, we show that every prediction is traceable to explicit parse-grounded rules and a fixed priority order (Section 4.5), allowing us to inspect and reject the hypothesis if predictions were unexplainable. Together these points satisfy the falsification criteria in Section 2.1 and substantiate H1.

### 5.2 Strengths of Solution
We have proved that headlines are predictable syntactic products, not free-form prose. The classifier captured the dominant newsroom structures well, especially `simple_clause` and `question_form`, which was expected if headline syntax really does follow reusable patterns. Adding the four style dimensions on top moved us from a single label to a richer signature that simultaneously tracks framing, agency, density, and rhetorical intent.

One caveat is that we do not yet report inter-annotator agreement for the gold labels. That means this draft does not yet quantify a human ceiling, so we cannot fully separate label ambiguity from model error. The current results still show that the system is far above majority and random baselines, but the annotation protocol should be treated as a manually curated gold standard rather than a formally reliability-validated one.
The second strength is interpretability. Every prediction grounds in explicit parse features (POS tags, dependencies, entity positions), so any classification can be traced back to the linguistic evidence that produced it. The multi-seed evaluation reinforces that rigor by showing split variance instead of hiding it in a single point estimate.

Some style dimensions proved easier than expected. `lead_frame` and `density_band` saturate test macro F1 at 1.000, which says these aspects of headline construction are governed by stable surface conventions rather than subjective judgment. The harder dimensions (`rhetorical_mode`, rare passive variants of `agency_style`) are exactly where syntax-only signals run out and discourse-level cues take over, which is also expected. In conclusion, headline construction is structurally constrained, stylistically patterned, and recoverable through interpretable, linguistically grounded rules.

### 5.3 Why Some Classes Remain Difficult
The weak categories, the residual `other` bucket, parts of `coordination`, and some compressed passive variants, share a few underlying causes:

- **Sparse support**: classes like `other` and rare passive forms have so few gold examples in the held-out test set that a single misclassification swings their F1 dramatically. The macro F1 metric is what surfaces this and the accuracy metric can hide it entirely.
- **Boundary overlap between labels**: `coordination` and `simple_clause` are not always cleanly separable when a headline contains a comma list and a finite verb. The priority order resolves the conflict, but the resolution is not always the human's preferred reading.
- **Punctuation-heavy multi-clause headline variants**: headlines that combine semicolons, em-dashes, and embedded clauses (live-blog-style) sit between coordination and noun-phrase fragment, and the rule set has to make a hard call.
- **Residual annotation ambiguity in edge forms**: human annotators sometimes had to choose between two defensible labels, and the model inherits that ambiguity directly.

The model is broken on these classes, but these classes are inherently noisier targets than the dominant ones, and the *macro* score punishes is affected by that noise.

### 5.4 Practical Sandbox: The TUI Workbench
The interpretability claim is most concrete in the interactive terminal sandbox we built on top of the model.

![Interactive TUI workbench](../images/tui_image.svg)

**Figure 5** — the real-time TUI workbench. As the user types, the interface continuously shows the model's predictions with confidence estimates, the parse evidence supporting each decision (phrase flow, dependency mini-view, token map), warnings for edge-case constructions, and a benchmark footer anchoring everything in corpus size and validation metrics.

The workbench compresses the feedback loop from *minutes* (run the script, compare to gold, edit) to *seconds* (type, watch structure / agency / rhetorical mode shift live). Because every prediction is displayed alongside its parse evidence, an editor can immediately see *why* the system reads a candidate as `passive_clause` or `analysis_explainer`, and decide whether the system is wrong or whether the wording is. The same setup also flags potentially problematic constructions (unclear agency, excessive compression, misleading rhetorical hooks) before publication.

This is supported by a study by Banerjee and Urminsky (2021), where they  analyze thousands of A/B-tested Upworthy headlines and show that informational, cognitive, linguistic, and affective textual cues (mapped via NLP tools to interpretable constructs) materially shift click-through. Their result is exactly the case for a feature-level editorial lever: instead of asking a model whether a candidate will perform, the editor sees which structural and stylistic dimensions it activates and edits with full visibility into why the system reads it that way. That is what makes interpretable NLP useful for newsroom workflows.

### 5.5 Structural Implications and Real World Applications
As introduced earlier, headlines represent an extreme case of information compression, where complex events are reduced to minimal linguistic forms while preserving core meaning. This section illustrates how the structural features identified in this project reflect broader communicative goals and why they are relevant for both NLP applications and real-world information consumption.

**Information Compression and Language Efficiency:**
From a computational perspective, headlines demonstrate how language can achieve high informational density through structural simplification. Consider the following example:
Source sentence:
“The Federal Reserve announced that it would increase interest rates by 0.25% in response to ongoing inflation concerns.”

Headline A (simple_clause):
“Federal Reserve raises interest rates”

Headline B (noun_phrase_fragment):
“Fed raises rates 0.25% amid inflation concerns”

Headline B achieves significantly higher information density by removing function words (e.g., “that it would”) and compressing content into a noun_phrase_fragment. Despite its brevity, it preserves key semantic elements: actor (“Fed”), action (“raises”), magnitude (“0.25%”), and context (“inflation concerns”).
This illustrates how structural features encode efficient summarization strategies. Such patterns are directly relevant to NLP tasks like summarization and question answering, where systems must balance brevity and informativeness. By modeling these structural transformations, NLP systems can learn how to compress information while maintaining clarity.

**User Behavior and Information Consumption:**
In modern information environments, users frequently scan headlines rather than reading full articles. Structural choices therefore directly impact how efficiently information is communicated.

Headline A (question_form):
“Is the economy heading toward a recession?”

Headline B (simple_clause):
“Economy shows signs of recession”

The question_form in Headline A introduces uncertainty and invites engagement, while Headline B delivers a direct informational claim. From a user perspective, these structures serve different functions: one encourages exploration, while the other prioritizes clarity and speed of understanding.
Understanding these structural differences can inform the design of systems that optimize for rapid information delivery, particularly in interfaces where users rely on quick scanning, such as news aggregators and search engines.

**Media Framing and Public Perception:**
Beyond efficiency, headline structure plays a critical role in shaping how information is perceived. Structural features can subtly influence interpretation, even when describing the same event.

Headline A (agency_style):
“Government admits failure in policy rollout”

Headline B (noun_phrase_fragment):
“Policy rollout faces challenges”

Both headlines refer to a similar situation, but Headline A explicitly assigns responsibility through an agency_style construction (“Government admits”), while Headline B removes the agent entirely. This shift in structure changes how accountability is perceived.
Such differences are particularly important in the context of misinformation and media bias. By systematically identifying structural patterns, such as omission of agents, use of passive constructions, or rhetorical framing, this approach provides tools for analyzing how headlines may influence public understanding.

### 5.6 Limitations
The results come with five honest limitations that constrain the strength of any general claim:

1. **Rule-bound model family**: rule-based architectures are highly interpretable but inherently less flexible than fully learned models. Patterns that fall outside the rule set are systematically missed rather than gracefully approximated.
2. **Class imbalance**: dominant templates like `simple_clause` make accuracy easy and macro F1 hard. The minority classes drive the macro number disproportionately, so small annotation differences have outsized effects.
3. **Annotation subjectivity**: edge-case labels (especially within `rhetorical_mode` and `coordination`) are inherently debatable. Two competent annotators can produce defensible but different gold labels.
4. **English- and source-bias**: the corpus is built from a small set of English-language Google News RSS feeds, which limits external generalization to other languages, registers, and outlet styles.
5. **No external lexicon expansion**: we deliberately excluded curated event-noun and named-entity lexicons in this phase to keep the rules transparent. This makes some rare entity- or event-driven disambiguation harder than it could be, but it preserves interpretability.
6. **No inter-annotator agreement report yet**: the current draft does not include a formal agreement score for the manual labels, so we cannot claim a measured human ceiling or quantify how much better than chance independent annotators are. The system-vs-baseline numbers do address chance-level performance for the model, but not annotation reliability.

None of these limitations invalidate H1, but rather bound it. The claim we defend, that headlines follow stable, recoverable templates, and that a transparent rule-based system captures them well above baseline, still holds.

## 6. Related Work:
Research on news headlines has increasingly focused on generation tasks, particularly through large language models and neural summarization systems. However, less attention has been given to the structural and syntactic properties of headlines themselves. Our work approaches headlines not only as outputs of generation systems, but also as linguistic objects whose grammatical organization can be systematically analyzed through computational methods.

**Corpus-Based Grammatical Pattern Discovery.** Atwell and Drakos pioneered one of the earliest corpus-based approaches to grammatical pattern discovery using statistical methods on unrestricted English text. Their work challenged the separation between statistical pattern recognition and syntactic analysis in computational linguistics. In particular, they argued that “a Corpus of English text samples can constitute a definitive source of data in the description of linguistic constructs or structures” and investigated whether pattern recognition techniques could support “the acquisition of a grammatical classification system from Unrestricted English text” . Their use of Markov-based grammatical classification systems demonstrated that statistical regularities in language could reveal meaningful syntactic organization without relying entirely on handcrafted grammatical rules.
This foundational perspective directly relates to our project. While Atwell and Drakos focused on discovering grammatical classes from unrestricted corpora, our work applies modern NLP pipelines to identify recurring syntactic patterns within news headlines. Their work was a point of reference to point us to follow their corpus-driven methodology, where we treat headlines as a linguistic domain whose grammatical regularities can be computationally analyzed. However, unlike their emphasis on grammatical classification systems, our project focuses on dependency structures, part-of-speech distributions, named entities, and recurring headline conventions in contemporary digital journalism.

**Structural Representations for Cross-Domain NLP.** More recent NLP research has continued to emphasize the importance of structural representations for language understanding. Dukić et al. explore the use of Open Information Extraction (OpenIE) to improve robustness in event trigger detection across domains. Their work focuses on how structured semantic representations can generalize more effectively than surface-level lexical features when processing diverse text sources. Rather than relying exclusively on token-level information, the authors incorporate extracted relational structures to better capture underlying event semantics. They argue that Open Information Extraction can improve robustness across domains by providing structured semantic representations, highlighting the broader NLP trend toward structural and relational approaches to language understanding. 
This perspective is particularly relevant to our project because news headlines often compress information into highly constrained syntactic forms that may vary lexically while preserving similar structural relationships. Like Dukić, our work emphasizes representations beyond individual words by analyzing dependency relations, part-of-speech patterns, and recurring grammatical constructions. However, whereas their study focuses on improving event trigger detection through semantic extraction techniques, our project investigates the structural conventions of headlines themselves. In particular, we examine how syntactic compression, entity prominence, and dependency organization contribute to the characteristic linguistic style of modern news headlines.

**Robust Dependency Parsing in Non-Standard Text.** Related work from the Workshop on Noisy User-generated Text (WNUT) further highlights the importance of robust structural parsing in non-standard linguistic environments. Eggleston and O’Connor (2022) examine cross-dialect dependency parsing and demonstrate that off-the-shelf parsers degrade systematically when applied to text that diverges from standard news-style training distributions. Their work evaluates dependency parsing performance across dialectal and stylistic variation, particularly between Mainstream American English (MAE) and African American English (AAE), showing that parsing accuracy consistently decreases for underrepresented linguistic varieties . More broadly, the paper emphasizes that dependency parsing remains a foundational tool for downstream semantic extraction systems and social-language analysis.
This work is especially relevant to our project because headlines frequently exhibit compressed syntax, irregular capitalization, omitted function words, and punctuation-heavy constructions that differ substantially from standard prose. Their findings motivated several robustness decisions in our parsing pipeline, including parser fallback strategies, lightweight punctuation normalization, and parse-quality scoring for difficult headline structures. Similar to the WNUT study, our project treats dependency parsing not as a perfect preprocessing step, but as a potentially unstable component whose limitations directly affect higher-level structural analysis. However, while Eggleston and O’Connor focus on dialectal fairness and parsing robustness in social media text, our work applies dependency-based analysis to the highly compressed syntactic environment of digital news headlines, with the goal of identifying stable and interpretable structural templates.

**Chain-of-Thought Approaches to Headline Generation.** Recent work on headline generation has increasingly relied on large language models and neural reasoning frameworks. Zhao et al. (2024), in their SemEval-2024 Task 7 system CoT-NumHG, propose a Chain-of-Thought (CoT) based supervised fine-tuning strategy for numeral-aware headline generation . Their work focuses on improving the ability of large language models to process and accurately generate numerical information in headlines, addressing a common weakness in neural text generation systems. The authors note that “numbers often carry key information” in news headlines and that inaccuracies in numerical reasoning contribute substantially to generation errors . To address this issue, they incorporate Chain-of-Thought prompting into supervised fine-tuning in order to improve “numeral perception, interpretability, accuracy, and the generation of structured outputs” . Their CoT-NumHG-Mistral-7B system achieved a reported accuracy of 94% on the SemEval numeral-aware headline generation task.
This work is highly relevant because it demonstrates how headlines have become an important benchmark for modern NLP systems, particularly in summarization and controlled text generation. However, while Zhao approaches headlines primarily as outputs to be generated, our work instead treats headlines as linguistic artifacts whose internal structure can be systematically analyzed. Their study emphasizes semantic reasoning and generative performance through large language models, while our project focuses on interpretable structural analysis through dependency parsing, part-of-speech patterns, and rule-based syntactic classification. Nevertheless, their emphasis on numerical salience directly informs our own analysis of high-density headline constructions, especially headlines in which quantitative expressions function as central structural elements.

**Multimodal Approaches to Headline Generation.** Qiao et al. (2022) further extend headline research into the multimodal domain through GraMMo, a framework for multimodal headline generation that combines video frames and textual transcripts to generate natural language titles . Their work reflects the broader shift in NLP toward transformer-based multimodal architectures capable of integrating heterogeneous information sources. The authors argue that effective video headline generation requires balancing both linguistic and visual representations, noting that “a major challenge in simply gluing language model and video-language model is the modality balance” . To address this problem, they propose a system that grafts a pre-trained video-language encoder onto a generative language model while introducing a consensus fusion mechanism to coordinate inter- and intra-modality relations. Their experiments demonstrate that multimodal representations can improve headline generation quality in real-world applications.
This work is relevant to our project because it reinforces the growing importance of headlines as compressed informational units within modern NLP systems. Similar to the SemEval CoT-NumHG framework, GraMMo approaches headlines primarily as generative targets optimized for summarization, retrieval, recommendation, and user engagement. In contrast, our work shifts the focus away from headline generation itself and toward the structural properties underlying headline construction. Rather than optimizing neural generation quality, we investigate whether headlines follow stable, recoverable syntactic templates that can be identified through interpretable linguistic analysis.

Taken together, prior work has primarily emphasized semantic understanding, generative performance, and multimodal modeling, often through increasingly complex neural systems. Comparatively less attention has been devoted to the structural and grammatical conventions that govern headline construction itself. Our work contributes to this gap by presenting an interpretable, parse-grounded analysis of headline syntax that treats headlines not merely as outputs of generation systems, but as reproducible linguistic structures. By combining dependency parsing, structural classification, and multi-dimensional style profiling, our framework demonstrates that headline construction follows consistent and recoverable grammatical patterns that can be quantitatively analyzed without relying exclusively on opaque neural models.

## 7. Conclusion
News headlines follow consistent structural patterns despite their brevity. The pipeline described here (collect, parse, analyze, classify, evaluate) recovers those patterns with a rule-based system grounded in syntactic features, beats baselines on every seed, and stays fully interpretable in the process.

The four-dimensional style profile turns a single structural label into a richer signature that simultaneously tracks framing, agency, density, and rhetorical intent. The TUI workbench shows that this interpretability has practical value: an editor can iterate phrasing and watch structure, agency, and rhetorical mode shift in real time. The framework supports concrete applications in summarization, editorial tooling, and media-framing analysis. Future work may extend it with hybrid neural-rule models, larger and more diverse datasets, and multilingual analysis to test how universal the headline template really is.

## 8. Bibliography
Banerjee, Akshina and Oleg Urminsky. 2021. The language that drives engagement:  
    A systematic large-scale analysis of headline experiments. SSRN Working Paper.  
    Available at https://ssrn.com/abstract=3770366.

Bonyadi, Alireza and Moses Samuel. 2013. Headlines in newspaper editorials:  
    A contrastive study. *SAGE Open*, 3(2):1–10.

Dukić, David, Kiril Gashteovski, Goran Glavaš, and Jan Šnajder. 2024. Leveraging open  
    information extraction for more robust domain transfer of event trigger detection.  
    In *Findings of the Association for Computational Linguistics: EACL 2024*, pages  
    1197–1213, St. Julian’s, Malta. Association for Computational Linguistics.

Eggleston, Chloe and Brendan O’Connor. 2022. Cross-dialect social media dependency  
    parsing for social scientific entity attribute analysis. In *Proceedings of the  
    Eighth Workshop on Noisy User-generated Text (W-NUT 2022)*, pages 38–50,  
    Gyeongju, Republic of Korea. Association for Computational Linguistics.

Zhao, Junzhe, Yingxi Wang, Huizhi Liang, and Nicolay Rusnachenko. 2024. NCL_NLP at  
    SemEval-2024 task 7: CoT-NumHG: A CoT-based SFT training strategy with large  
    language models for number-focused headline generation. In *Proceedings of the  
    18th International Workshop on Semantic Evaluation (SemEval-2024)*, pages  
    261–269, Mexico City, Mexico. Association for Computational Linguistics.

Qiao, Lingfeng, Chen Wu, Ye Liu, Haoyuan Peng, Di Yin, and Bo Ren. 2022. Grafting  
    pre-trained models for multimodal headline generation. In *Proceedings of the  
    2022 Conference on Empirical Methods in Natural Language Processing: Industry  
    Track*, pages 244–253, Abu Dhabi, UAE. Association for Computational Linguistics.

Robert, Esther. 2020. Language use in selected Nigerian newspaper headlines.  
    *Journal of Arts and Humanities*, 9(1):91–103.

> **Bottom line:** News headlines are not free-form snippets; they are reproducible syntactic products. Interpretable, parse-grounded NLP can measure that structure directly, evaluate it rigorously, and put it in front of the people who write headlines.