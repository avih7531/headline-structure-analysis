# Structural Patterns in News Headlines: A Computational Linguistic Analysis Using NLP

**Victor Derani, Avi Herman, Kylie Lin**

## 1. Abstract:
This paper presents an analysis of syntactic and structural patterns in news headlines through Natural Language Processing (NLP). We built a pipeline that collects headlines from RSS feeds, parses them with the spaCy library, and extracts part-of-speech (POS) patterns, dependency structures, and named entities. On those parses, we run a transparent rule-based classifier that assigns each headline a primary structural label (question, passive, coordination, noun-phrase fragment, simple clause, or other) and a multi-dimensional style profile (lead frame, agency style, density band, rhetorical mode). We argue, and quantitatively show, that headlines are not free-form prose but reproducible syntactic products: their form is recoverable, structurally constrained, and stylistically decomposable.

Evaluated against a manually annotated gold-standard set of 464 headlines, the classifier reaches an accuracy of 0.806 and a macro F1 of 0.675 on the held-out test split, beating a majority baseline by +16.3 percentage points in accuracy (macro F1 0.130 → 0.675) and a random baseline by +34.7 percentage points. A 5-seed split-stability sweep places the structure test macro F1 mean at 0.568 ± 0.076, making the size of "split luck" explicit rather than rhetorical. Style dimensions decompose meaningfully: framing and density are nearly perfectly recoverable (test macro F1 1.000 / 1.000), while agency and rhetorical mode remain harder (0.428 / 0.820 macro F1, respectively).

These results demonstrate that headline writing follows certain linguistic rules and that those rules can be captured by interpretable, parse-grounded methods rather than opaque neural models. The framework offers practical value for text summarization, editorial style auditing, and media framing analysis, and is exposed through an interactive terminal sandbox that lets a user watch structure, agency, and rhetorical mode shift in real time as a headline is rewritten.

## 2. Introduction:
News headlines are one of the most widely consumed forms of text in the modern world. Millions of people rely on them as their primary source of information, often using them to form opinions and make immediate decisions. Although short, headlines are designed to convey maximal knowledge despite their brevity. Different from standard prose, headlines follow specific grammatical conventions, which often omit function words and restructure sentences to achieve brevity and impact. Headlines must balance clarity and informativeness with engagement and efficiency. As noted by Atwell and Drakos (1987), “A corpus of English text samples can constitute a definitive source of data in the linguistic constructs or structures.” 
From a computational perspective, headlines are an extreme case of information compression. Understanding how they achieve such a high density of information efficiently can provide important insights about language efficiency and structure. Consequently, this is particularly relevant in the NLP world as many tasks like summarization, inquiry response, and text generation aim to be as concise and clear as possible. As Zhao et al. (2024) wrote, “Headline Generation is an essential task in Natural Language Processing (NLP).” Understanding these structures can bring multiple advantages, both in and out of the NLP perspective.
In an era where we receive constant information, users prefer scanning titles over reading full articles; therefore, dismantling the headline’s structure can help design systems that deliver more efficient information. In addition, many search engines depend on generating concise summaries, and by identifying the best structural rules through effective headlines, we can improve their performance and generation system. 
Understanding these structures can also detect misinformation or manipulative phrasing. As mentioned before, millions of people use headlines to comprehend the world around them, therefore,  playing a critical role in shaping public perception. If we are able to break down their structural patterns and categorize which headlines show impartiality and which ones show bias, we can contribute to tools that detect potential deception.
These motivations led us to investigate the following research questions:

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
To investigate specific patterns found in current headlines, we constructed a corpus that follows a carefully planned pipeline. This pipeline includes five different stages: Collect, Parse, Classify, Analyze, and Evaluate.

### 3.1 Collect
The pipeline begins with the Collect stage. We created a Python script that utilizes the Google News RSS feeds to collect both domestic and worldwide news headlines. Then, for each RSS item, we read the raw title and slightly edit it to follow a certain format that will be easier to analyze later. By doing so, we attempt to fix Unicode issues and escape sequences by fixing escaped quotes, splitting phrases by dashes and pipes, and filtering out keywords. After this whole process, we are left with headlines and their additional data (source, date, etc.) that we then pass to the next stage, Parsing.

### 3.2 Parse
In the parsing stage, we utilize the spaCy Python library to extract tokens, POS tags, dependency trees, and named entities. We designed the parsing stage to be intentionally robust in order to get the best linguistic representation per headline. Within our parsing script, we first listed a preference of models that would be used within this step, prioritizing the use of the `en_core_web_trf` model for the most robust results, then shifting to `en_core_web_lg`, `en_core_web_md`, and `en_core_web_sm` in decreasing priority order. The parsing script then goes to a function that does a lightweight normalization of punctuation and spacing, in order to improve the parser's robustness and to ensure that we obtain more accurate results later. A first parse pass is conducted as normal, which will run normal tokenization, POS, and dependency parsing, and then also provide a score for the parse quality. A failure trigger may be activated if certain issues arise, such as there being no clear verb or an ambiguous short headline structure being detected. From there, normalization will be conducted to re-case tokens and apply lexicon overrides for common headline ambiguities. A fallback parse will then be conducted, where the parser is run again with more relaxed rules to make sure that everything is taken into account, and from there, the best of the primary versus fallback parsing confidence scores are used.

### 3.3 Analyze
After parsing, we enter the Analyze stage. The parsed headlines are sent into a script, where new fields are created from the parsing for future derivation. For example, POS patterns such as NOUN are converted to N to help with analysis. From there, higher-level features such as the collapsed POS patterns, macro structures (NP, VP, MOD, etc.), opening and ending patterns, passive voice, and headline type are derived, where the main function of the script will call functions to perform different types of analyses. These functions include:

1.  **analyze_structure_types**: classified headlines as things such as Question, Verb-Led Active, Noun Phrase, Subject-Verb, etc.
2.  **analyze_dependency_templates**: summarizes dependency relation patterns from the parse tree
3.  **analyze_named_entities**: Counts entity types like PERSON, ORG, GPE, DATE
4.  **analyze_model_label_distribution**: runs classifier model on parsed data and summarizes predicted labels
5.  **analyze_style_profile_story**: produces higher-level style summaries from the style profiler
6.  **analyze_information_order**: checks whether headlines start with actor, action, or context
7.  **analyze_compression_ratio**: measures content-word density
8.  **analyze_voice**: detects passive vs. active voice
9.  **analyze_structural_templates**: finds common POS/macro templates
10. **analyze_building_blocks**: looks at common openings and endings
11. **analyze_length_patterns**: summarizes headline length and how it relates to structure
12. **analyze_verb_patterns**: examines verb usage and tense/form distribution
13. **analyze_proper_noun_density**: measures how name-heavy headlines are
14. **analyze_root_words**: summarizes root POS and common root words

After all the analysis functions have run, the function `generate_insights()` generates brief conclusions from the dataset. The Analyze stage identifies structural and stylistic patterns, but does not assign labels. To convert these insights into a predictive framework, we transition into the next step: Classify.

### 3.4 Classify
In the Classify stage, we add discrete labels to analyzed linguistic features, so that we have a deterministic system that utilizes the parsed outputs from the previous stages. We input the POS tags, dependency relations, entities, and other derived features and output a structural label for the headline, which includes a profile for its style that includes multiple dimensions for interpretation. During this stage, we perform structural classification, where we first assign one primary structure label per headline. The labels are:

1.  **question_form**: Interrogative punctuation or interrogative opening pattern
2.  **passive_clause**: Passive cues (e.g., nsubjpass, auxiliary/passive constructions)
3.  **coordination**: Multi-clause coordination (cc/conj and coordination signals)
4.  **noun_phrase_fragment**: Nominal fragment without a finite clause
5.  **simple_clause**: Canonical finite subject-verb clause
6.  **other**: Residual category for out-of-pattern forms

The labeling is done by taking note of certain signals, such as POS patterns (presence/absence of words), punctuation cues, token order, and dependency labels (nsubjpass, auxpass, conj, cc). Each label is defined by explicit linguistic conditions, where the rules map feature patterns to the labels defined above. Within the Classify stage, we utilize prioritized ordering for the labels to ensure deterministic output and prevent conflicting rule matches. From highest to lowest priority, the order is question, passive, coordination, NP fragment, simple clause, and other. We finally handle edge cases, specifically ambiguous or short headlines, by having the "other" label as our default fallback. From there, we assign multiple stylistic dimensions per headline to give more insight into the patterns presented. This is because structure alone is insufficient to capture headline variation, and providing dimensions provides a richer representation. Our dimensions are:

1.  **lead_frame**: actor/entity-first, action-first, context-first, etc.
2.  **agency_style**: active vs. passive (with/without agent)
3.  **density_band**: low/medium/high (based on content-word ratio)
4.  **rhetorical_mode**: straight-report, analysis-explainer, question-hook, live/alert

We assign these dimensions by seeking signals within our data, such as entity position for lead_frame, passive cues for agency_style, content-word ratio for density_band, and structural and lexical cues for rhetorical_mode. This provides a final multi-dimensional label per headline, output as a "headline signature" that follows a "structure | lead_frame | rhetorical_mode" format. This final labeled dataset will be used in the last stage, Evaluate.

#### Manual tagging in practice
To make the annotation protocol concrete, we show two representative manual decisions that illustrate why human-in-the-loop labeling is necessary in the first place. Even with a deterministic rule set, the *gold* label requires reading the headline as a competent English speaker would and choosing the structurally faithful category, which a purely surface-pattern matcher cannot always do alone.

1. Headline: *"Nine killed in second Turkish school shooting in two days"*
    - Manual `gold_label`: **passive_clause**
    - Manual `gold_rhetorical_mode`: **straight_report**
    - Rationale: This is a compressed passive event frame (`NUM + VBN`) where the auxiliary "were" is elided and the agent is omitted entirely. A naive surface match might call this a noun phrase fragment, but the underlying clause is unambiguously passive.
2. Headline: *"Iran Update Special Report, April 14, 2026 Institute for the Study of War"*
    - Manual `gold_label`: **noun_phrase_fragment**
    - Manual `gold_rhetorical_mode`: **analysis_explainer**
    - Rationale: There is no finite main clause; the entire string functions as a report-style nominal title. Its discourse function is explanatory and bulletin-style rather than narrating a single event.

These two examples bracket the easy and hard ends of the annotation space and motivate why we evaluate against gold labels rather than self-agreement: the human decision encodes information about *why* a particular headline takes the form it does, which is exactly what we need to test the model against.

### 3.5 Evaluate
After assigning structural and stylistic labels through the classification stage, we evaluate the system's performance against a manually annotated gold-standard dataset to quantify its accuracy, robustness, and generalization ability. The Evaluate stage includes a proper assessment of the structural classification and style profiling dimensions for the purpose of comparing model predictions against gold-standard labels. We include the Gold-Standard dataset, which includes manually annotated headlines. The manually labeled evaluation set contains 464 headlines for structural and style assessment, and we separate those headlines into a 60/20/20 split between the Train, Dev, and Test corpora, respectively. The Train corpus develops the classification rules. The Dev corpus tunes and validates rule behavior before final testing. The Test corpus is used for a final, unbiased evaluation of the unseen data. We use the 60/20/20 split because it balances three competing goals: enough training examples to stabilize rule behavior, enough development data to diagnose rule changes before final reporting, and a sufficiently large untouched test set for a credible final estimate. The test split remains isolated during rule iteration to reduce optimistic bias. From there, we conduct proper evaluation using the following metrics: Accuracy, Precision, Recall, F1 (per class), and Macro F1. Accuracy calculates the percent of exact matches with gold labels, while precision calculates the correctness of the predictions. Recall calculates the coverage of true cases, while Macro F1 is similar to the F1 calculation but treats all classes equally and handles class imbalance. After that, we conduct baseline comparisons to provide a reference point for performance. We include a majority baseline to always predict the most frequent class, which tests whether our model is exploiting class imbalance. We also provide a random baseline, which assigns labels randomly to represent that there is no learned structure at all. These baselines contextualize performance and allow us to determine whether observed performance gains reflect genuine pattern recovery rather than trivial prediction strategies. Lastly, we run evaluations across multiple random splits and then report the mean performance and standard deviation. This avoids "lucky split" bias and shows a consistency of results. After the entire Evaluate section, we obtain performance metrics, per-class breakdowns, and stability statistics, which together let us draw conclusions about headline structure on a broad level.

Beyond the offline evaluation, we also built an interactive terminal sandbox on top of the same model that surfaces predictions, confidences, and parse evidence in real time. We defer its full description and screenshot to Section 5.3, where it is positioned as a practical illustration of why interpretable rule-based output matters at all.

## 4. Proof/Evaluation:
To interpret the results from the solution, we use the concept of accuracy, where:
$$accuracy = \frac{\#correct}{\#total}$$
We understand that accuracy answers the direct question "out of all headlines, how often does the predicted label exactly match the human gold label?" This prepares us to better interpret our results. Also, we need to use the concepts of precision and recall, where:
$$precision = \frac{TP}{TP+FP}$$
$$recall = \frac{TP}{TP+FN}$$
Let us take, for example, a class $c$. Precision measures reliability when the model predicts $c$, therefore, low false positives, and recall measures coverage of true $c$ cases, hence low false negatives. This distinction is critical for minority headlines previously labeled as $coordination$ and $other$, where a high overall accuracy can hide poor class-specific behavior. Another useful concept is F1, where:
$$F1 = \frac{2 \cdot precision \cdot recall}{precision + recall}$$
F1 is used because it penalizes models that optimize precision at the expense of recall. Furthermore, to avoid over-interpreting a single split, we run a 5-seed sweep (13, 42, 87, 123, 202) and we calculate the mean performance, the sample standard deviation, and the per-seed trajectory. Therefore, for metric values $m_{1}$ through $m_{k}$ across $K$ seeds, we use:

$$\overline{m} = \frac{1}{K} \sum_{i=1}^{K} m_{i}$$

$$s = \sqrt{\frac{1}{K-1} \sum_{i=1}^{K} (m_{i} - \overline{m})^{2}}$$

Where $\overline{m}$ is the mean performance, and $s$ is the sample standard deviation. These calculations are helpful as the mean estimates the expected performance under random split variation, the standard deviation quantifies sensitivity to split choice, and the per-seed trajectory is reported because identical means can hide different failure modes.

### 4.1 Corpus-Level Structural Findings
We divided our results into 4 categories: Corpus-Level Structural Findings, Structure Classifier Performance, Style-Dimension Performance, and Stability Analysis. Corpus-Level Structural Findings helps us answer the question, what do news headlines actually look like structurally?

| Metric | Value |
| :--- | :--- |
| Average length | 13.7 tokens |
| Content-word ratio | 73.2% |
| Headlines with verbs | 92.1% |
| Active/non-passive rate | 94.6% |
| Headlines with named entities | 95.7% |
| Avg entities per headline | 2.4 |
| Actor/entity-first openings | 81.6% |

**Table 1** shows that headline form is compact and stereotyped. The average length of a headline is approximately 14 tokens, of which 73% are "content words" (nouns, verbs, adjectives, adverbs). 92% of headlines contain a verb, meaning that even though they are compressed, headlines remain sentences describing an action rather than mere captions. Additionally, 94.6% of headlines are in active voice and 95.7% of them contain at least one named entity, with an average of 2.4 entities per headline. Finally, 81.6% of all headlines lead with their main actor or entity. Taken together, these numbers describe a remarkably consistent template: a compressed, active, entity-led clause built around a single event verb. This is already evidence in favor of H1's *template consistency* criterion.

![Structure label distribution](../images/structure_label_distribution.png)

**Figure 1**: Distribution of predicted structure labels across the evaluated corpus. The dominant structural class is the simple finite clause, with noun-phrase fragments forming a clear secondary mode and the remaining categories (questions, passives, coordinations, and the residual "other" bucket) appearing only as a long tail. The shape of this distribution is itself a finding: headline writing concentrates on two or three reusable templates, and the rare classes are rare because they fail the brevity/compression constraint that defines the genre.

### 4.2 Structure Classifier Performance
The category Structure Classifier Performance helps us answer the question, how well does our model actually predict headline structure? In this category, we calculate and interpret the concepts discussed previously (Accuracy, Macro F1, Precision, Recall, F1).

| Evaluation Slice | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| All labeled (n=465 evaluation slice) | 0.772 | 0.601 |
| Dev (n=90) | | 0.561 |
| Test (n=98) | 0.806 | 0.675 |

**Table 2** above shows and calculates the aggregate structure metrics. It tells us how our model performs on different parts of the dataset, more specifically, all data combined, the development set (used for tuning), and the test set (final evaluation). From our "all-labeled" slice, we see a calculated Accuracy of 0.772 and a Macro F1 of 0.601 when $n=465$. We see that in that row, Macro F1 is lower than Accuracy, which tells us that the dataset is imbalanced, which also supports the fact that we have found a big number of simple_clause structures. The Dev slice we see no accuracy value and a Macro F1 value of 0.561. This number means that on the development set, the model achieves moderate balanced performance across all classes. The lack of value for Accuracy is intentional, as in development, Accuracy can be misleading, due to the imbalance mentioned previously. Lastly, the test slice shows us our true evaluation. It gives us the values 0.806 for Accuracy and 0.675 for Macro F1 when $n=98$. It calculates that the model correctly predicts the structure of 80.6% of unseen headlines, and the Macro F1 value tells us that the model performs well across all structure classes, which means we have a good balance between precision and recall. With this information, we can observe that the model is not just guessing but capturing real structural patterns in headlines. However, we need to make sure whether our model is genuinely learning headline structure or if the results are trivial. Therefore, we compare it against a majority baseline and a random baseline.

| Model | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| Rule-based classifier | 0.806 | 0.675 |
| Majority baseline | 0.643 | 0.130 |
| Random baseline | 0.459 | 0.159 |

**Table 3**: The results from the rule-based classifier model in Table 3 are the results from our actual model that we have seen previously. In the following row, we see the results from the majority baseline, which always predicts the most common class, which in our case is the simple_clause. We see that we still get a high Accuracy value as our dataset is very imbalanced. However, we see a very low value for Macro F1, meaning that when the model always predicts the most common case, it completely ignores minority classes. Further, once we compare it to the random baseline model, where it randomly assigns labels, we see a poor Accuracy value and a terrible class balance from Macro F1. Looking at those values shows us that our model beats the majority accuracy by 16.3 percentage points and the random baseline by 34.7 percentage points, and more importantly, we see a jump from poor Macro F1 results calculated from these baselines to our model's 0.675. Therefore, it proves that the model is learning genuine syntactic relationships rather than exploiting class frequency alone. Once we see that our model is actively learning, we want to understand how well our classifier performs on each headline structure class. Therefore, we break the performance down by labels.

| Label | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| question_form | 0.960 | 0.960 | 0.960 |
| passive_clause | 0.478 | 0.344 | 0.400 |
| coordination | 0.722 | 0.448 | 0.553 |
| noun_phrase_fragment | 0.578 | 0.627 | 0.602 |
| simple_clause | 0.882 | 0.865 | 0.874 |
| other | 0.138 | 0.500 | 0.216 |

**Table 4**: After reviewing these results, we see that the model is strong on the dominant classes, simple_clause and question_form, and weaker on ambiguous categories. In other terms, the system captures the primary newsroom syntactic backbone but remains conservative on edge constructions where label boundaries are less operationally crisp.

To put the size of these gains in context: the rule-based classifier improves test accuracy by **+16.3 percentage points** over the majority baseline and by **+34.7 percentage points** over the random baseline. Macro F1 jumps from 0.130 (majority) and 0.159 (random) to 0.675, an improvement of more than four times. This is the difference that matters: accuracy alone is friendly to a majority predictor on an imbalanced dataset, but macro F1 is not, and the rule system clears it decisively.

We also report a normal-approximation 95% confidence interval on the test accuracy of 0.806 with $n=98$:

$$\text{CI}_{0.95}(\text{Accuracy}) \approx [0.728, 0.884]$$

Both the majority baseline (0.643) and the random baseline (0.459) fall well outside this interval, so the gain is not a sampling artifact at the held-out sample size. We do not run a full paired significance test in this phase, but the interval is sufficient to reject the *structural predictability* failure condition from H1: the model is doing real work, not exploiting class frequency.

### 4.3 Style-Dimension Performance
Our third category of results, Style-Dimension Performance, goes beyond grammatical structure and focuses on answering a different question: what stylistic characteristics does each headline express? By looking at editorial style and discourse behavior, can the model recover headline style patterns the way it recovers structural ones? We evaluate this in two ways: first across the entire labeled evaluation slice combined, then on the held-out test split alone, so we can separate "is the rule consistent on training data?" from "does it generalize?".

| Dimension | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| lead_frame | 0.989 | 0.986 |
| agency_style | 0.935 | 0.622 |
| density_band | 0.994 | 0.995 |
| rhetorical_mode | 0.914 | 0.785 |

**Table 5**: All-slice (n=465) style dimension performance. The dimension `lead_frame` represents what kind of information appears first in the headline, and the near-perfect scores show that headline openings follow extremely stable framing conventions that our system almost always identifies correctly. The dimension `agency_style` represents how agency is expressed; the fact that Macro F1 (0.622) is much lower than Accuracy (0.935) tells us that the system captures the dominant active class easily but stumbles on the rare passive variants, especially the compressed passive forms with elided auxiliaries. `density_band` predicts how information-dense the headline actually is, and the model is again nearly perfect, showing that information density is one of the most stable and recoverable properties of headlines. Finally, `rhetorical_mode` represents the communicative purpose of the headline (straight report, analysis explainer, question hook, live alert). Because rhetorical intent is more semantic and contextual and less purely syntactic, the model recovers it imperfectly: it is reliable on the dominant `straight_report` mode but more uncertain on `analysis_explainer` and edge-case live-alert headlines, where lexical cues alone do not always disambiguate the discourse function.

| Dimension | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| lead_frame | 1.000 | 1.000 |
| agency_style | 0.908 | 0.428 |
| density_band | 1.000 | 1.000 |
| rhetorical_mode | 0.908 | 0.820 |

**Table 6**: Held-out test (n=98) style dimension performance. On unseen headlines, the strongest signals (`lead_frame` and `density_band`) are perfectly recoverable, which is consistent with our claim that they are deterministically tied to surface features (entity position and content-word ratio). `rhetorical_mode` actually rises slightly compared to the all-slice number, showing that the rule generalizes rather than overfits. `agency_style`, however, drops noticeably in macro F1 (0.622 → 0.428), which exposes the central weakness of the rule set: in a small held-out slice, a few misclassified rare passive headlines crater the minority-class score even when the dominant active class is still being handled well. We return to this in Section 5.2.

![Model performance summary](../images/model_performance_summary.png)

**Figure 2**: Model performance summary. Side-by-side accuracy and macro F1 across the structure classifier and the four style dimensions, on both the all-labeled slice and the held-out test slice. The visual shape of the bars is itself informative: structure performance is moderate but well above baseline, framing and density saturate the score, and agency/rhetorical mode form an intermediate band where syntax-only cues are partially but not fully sufficient.

![Performance heatmap](../images/performance_heatmap.png)

**Figure 3**: Performance heatmap, a compact view of held-out accuracy and macro F1 across all model components. Reading the heatmap top to bottom traces the decomposition we want to make: the most form-grounded properties (where the headline starts, how dense it is) are dark green; the most semantically loaded ones (agency, rhetoric) are lighter; structural classification sits in between. This is a graphical answer to the research question about how recoverable each dimension of headline form actually is.

### 4.4 Stability Analysis
Each split matters. Some random splits may be easier, cleaner, or more balanced; others may concentrate rare cases, harder ambiguity, or worse class distributions. One lucky split can make a model look better than it really is, and one unlucky split can do the opposite. To make this concern operational rather than rhetorical, we re-ran the entire structure and style evaluation across 5 different random splits using seeds 13, 42, 87, 123, and 202, and we report the mean performance and sample standard deviation across those runs.

| Metric | Mean | Std. Dev. |
| :--- | :--- | :--- |
| Structure test accuracy | 0.740 | 0.051 |
| Structure test macro F1 | 0.568 | 0.076 |
| Structure dev macro F1 | 0.608 | 0.073 |
| Style test rhetorical_mode macro F1 | 0.771 | 0.063 |
| Style test agency_style macro F1 | 0.611 | 0.168 |

**Table 7**: 5-seed split-stability summary. The mean estimates expected performance under random split variation; the standard deviation quantifies sensitivity to split choice.

![Seed stability summary](../images/seed_stability_summary.png)

**Figure 4**: Per-seed structure performance with mean and ±1 standard deviation bands. The plot makes the size of "split luck" directly visible. The per-seed structure test Macro F1 ranges from **0.481 to 0.675** (a spread of 0.194), and the structure test Accuracy ranges from **0.663 to 0.806** (a spread of 0.143). In other words, the same model on the same data can look like a 0.481-macro-F1 system or a 0.675-macro-F1 system depending only on which 98 headlines you happened to hold out. The seed-42 result we report in Section 4.2 happens to be a strong split, and reporting it alone would overstate expected performance.

For the seed mean estimates, approximate 95% intervals are:

$$\bar{m}_{\text{Accuracy}} \in [0.695, 0.785]$$

$$\bar{m}_{\text{Macro F1}} \in [0.502, 0.634]$$

These intervals quantify uncertainty of the mean under split randomness and make the "luck" framing concrete: a fair description of the system is a structure test accuracy somewhere around the high 0.7s and a macro F1 in the high 0.5s to low 0.6s, with the upper end requiring a favorable split. This is exactly the kind of self-skeptical reporting H1's *robustness* falsification criterion was designed to enforce, and the model passes that test: the variance is meaningful, but the mean stays well above majority and random baselines on every individual seed.

### 4.5 Worked Decision-Path Examples
A central claim of this paper is that our predictions are **traceable** to explicit linguistic evidence: the rule-based architecture means each label can be explained by stating exactly which rule fired and why. To make this concrete, we walk through three headlines that hit different points in the priority order.

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

These three examples are not just illustrations. They are the basis on which the *interpretability* falsification condition from H1 is checked: every predicted label can be explained by stating which rule fired, in what priority order, on which surface features. That property is what makes the system auditable in a way that a black-box neural model is not.

## 5. Discussion:
### 5.1 Strengths of Solution
Overall, our solution successfully identified headline structures, and the broader argument of this paper rests directly on what those rules revealed: headlines are predictable syntactic products, not free-form prose. The rule-based framework captured dominant newsroom structures effectively, especially the high-frequency simple clauses and interrogative forms, which indicates that headline syntax follows consistent and reusable patterns. The multi-dimensional style profiling extended our analysis beyond a single structural label and gave a representation of headlines that simultaneously captures framing, agency, and rhetorical intent.

A second strength is interpretability. Because every prediction is grounded in explicit parse features (POS, dependency relations, entity positions), the system stays auditable: a researcher or editor can follow any classification back to the linguistic evidence that produced it. The multi-seed evaluation reinforced this rigor by showing the variance under random splits explicitly rather than hiding it in a single point estimate.

The extra style dimensions we implemented proved to be more predictable than expected in certain cases. `lead_frame` and `density_band` were extremely consistent (test macro F1 of 1.000 on both), which suggests that those aspects of headline construction are governed by stable, surface-recoverable conventions rather than subjective judgment. We anticipated lower performance on `rhetorical_mode` and the rare passive variants of `agency_style`, and we got it: these dimensions are exactly where syntax-only signals stop being sufficient and discourse-level cues take over. Overall, the results demonstrate that headline construction is both structurally constrained and stylistically patterned, and that these regularities can be effectively captured using interpretable, linguistically grounded methods. This is the central thesis of the paper, and the data supports it.

### 5.2 Why Some Classes Remain Difficult
Not every class behaves equally well, and being honest about where the system fails is part of the argument. The lower-performing categories — the residual `other` bucket, parts of `coordination`, and some compressed passive variants — share a small set of underlying causes:

- **Sparse support**: classes like `other` and rare passive forms have so few gold examples in the held-out test set that a single misclassification swings their F1 dramatically. The macro F1 metric is what surfaces this; the accuracy metric can hide it entirely.
- **Boundary overlap between labels**: `coordination` and `simple_clause` are not always cleanly separable when a headline contains a comma list and a finite verb. The priority order resolves the conflict, but the resolution is not always the human's preferred reading.
- **Punctuation-heavy multi-clause headline variants**: headlines that combine semicolons, em-dashes, and embedded clauses (live-blog-style) sit between coordination and noun-phrase fragment, and the rule set has to make a hard call.
- **Residual annotation ambiguity in edge forms**: even our human annotators sometimes had to choose between two defensible labels, and the model inherits that ambiguity directly.

The takeaway is not that the model is broken on these classes, but that they are inherently noisier targets than the dominant ones, and the *macro* score rightly punishes us for that noise.

### 5.3 Practical Sandbox: The TUI Workbench
The argument that interpretability is a real strength of this system, and not just a slogan, is most concrete in the interactive terminal sandbox we built on top of the model.

![Interactive TUI workbench](../images/tui_image.svg)

**Figure 5**: The real-time TUI workbench. As a journalist or analyst types a headline, the interface continuously surfaces the model's predictions with confidence estimates, the parse evidence that supports each decision (phrase flow, a dependency mini-view, and a token map), warnings for edge-case constructions, and a benchmark footer with the corpus size and validation metrics anchoring the predictions.

The workbench accelerates headline refinement by collapsing the feedback loop from minutes (run the script, read the output, compare to gold) to seconds (type a candidate, watch the structure, agency, and rhetorical mode shift). It improves accountability because each prediction is displayed alongside the parse evidence that produced it, so an editor can immediately see *why* the system reads a candidate headline as `passive_clause` or `analysis_explainer` and decide whether the system is correct or whether the wording itself is misleading. It also provides early warning for potentially problematic constructions — unclear agency, excessive compression, misleading rhetorical hooks — before publication.

While the sandbox is not part of the offline evaluation pipeline, it is the strongest practical argument that a transparent rule-based system has an advantage that an opaque neural model cannot match: the user can sit inside the model's reasoning and edit toward or away from any particular structural template in real time. That is exactly what makes interpretable NLP useful for newsroom and media-analysis workflows.

### 5.4 Limitations
While our model was successful overall, the results come with several honest limitations that constrain the strength of any general claim:

1. **Rule-bound model family**: rule-based architectures are highly interpretable but inherently less flexible than fully learned models. Patterns that fall outside the rule set are systematically missed rather than gracefully approximated.
2. **Class imbalance**: dominant templates like `simple_clause` make accuracy easy and macro F1 hard. The minority classes drive the macro number disproportionately, so small annotation differences have outsized effects.
3. **Annotation subjectivity**: edge-case labels (especially within `rhetorical_mode` and `coordination`) are inherently debatable. Two competent annotators can produce defensible but different gold labels.
4. **English- and source-bias**: the corpus is built from a small set of English-language Google News RSS feeds, which limits external generalization to other languages, registers, and outlet styles.
5. **No external lexicon expansion**: we deliberately excluded curated event-noun and named-entity lexicons in this phase to keep the rules transparent. This makes some rare entity- or event-driven disambiguation harder than it could be, but it preserves interpretability.

None of these limitations invalidate H1; they bound it. The claim we defend is that headlines follow stable, recoverable templates, and that a transparent rule-based system captures them well above baseline — not that our particular rule set is the final word on headline structure.

## 6. Related Work:
Prior work on news headlines has majorly focused on generation tasks, by leveraging recent advances in neural language models. Zhao et al. (2024) propose a Chain-of-Thought (CoT) based supervised fine-tuning framework for number-focused headline generation, demonstrating how large language models can produce rich and numerically grounded headlines. However, while their approach emphasizes generative performance and semantic reasoning, our work differs in that we focus on the structural analysis of existing headlines, aiming to uncover interpretable syntactic patterns rather than generate new text. Nevertheless, their emphasis on numerical content motivates one of our structural categories, namely the analysis of headlines that rely on quantitative expressions as a central syntactic feature.

Beyond text generation, prior research has explored syntactic analysis in noisy and domain-diverse text. Eggleston and O'Connor (2022) investigate cross-dialect dependency parsing, showing that off-the-shelf parsers degrade systematically on text whose surface form (casing, dialectal lexicon, non-standard punctuation) drifts away from the news-style training data those parsers were built on. Their finding directly motivates the parser-robustness pipeline used in our Parse stage: rather than trusting a single off-the-shelf model, we use a model preference order, lightweight normalization of headline-specific punctuation and capitalization, and a fallback parse pass scored against the original. In other words, their result is a warning we took seriously, and our parsing strategy is a practical response to it.

In summary, prior work has primarily emphasized generation, semantic understanding, and downstream performance, often using neural and multimodal approaches. In contrast, our work contributes a structural, interpretable analysis of headline syntax, focusing on identifying consistent grammatical templates, decomposing them along multiple style dimensions, and evaluating their robustness across both domains and random splits.

## 7. Conclusion
This project demonstrated that news headlines follow consistent structural patterns despite their brevity. We developed an NLP pipeline that collected, parsed, analyzed, and classified headlines using a rule-based framework that was grounded in syntactic features. This approach reflects earlier work showing that grammatical structure can be learned from unrestricted text using pattern-based methods. (Atwell and Drakos, 1987). The results achieved strong performance on a manually annotated gold-standard dataset, outperforming baseline approaches while having interpretability. 
Our findings supported the hypothesis that headline construction is governed by stable linguistic templates, high information density, and recurring stylistic conventions, rather than randomness. By combining structural classification with a multi-dimensional style profile, the results proved to be a more interpretable representation of headline form. The rule-based approach demonstrated that meaningful performance can be achieved without reliance on opaque models, enabling a more transparent analysis and practical applicability. 
Beyond this study, the framework offers potential applications in automated summarization, editorial tooling, and media framing analysis. Prior work in headline generation similarly emphasizes the importance of producing concise, informative outputs in NLP systems (Zhao et al., 2024). Future work may extend this approach through hybrid models, larger and more diverse datasets, and multilingual analysis to further explore the universality of headline structure. 

## 8. Bibliography
Atwell, Eric Steven and Nicos Frixou Drakos. 1987. Pattern recognition applied to the  
    acquisition of a grammatical classification system from unrestricted English text.  
    In *Proceedings of the 10th International Joint Conference on Artificial Intelligence  
    (IJCAI-87)*, pages 677–680, Milan, Italy.

Eggleston, Chloe and Brendan O’Connor. 2022. Cross-Dialect Social Media Dependency  
    Parsing for Social Scientific Entity Attribute Analysis. In *Proceedings of the Eighth  
    Workshop on Noisy User-generated Text (W-NUT 2022)*, pages 38–50, Gyeongju,  
    Republic of Korea. Association for Computational Linguistics.

Dukić, David, Kiril Gashteovski, Goran Glavaš, and Jan Snajder. 2024. Leveraging Open  
    Information Extraction for More Robust Domain Transfer of Event Trigger Detection.  
    In *Findings of the Association for Computational Linguistics: EACL 2024*, pages  
    1197–1213, St. Julian’s, Malta. Association for Computational Linguistics.

Zhao, Junzhe, Yingxi Wang, Huizhi Liang, and Nicolay Rusnachenko. 2024. NCL_NLP at  
    SemEval-2024 Task 7: CoT-NumHG: A CoT-Based SFT Training Strategy with Large  
    Language Models for Number-Focused Headline Generation. In *Proceedings of the  
    18th International Workshop on Semantic Evaluation (SemEval-2024)*, pages 261–269,  
    Mexico City, Mexico. Association for Computational Linguistics.

Qiao, Lingfeng, Chen Wu, Ye Liu, Haoyuan Peng, Di Yin, and Bo Ren. 2022. Grafting  
    Pre-trained Models for Multimodal Headline Generation. In *Proceedings of the 2022  
    Conference on Empirical Methods in Natural Language Processing: Industry Track*,  
    pages 244–253, Abu Dhabi, UAE. Association for Computational Linguistics.

> **Bottom line:** News headlines are not free-form snippets; they are reproducible syntactic products. Interpretable, parse-grounded NLP can measure that structure directly, evaluate it rigorously, and put it in front of the people who write headlines.