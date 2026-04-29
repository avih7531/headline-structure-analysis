To ensure the larger mathematical equations are easy to read, I have placed them on their own lines using "display math" syntax (using double dollar signs `$$...$$`). This centers the equations and provides proper spacing.

Here is the complete file in Markdown:

# Structural Patterns in News Headlines: A Computational Linguistic Analysis Using NLP

**Victor Derani, Avi Herman, Kylie Lin**

## 1. Abstract:
This paper presents an analysis of syntactic and structural patterns in news headlines through Natural Language Processing (NLP). We built a pipeline that collects headlines from RSS feeds, processes them by using the spaCy library, and extracts part-of-speech (POS) patterns, dependency structures, and named entities. We discovered that headlines follow optimized grammatical conventions, which include the ability to contain high information, dominant use of active voice, and consistent structures such as noun phrase and verb phrase constructions. We also compare domestic and international news headlines and found there is minimal variation in structural patterns. These findings demonstrate that headline writing follows certain linguistic rules, which offer insight for text summarization and media analysis, both of which explain trends in audience engagement across news outlets.

## 2. Introduction:
News headlines are one of the most widely consumed forms of text in the modern world. Millions of people rely on them as their primary source of information, often using them to form opinions and make immediate decisions. Although short, headlines are designed to convey maximal knowledge despite their brevity. Different from standard prose, headlines follow specific grammatical conventions, which often omit function words and restructure sentences to achieve brevity and impact. Headlines must balance clarity and informativeness with engagement and efficiency. From a computational perspective, headlines are an extreme case of information compression. Understanding how they achieve such a high density of information efficiently can provide important insights about language efficiency and structure. Consequently, this is particularly relevant in the NLP world as many tasks like summarization, inquiry response, and text generation aim to be as concise and clear as possible. Understanding these structures can bring multiple advantages, both in and out of the NLP perspective. In an era where we receive constant information, users prefer scanning titles over reading full articles; therefore, dismantling the headline's structure can help design systems that deliver more efficient information. In addition, many search engines depend on generating concise summaries, and by identifying the best structural rules through effective headlines, we can improve their performance and generation system. Understanding these structures can also detect misinformation or manipulative phrasing. As mentioned before, millions of people use headlines to comprehend the world around them, therefore, playing a critical role in shaping public perception. If we are able to break down their structural patterns and categorize which headlines show impartiality and which ones show bias, we can contribute to tools that detect potential deception. These motivations led us to investigate the following research questions:

* What grammatical structures are most common in news headlines? How do headlines achieve high information density?
* What syntactic templates dominate headline construction? Are these patterns consistent across different new domains?

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

### 3.5 Evaluate
After assigning structural and stylistic labels through the classification stage, we evaluate the system's performance against a manually annotated gold-standard dataset to quantify its accuracy, robustness, and generalization ability. The Evaluate stage includes a proper assessment of the structural classification and style profiling dimensions for the purpose of comparing model predictions against gold-standard labels. We include the Gold-Standard dataset, which includes manually annotated headlines. The manually labeled evaluation set contains 464 headlines for structural and style assessment, and we separate those headlines into a 60/20/20 split between the Train, Dev, and Test corpora, respectively. The Train corpus develops the classification rules. The Dev corpus tunes and validates rule behavior before final testing. The Test corpus is used for a final, unbiased evaluation of the unseen data. We use the 60/20/20 split because it balances three competing goals: enough training examples to stabilize rule behavior, enough development data to diagnose rule changes before final reporting, and a sufficiently large untouched test set for a credible final estimate. The test split remains isolated during rule iteration to reduce optimistic bias. From there, we conduct proper evaluation using the following metrics: Accuracy, Precision, Recall, F1 (per class), and Macro F1. Accuracy calculates the percent of exact matches with gold labels, while precision calculates the correctness of the predictions. Recall calculates the coverage of true cases, while Macro F1 is similar to the F1 calculation but treats all classes equally and handles class imbalance. After that, we conduct baseline comparisons to provide a reference point for performance. We include a majority baseline to always predict the most frequent class, which tests whether our model is exploiting class imbalance. We also provide a random baseline, which assigns labels randomly to represent that there is no learned structure at all. These baselines contextualize performance and allow us to determine whether observed performance gains reflect genuine pattern recovery rather than trivial prediction strategies. Lastly, we run evaluations across multiple random splits and then report the mean performance and standard deviation. This avoids "lucky split" bias and shows a consistency of results. After the entire Evaluate section, we receive performance metrics, per-class breakdowns, and stability statistics, which can then help us draw conclusions about headline structure on a broad level. Our project includes an interactive terminal workbench for real-time interpretability and error analysis. For each input headline, the interface exposes:

1.  Categorical predictions with confidence estimates
2.  Parse evidence (phrase flow, dependency mini-view, and token map)
3.  Warnings for edge-case headline constructions
4.  Benchmark footer with validation metrics and corpus size

This interface accelerates headline refinement by shortening the feedback loop from minutes to seconds. It also improves accountability by making model decisions auditable through explicit parse evidence rather than the opaque scores alone. Lastly, it provides early warning for potentially problematic constructions (e.g., unclear agency, excessive compression, or misleading rhetorical hooks), allowing editorial correction before the headline reaches readers. While not part of the official pipeline, this interface provides a more interactive aspect so that a user can use a proper real-time application, rather than running scripts separately.

## 4. Proof/Evaluation:
To interpret the results from the solution, we use the concept of accuracy, where:
$$accuracy = \frac{\#correct}{\#total}$$
We understand that accuracy answers the direct question "out of all headlines, how often does the predicted label exactly match the human gold label?" This prepares us to better interpret our results. Also, we need to use the concepts of precision and recall, where:
$$precision = \frac{TP}{TP+FP}$$
$$recall = \frac{TP}{TP+FN}$$
Let us take, for example, a class $c$. Precision measures reliability when the model predicts $c$, therefore, low false positives, and recall measures coverage of true $c$ cases, hence low false negatives. This distinction is critical for minority headlines previously labeled as $coordination$ and $other$, where a high overall accuracy can hide poor class-specific behavior. Another useful concept is F1, where:
$$F1 = \frac{2 \cdot precision \cdot recall}{precision + recall}$$
F1 is used because it penalizes models that optimize precision at the expense of recall. Furthermore, to avoid over-interpreting a single split, we run a 5-seed sweep (12, 42, 87, 123, 202) and we calculate the mean performance, the sample standard deviation, and the per-seed trajectory. Therefore, for metric values $m_{1}$ through $m_{k}$ across $K$ seeds, we use:

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

**Table 1** shows us that the headline form is compact. We see that the average length of a headline is approximately 14 words, while 73% of those words are what we call "content words" (nouns, verbs, etc.). Further, 92% of headlines contain a verb, meaning that even though they are compressed, headlines are still sentences describing an action. Additionally, we see that 94.6% of headlines are in active voice and 95.7% of them contain entities, with an average of 2.4 entities per headline. Lastly, we see that 81.6% of all headlines start with their main actor, showing a certain consistent structural template. Through these results, we were also able to create a graph distribution.

**Figure 1** shows us the distribution of predicted structure labels in the evaluated corpus. It is a bar chart showing how often each structural class appears. We see that 306 headlines, which is the vast majority of our sample, are simple clauses, which supports the idea that headlines are action-oriented and compressed sentences. The second most common structure is noun_phrase_fragment, which are not full sentences and work more similarly to titles and labels, for example, "Election Results 2026". The rest of the structures seem to be relatively rare, showing one more time that headline writing is actually very standardized.

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

**Table 4**: After reviewing these results, we see that the model is strong on the dominant classes, simple_clause and question_form, and weaker on ambiguous categories. In other terms, we see that the system captures the primary newsroom syntactic backbone but remains conservative on edge constructions where label boundaries are less operationally crisp.

### 4.3 Style-Dimension Performance
Furthermore, our third category of results, Style-Dimension Performance, goes beyond the headline grammatical structure and focuses on answering the question, what stylistic characteristics does this headline express? By focusing on editorial style and discourse behavior, can the model predict headline style patterns? First, we test to see how accurately the system can predict different style dimensions of headlines on the entire labeled evaluation slice combined.

| Dimension | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| lead_frame | 0.985 | 0.985 |
| agency_style | 0.938 | 0.622 |
| density_band | 0.994 | 0.985 |
| rhetorical_mode | 0.842 | 0.782 |

**Table 5**: The dimension lead_frame represents what kind of information appears first in the headline, and the near-perfect scores show that headline openings follow extremely stable framing conventions, and our system almost always identifies them correctly. The following dimension, agency_style, represents how agency is expressed, and the fact that the Macro F1 value is much lower than Accuracy tells us that the system captures agency patterns reasonably well, but compressed passive forms remain difficult. Another dimension calculated is density_band, which predicts how information-dense the headline actually is. We also observe that the model performance is again nearly perfect, showing us that information density is one of the most stable and recoverable properties of headlines. Lastly, we calculate the rhetorical_mode dimension, which represents the communicative purpose of the headline. As rhetorical intent is more semantic, contextual, and less purely syntactic...

### 4.4 Stability Analysis
Each split matters, and some random splits may be easier, cleaner, or more balanced, while others may contain more rare cases, harder ambiguity, or worse class distributions, one lucky split can make a model look better than it really is. Following this line of thought, we run 5 different splits using seeds 13, 42, 87, 123, and 202.

| Metric | Mean | Std. Dev. |
| :--- | :--- | :--- |
| Structure test accuracy | 0.740 | 0.051 |
| Structure test macro F1 | 0.568 | 0.076 |
| Structure dev macro F1 | 0.503 | 0.073 |
| Style test lead_frame macro F1 | 0.971 | 0.013 |
| Style test agency_style macro F1 | 0.411 | 0.153 |

**Table 7** and **Figure 2**: per-seed structure performance with mean and +/- 1 standard deviation bands. This quantifies split luck directly rather than relying on one split. We observe that the per-seed structure test Macro F1 ranges from 0.481 to 0.675, while the structure test Accuracy ranges from 0.663 to 0.806. This confirms that split choice can materially change both class balance and top-line reliability. In this run, seed 42 happens to be a strong split and would overestimate the expected performance if reported alone. For the seed mean estimates, approximate 95% intervals are:

$$\mu_{\text{Accuracy}} = [0.692, 0.788]$$

$$\mu_{\text{Macro F1}} = [0.495, 0.641]$$

These intervals quantify the uncertainty of the mean under split randomness and make the “luck” framing operational rather than rhetorical.

## 5. Discussion:
### 5.1 Strengths of Solution
Overall, our solution successfully identified headline structures. The rules showed that the headlines followed a predictable syntax pattern rather than being randomly generated. The rule-based framework captured dominant newsroom structures effectively, especially for high-frequency structures including simple clauses and interrogative forms, indicating that headline syntax follows consistent patterns. The addition of the multi-dimensional style profiling within our solution extended our analysis beyond simple and single labels. This style enabled a representation of headlines that captures framing, agency, and rhetorical intent. Our model grounded predictions in explicit parse features, including dependency relations, and by doing so, the system remained highly interpretable. This high interpretability can allow researchers to follow classifications back to linguistic evidence. Our multi-seed evaluation strengthened our methodology by demonstrating headlines as highly compressed informational units. The extra style dimensions that we implemented proved to be more predictable than expected in certain cases. Dimensions such as lead_frame and density_band were consistent, which suggests that stylistic conventions are systematic rather than subjective. The strong performance on certain style dimensions, specifically lead framing and density, suggests that these aspects of headline construction follow consistent and recoverable patterns. We anticipated lower performance on certain dimensions, such as rhetorical_mode. This is due to the limits of syntax-only approaches, as we cannot capture deep discourse intent through just words alone. Overall, the results demonstrate that headline construction is both structurally constrained and stylistically patterned, and that these regularities can be effectively captured using interpretable, linguistically grounded methods.

### 5.4 Limitations
While our model was successful overall, several limitations...

## 6. Related Work:
Prior work on news headlines has majorly focused on generation tasks, by leveraging recent advances in neutral language models. Zhao et al. (2024) propose a Chain-of-Thought (CoT), which is a based supervised fine-tuning framework for number-focused headline generation, demonstrating how large language models can produce rich and numerically grounded headlines. However, while their approach emphasizes generative performance and semantic reasoning, our work differs in that we focus on the structural analysis of existing headlines, aiming to uncover interpretable syntactic patterns rather than generate new text. Nevertheless, their emphasis on numerical content motivates one of our structural categories, namely the analysis of headlines that rely on quantitative expressions as a central syntactic feature. Beyond text generation, prior research has explored syntactic analysis in noisy and domain-diverse text. Eggleston and O'Connor (2022) investigate cross-dialect dependency... In summary, prior work has primarily emphasized generation, semantic understanding, and downstream performance, often using neural and multimodal approaches. In contrast, our work contributes a structural, interpretable analysis of headline syntax, focusing on identifying consistent grammatical templates and evaluating their robustness across domains.

## 7. Conclusion
This project demonstrated that news headlines follow consistent structural patterns despite their brevity. We developed an NLP pipeline that collected, parsed, analyzed, and classified headlines using a rule-based framework that was grounded in syntactic features. The results achieved strong performance on a manually annotated gold-standard dataset, outperforming baseline approaches while having interpretability. Our findings supported the hypothesis that headline construction is governed by stable linguistic templates, high information density, and recurring stylistic conventions, rather than randomness. By combining structural classification with a multi-dimensional style profile, the results proved to be a more interpretable representation of headline form. The rule-based approach demonstrated that meaningful performance can be achieved without reliance on opaque models, enabling a more transparent analysis and practical applicability. Beyond this study, the framework offers potential applications in automated summarization, editorial tooling, and media framing analysis. Future work may extend this approach through hybrid models, larger and more diverse datasets, and multilingual analysis to further explore the universality of headline structure. 