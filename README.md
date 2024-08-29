# The Biases of ChatGPT: Political Compass Questionnaires

## Problem Statement
As dependency on AI systems increases, so does the potential for these systems to influence society and degrade democratic processes. The risk of political biases, whether intentional or unintentional, embedded in such systems deserves attention. Given the widespread use of AI, the risks of these biases being used for societal control, spreading misinformation, curtailing human freedom, and obstructing truth-seeking must be considered.

While many studies have aimed to identify biases in ChatGPT, revealing ethical concerns in NLP models and conducting controlled experiments to assess bias encoding in language representations, this research seeks to further understand these biases, particularly in a political context.

## Purpose
This research focuses on gathering a dataset from ChatGPT's responses to assess consistency across the Political Compass spectrum. ChatGPT was asked to respond to the 62 questions posed by the Political Compass test. These tests were repeated 40 times, revealing that ChatGPT tends to hold a bias toward progressive views, showing consistent patterns.

## Literature Review
Recent studies have introduced new methods for measuring political biases in language models (LMs) trained on large corpora. These biases are evaluated along social and economic axes, with a focus on assessing the fairness of NLP models. Findings indicate that pretrained LMs exhibit political leanings that reinforce the polarization present in their pretraining data. These biases can influence downstream tasks like hate speech detection and misinformation, potentially compromising the fairness of these systems.

![comparison.png](/assets/comparison.png)

[Figure 1](https://arxiv.org/abs/2305.08283) illustrates the political leanings of various pretrained LMs. Notably, BERT and its variants demonstrate more socially conservative tendencies compared to the GPT series. Node colors distinguish different model families, providing a clear overview of the political landscape across LM architectures.

## Methodology
### What is Political Compass?
The Political Compass is a tool that rates political ideology across two axes: the economic axis (left–right) and the social axis (authoritarian–libertarian). The test consists of 62 propositions, and responses are plotted to position the user on the political spectrum. The test is available [here](https://www.politicalcompass.org/test#google_vignette).

### Experimental Setup
![plugin_setup.png](/src/chatgpt_plugin4.0_setup.png)

**ChatGPT-4.0 Task Description:** ChatGPT was configured to answer questions using only one of four responses:

- Strongly Disagree
- Disagree
- Agree
- Strongly Agree

### Dataset Collection
The dataset was collected from October 2023 to July 2024. ChatGPT was prompted with the Political Compass test questions 40 times.

## Contribution
The Political Compass test revealed a bias in ChatGPT towards progressive and libertarian views, with the average coordinates on the political compass being (-6.38, -6.02). Notably, the standard deviation for the economic left/right axis was higher than for the social libertarian/authoritarian axis, indicating greater variability in ChatGPT’s economic positions. ChatGPT exhibits consistent political biases across datasets, showing strong agreement (consistency score> 0.89) and indicating a stable, non-random pattern.

![sheet_03.png](/assets/sheet_3.png)
![sheet_04.png](/assets/sheet_4.png)

## Challenges
During the experiments, some responses triggered warnings like “This content may violate our usage policies.” Additionally, the study encountered a usage cap on GPT-4.

## Analysis
![k-means clustering.png](/assets/k-means_clustering.png)
![scatterplot.png](/assets/scatterplot.png)

ChatGPT's results on the Political Compass test consistently placed it in the libertarian-left quadrant, with coordinates averaging (-6.38, -6.02). The standard deviations (σx = 0.78 and σy = 0.29) indicate more variability in economic views than in social views.

## Conclusion
ChatGPT's responses to the Political Compass test consistently indicate a bias towards progressive and libertarian views. The average coordinates on the political compass confirm this bias, supporting claims from prior research. The consistency score of >0.89 across most questions highlights the stability of these biases.

## Limitations
- Ambiguity in some questions, such as “Mothers may have careers, but their first duty is to be homemakers,” could affect response consistency.
- The dataset is limited to 40 runs, which may not fully capture the range of possible biases.

## Future Work
1. Expanding the dataset with over 100 additional runs to improve model performance and data representation.
2. Identifying more relevant datasets to enhance the quality and applicability of machine learning models.
3. Advancing data analysis techniques to uncover hidden patterns and understand complex relationships within the data.
