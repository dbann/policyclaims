
## Overview

The paper analyzes 45,807 abstracts from ten Epidemiology and Public Health journals and classifies whether the abstract contains a policy claim. 
The objective is descriptive. The study quantifies trends in the prevalence of policy claims by time, country, journal, field, and study design, with classification performed using a large language model plus human validation.

Preprint: https://www.medrxiv.org/content/10.1101/2025.11.13.25340175v1

---

## Methodology

### Corpus construction

- Journals: Ten established epidemiology and public health journals that publish original empirical research. The journal list extended prior manual evaluations and was finalized after discussion among the authors.
- Time window: 1990 to 2024, spanning periods before and during the rise of the policy impact agenda.
- Source and fields: Abstracts and metadata were retrieved through the Scopus API. Retrieved fields included publication year, keywords, citation counts, and corresponding author country.
- Inclusion criteria: Records classified as research articles. Additional filtering removed non-empirical content such as systematic reviews and commentaries.

### Classification of policy claims

- Definition: A policy claim is a concluding abstract statement that calls for policy attention or action, ranging from explicit recommendations to broader implications for policy.
- Model: DeepSeek V3.1 was run at low temperature to improve determinism. Prompts were designed to identify both explicit and implicit policy recommendations.
- Aim: The classification was used to map policy claims at scale for descriptive purposes. The study does not assess the validity of individual claims.

### Analytic outputs

- Primary measures: Prevalence of policy claims by year, country, journal, keywords/topics, and study design.
- Deliverables: Summary tables and figures for the manuscript and supplementary materials.

### Use of AI coding assistants
Multiple LLM coding assistants were used to support code drafting, refactoring, debugging, and code review/checking. These included Codex (GPT-5.5) and Claude Code (Claude Sonnet 4.6 and Claude Opus 4.7). The authors reviewed and retained responsibility for the analysis code, outputs, and interpretation.

## Data availability
Due to licensing restrictions, the full set of Scopus abstracts cannot be shared; not all publishers enable free sharing of abstracts, see https://i4oa.org.  

The shareable derived dataset is provided in `derived_data/`. It contains publicly available bibliographic metadata (DOI, title, journal, publication year, keywords, and corresponding author country) together with large language model classifications, but excludes full abstracts. The private `data/` directory contains licensed Scopus source files and intermediate analysis files with abstracts, and is not intended for redistribution. Researchers with Scopus access can reproduce the complete corpus using the included identifiers and code.

## Cost and time to process
The cost and time to process such a large number of abstracts are dependent on the LLM compute / API costs; for the Deepseek API, for example, the analysis incurred ~$3 and ~10 hours of processing time. Since Deepseek is open-weight, this or other open-weight models can be run on local hardware with sufficiently high RAM. 

---

## File Structure

```
├── code                  # data processing, LLM classification, validation, and analysis scripts
├── concordance           # repeated LLM run outputs for concordance analyses
├── data                  # private source/intermediate files; not shared because abstracts are licensed
│   ├── analysis          # full analytic datasets with abstracts
│   └── json_files        # Scopus JSON exports and filtered/LLM-labelled abstract files
├── derived_data          # public/shareable derived metadata and LLM classifications
├── figures               # main and supplementary figures
└── table                 # validation files and exported main/supplementary tables
           
```

## Analysis Workflow

The analysis follows the sequence laid out in the `code/` directory:

File provenance: Scopus JSON exports in `data/json_files/` are filtered by `code/2_filter_records.py` into `data/json_files/filtered/all_abstracts.json`; `code/3_run_llm_classification.py` adds policy-claim labels in `data/json_files/filtered/all_abstracts_LLM.csv`; `code/4_build_analysis_dataset.py` merges those labels and writes `data/analysis/analysis_dataset.csv`; `code/5_add_study_design_and_topics.py` adds design/topic variables in `data/analysis/analysis_dataset_enriched_v2.csv`, which is used for the main analyses. Earlier study-design classifier variants are archived in `code/study_design_supplemental/` for provenance and sensitivity checks.

1. **Download metadata**  
   Query Scopus for each journal over 1990-2024 and save abstracts and metadata fields including publication year, keywords, citation counts, and corresponding author country.

2. **Clean corpus**  
  Restrict the dataset to research articles and remove non-empirical items, systematic reviews, and commentaries to produce an analysis-ready corpus.

3. **Classify policy claims**  
   Run DeepSeek V3.1 at low temperature on each abstract using the study prompt and generate a binary indicator for the presence of a policy claim.

4. **Human validation**  
   Draw samples for blinded human review and compute agreement metrics against model outputs to assess reliability of the automated classification.

5. **Primary analyses**  
   Estimate prevalence by year, country, journal, field, and study design. Generate time series, country rankings, and journal contrasts.

6. **Keyword analyses**  
   Describe variation in claim rates across keywords and examine changes over time by topic.  

7. **Reporting**  
   Export figures and tables for the manuscript and supplementary materials.
   
---

# Authors and acknowledgments
David Bann<sup>1</sup>  \
Mengyao Wang<sup>2</sup>


### Author Affiliations:

1. Centre for Longitudinal Studies, University College London, UK
2. Department of Biostatistics, Yale University, US 
