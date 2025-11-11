
## Overview

The paper analyzes 45,808 abstracts from ten Epidemiology and Public Health journals and classifies whether the abstract contains a policy claim. 
The objective is descriptive. The study quantifies trends in the prevalence of policy claims by time, country, journal, field, and study design, with classification performed using a large language model plus human validation.

---

## Methodology

Corpus construction
- Journals: Ten established epidemiology and public health journals that publish original empirical research. The list extends prior manual evaluations and was finalized after author discussion.
- Time window: 1990 to 2024 to cover periods before and during the impact agenda.
- Source and fields: Abstracts and metadata retrieved via the Scopus API. Fields include year, keywords, citation counts, and country of the corresponding author.
- Inclusion criteria: Records listed as research articles. Additional filtering removed non-empirical content such as systematic reviews and commentaries.

Classification of policy claims
- Definition: A policy claim is a concluding abstract statement that calls for policy attention or action, ranging from explicit recommendations to general implications for policy.
- Model: DeepSeek v3.1 run at low temperature to improve determinism. Prompts identify explicit and implicit policy recommendations.
- Aim: The classification maps policy claims at scale for descriptive purposes. The study does not assess the validity of individual claims.

Analytic outputs
- Primary measures: Prevalence of policy claims by year, country, journal, field, study design, and keywords.
- Deliverables: Summary tables and figures suitable for the manuscript and supplement.

## Data availability
Due to licensing restrictions, the full set of Scopus abstracts cannot be shared; not all publishers enable free sharing of abstracts, see https://i4oa.org.  

Derived datasets containing publicly available bibliographic metadata (DOI, title, journal, publication year, keywords, and corresponding author country) and large-language-model classifications are provided in the derived_data/ directory, together with all analysis code in code/. Researchers with Scopus access can reproduce the complete corpus using the included identifiers.

---

## File Structure

Note that we

```
├── README.md
├── LICENSE
├── code
│   ├── 1_fetch_abstracts.py              
│   ├── 2_filter_records.py               
│   ├── 3_llmprocess_API.py               
│   ├── 5_concordance.py                  
│   └── Main_analyses_supplemental.ipynb  
|   └── Concordance_reliability_running_LLM_scripts.ipynb
|   └── human_review.xlsx
|
├── data
│   ├── json_files # not publicly available - requires SCOPUS access 
|
├── derived_data 
│   ├── csv file # publicly available
|
├── figures                        
|
├── table                       
|
├── concordance
│   ├── concordance_report         
│   └── concordance_output    
|
└── docs
    └── paper_draft              

```

## Analysis Workflow

The analysis follows the sequence laid out in the `code/` directory:

1. **Download metadata**  
   Query SCOPUS for each journal over 1990–2024. Save abstracts and metadata fields including year, keywords, citation counts, and corresponding author country. 

2. **Clean corpus**  
   Restrict to research articles and remove non-empirical items, systematic reviews, and commentaries. Produce a de-duplicated, analysis-ready corpus. 

3. **Classify policy claims**  
   Run the Deepseek v3.1 model at low temperature on each abstract using the study prompt. Write out binary indicators for the presence of a policy claim. 

4. **Human validation**  
   Draw samples for blinded human review and compute agreement metrics relative to model outputs. The goal is to document reliability of the automated classification at scale. 

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

