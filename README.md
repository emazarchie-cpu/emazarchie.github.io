I'm Emmanuel Archibong, an aspiring machine learning professional currently developing my skills through practical assignments and AI assisted learning.

As an AI intern at FlyRank, I'm learning to frame machine learning problems, work with notebooks, navigate GitHub, and understand the reasoning behind technical decisions. I'm interested in more than getting code to run. I want to understand how machine learning works, how to evaluate its outputs, and how to use AI responsibly.

My approach is simple: stay curious, question what I don't understand, take action, and keep improving.                                                                   

 **Featured Case Studies**

##  Case Study 1: Building My Machine Learning Foundations at FlyRank

**Project:** Machine Learning Pipeline on Anonymized Search Data  
**Role:** Machine Learning Intern  
**Tools:** Python, Jupyter Notebook, Google Colab, GitHub, AI-assisted learning

### 1. The Problem

My FlyRank internship gave me an opportunity to move beyond learning AI concepts and begin working with a practical machine learning workflow.

The assignment challenged me to work with anonymized search data, develop a simple rule for ranking pages, train a decision tree, and compare the two approaches. I also needed to understand why data leakage occurs and why information about the outcome must not improperly influence the model's predictions.

As someone developing my machine learning skills, I wanted to understand not only how to run the code but also the reasoning behind the process.

### 2. What I Did

I worked with the `w02_ml_task_framing.ipynb` notebook in my FlyRank internship repository.

The assignment involved:

- Working with real, anonymized search data.
- Creating a simple hand written rule to rank pages.
- Training a depth 2 decision tree and printing its learned decision rules.
- Comparing the model's performance against the hand-written baseline.
- Examining where the learned model performed better and where it did not.
- Learning about data leakage and why outcome information must be kept out of model inputs.

I used AI as a learning assistant to help me understand technical concepts and notebook outputs. I also practised navigating GitHub, running notebooks, and interpreting results.

### 3. What Came of It

This assignment helped me understand how a machine learning model can learn decision rules from data and how those rules can be compared against a simpler, manually defined approach.

It also introduced me to an important principle in machine learning: a model's results are only meaningful when the evaluation is designed correctly and information from the outcome does not leak into the inputs.

Beyond the technical concepts, I gained practical experience working with notebooks and learned to pay closer attention to the reasoning behind a machine learning workflow.

The experience reinforced my goal of becoming someone who understands not just how to run a model, but also how to evaluate its results and explain its decisions.
# Case Study 2: Search Intelligence Data Contract
**Role:** Machine Learning Intern  
**Organization:** FlyRank  
**Project:** ML-04 — Search Intelligence Data Contract  
**Tools:** Python, DuckDB, SQL, Pandas, Hugging Face, Google Colab

## 1. The Problem

Businesses with large numbers of web pages need to understand which pages may benefit from editorial review to improve their search performance.

However, using machine learning to support these decisions requires more than collecting data. The data must be reliable, its limitations must be understood, and information from the future must not accidentally influence predictions.

The challenge was to establish a clear data contract for a search intelligence task: define what each record represents, distinguish predictive features from future outcomes, validate the underlying data, and document the limitations before model development.

## 2. What I Did

As a Machine Learning Intern at FlyRank, I developed a structured data contract using the company's internship warehouse.

**Data access and exploration**
- Connected to the FlyRank warehouse hosted on Hugging Face.
- Used DuckDB to query remote Parquet datasets and inspect their schemas.
- Examined daily search performance records alongside content and client metadata.

**Defined the analysis framework**
- Established the content item belonging to a client as the unit of analysis.
- Defined February 2026 as the feature window and March 2026 as the outcome window.
- Set a decision cutoff of 28 February 2026 to help prevent future-information leakage.
- Established an initial eligibility threshold of at least 100 Google Search Console impressions during March.

**Prepared and validated the data**
- Aggregated daily performance records into content-level summaries.
- Calculated candidate outcome measures, including click-through rate (CTR) and average search position.
- Checked data availability, missing values, record counts, identifier uniqueness, and coverage across the two periods.
- Matched content items across February and March to establish which items had both historical features and subsequent outcomes.

**Documented data limitations**
- Identified differences in Google Search Console and Google Analytics coverage.
- Distinguished unavailable data from genuine zero activity.
- Documented risks associated with historical metadata, outcome eligibility, and temporal leakage.
- Clarified that observed relationships do not establish causation.

## 3. What Came of It

The work produced a documented data contract and a set of validation checks to support future machine learning development.

Key findings from the warehouse analysis included:

| Validation measure | Result |
|---|---:|
| February content items aggregated | 321,546 |
| March content items observed | 331,437 |
| March content items meeting the initial eligibility criteria | 101,441 |
| Content items matched across February and March | 93,671 |
| Clients represented in the matched dataset | 38 |
| Duplicate content IDs in the matched dataset | 0 |
| Matched rows below the 100-impression threshold | 0 |
| February items with available GA4 data | 33,422 |

The validation also showed that source coverage was uneven. Many content items had no available GA4 observations during February, demonstrating why unavailable measurements must not automatically be interpreted as zero activity.

The resulting contract established a clearer foundation for evaluating potential relationships between historical page characteristics and subsequent search performance.

**These results describe data coverage and validation, not predictive model performance.** No model accuracy or improvement in search rankings was claimed as an outcome of this assignment.

## 4. Technical Approach

The analysis used Python to orchestrate the workflow, DuckDB and SQL to query and aggregate remote Parquet data, and Pandas to inspect and summarize the resulting datasets.

The workflow followed four stages:

1. **Define:** Establish the unit of analysis, feature window, outcome window, and eligibility criteria.
2. **Classify:** Separate candidate features, labels, contextual fields, and excluded variables.
3. **Validate:** Query record counts, data availability, missing values, uniqueness, and cross-period overlap.
4. **Document:** Record data limitations and safeguards against temporal leakage and unsupported conclusions.

## 5. Key Lessons

This assignment strengthened my understanding of the work required before machine learning model development.

I learned that:
- A well-defined prediction task begins with a clear unit of analysis and decision point.
- Data availability and data quality can significantly affect which observations are suitable for analysis.
- Feature engineering must respect the information available at the time a decision would be made.
- Validation queries provide evidence for data-contract claims instead of relying on assumptions.
- Responsible machine learning requires communicating uncertainty and limitations, not just producing numbers.

## 6. Conclusion

This project focused on making search performance data more structured, interpretable, and suitable for future machine learning work.

My contribution was to define the analysis rules, query and aggregate performance data, validate the eligible population, and document the limitations that future modelling work must address.

The experience reinforced an important principle in machine learning: **before building a model, make sure you understand what the data represents, what it can support, and what it cannot tell you.**

---

**Disclaimer:** This case study describes my work on the ML-04 internship assignment using the FlyRank internship warehouse. The reported figures are descriptive validation results and do not represent a deployed system or demonstrated improvement in search performance.
###  Project Repository

[View my FlyRank ML Internship repository](https://github.com/emazarchie-cpu/Emmanuel-Archibong-ML-Intern)

## Skills I'm Developing

- **Machine Learning:** ML problem framing, decision trees, model evaluation, and data leakage prevention.
- **Python & Data Analysis:** Python, Pandas, SQL, and DuckDB for data exploration, aggregation, and validation.
- **Data Engineering Fundamentals:** Data contracts, schema inspection, data quality checks, and feature preparation.
- **AI Assisted Problem Solving:** Using AI tools to understand technical concepts, investigate errors, and improve my workflow while critically evaluating their outputs.
- **Data Driven Decision Making:** Interpreting results, questioning assumptions, identifying data limitations, and communicating findings responsibly.
- **Developer Tools:** Google Colab, Jupyter Notebooks, Git, GitHub, and Hugging Face Datasets.
- **Responsible AI:** Understanding data leakage, recognizing limitations, and distinguishing observed relationships from causal claims.
- **Technical Communication:** Documenting projects, explaining technical decisions, and presenting findings clearly.

I'm building these skills through hands on projects, practical assignments, independent study, and continuous experimentation.

##  My Learning Philosophy

I don't want to rely on AI without understanding what it produces. My goal is to strengthen my technical skills, question assumptions, evaluate results, and learn to explain the reasoning behind my decisions.

##  Connect With Me

- **GitHub:** [Explore my repositories](https://github.com/emazarchie-cpu/Emmanuel-Archibong-ML-Intern)
- **LinkedIn:** www.linkedin.com/in/emmanuel-archibong-342120212
- **Email:** emazarchie@gmail.com

I'm open to learning opportunities, collaboration, and conversations about machine learning, artificial intelligence, and responsible AI.
