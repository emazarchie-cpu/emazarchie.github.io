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


# Case Study 3: Blue Nike Hat Predictor AI

**Project Type:** Personal Machine Learning Project  
**Role:** AI / Machine Learning Project Developer  
**Tools:** Python, Google Colab, Pandas, Scikit-learn  
**Model:** Logistic Regression  
**Date:** October 2026

## 1. Problem

I wanted to explore how machine learning could be used to predict a person's behaviour using structured data.

For this project, I created a simple prediction system to estimate whether I would wear a blue Nike hat under different circumstances. I considered factors such as weather, hat availability, activity, outfit compatibility, mood, and Nike preference.

The main challenge was to transform these observations into a machine-learning problem, train a model to identify patterns, and evaluate whether it could make predictions on previously unseen data.

## 2. Work Done

I developed a supervised machine learning workflow in Google Colab using Python.

**Data collection:** I created a dataset containing 20 observations. Each observation included contextual features and a binary target variable indicating whether the hat was worn (`1`) or not worn (`0`).

**Data preparation:** I loaded the dataset into Pandas and examined its structure, missing values, and target distribution. The dataset contained 10 positive outcomes and 10 negative outcomes.

**Model development:** I implemented a Logistic Regression classifier using Scikit-learn. I prepared the categorical and numerical features for training and separated the dataset into training and testing subsets.

**Model evaluation:** I trained the model on 15 observations and evaluated it on five held out observations. I used test accuracy, a classification report, and a confusion matrix to assess its predictions.

**Prediction:** I supplied a new scenario to the trained model, including weather, activity, hat availability, outfit compatibility, and mood, to obtain a probability estimate for wearing the hat.

Throughout the process, I used the project to practise the complete machine-learning workflow, from collecting data and preparing features to training, evaluation, and prediction.

## 3. Results and Outcomes

The initial model produced the following results:

| Evaluation Metric | Result |
|---|---:|
| Total observations | 20 |
| Training observations | 15 |
| Testing observations | 5 |
| Test accuracy | 80% |
| Correct test predictions | 4 out of 5 |
| Predicted probability of wearing the hat | 77.2% |

For the test scenario, the model estimated a 77.2% probability of wearing the blue Nike hat and classified the outcome as **likely to wear the hat**.

The initial evaluation demonstrated that I could build and execute a complete supervised-learning pipeline using a real machine-learning library.

However, the dataset was very small, and the test set contained only five observations. Therefore, the 80% accuracy is an early experimental result rather than evidence of reliable predictive performance. The 77.2% probability is also a model estimate, not a confirmed real-world probability.

## 4. Key Learnings

This project helped me develop practical experience in:

- Framing a real  world question as a supervised binary classification problem.
- Collecting and structuring data for machine learning.
- Working with Pandas and Scikit-learn in Google Colab.
- Separating training data from testing data.
- Interpreting accuracy, precision, recall, and a confusion matrix.
- Generating probability based predictions.
- Recognising the limitations of small datasets and avoiding unsupported conclusions.

An important lesson was that successfully training a model does not automatically make its predictions reliable. Data quality, feature relevance, sample size, and appropriate evaluation are essential to building useful machine-learning systems.

## 5. Next Steps

The next phase will focus on improving the experiment by:

1. Validating the observations and correcting inconsistent or missing data.
2. Collecting more observations from distinct, genuine situations.
3. Removing features that provide no variation or useful predictive information.
4. Comparing Logistic Regression with a Decision Tree classifier.
5. Evaluating performance more robustly as the dataset grows.
6. Investigating whether the model's predicted probabilities are well calibrated.

## Conclusion

The Blue Nike Hat Predictor AI was my first end to end personal machine learning experiment. It gave me practical experience turning a simple behavioural question into a structured dataset, training a classifier, evaluating its predictions, and identifying opportunities for improvement.

The project represents an early step in my development as a machine learning practitioner, with an emphasis on understanding the process rather than simply producing code that runs.

# Case Study 4 GymPulse AI: Predicting Gym Member Cancellation Risk with Machine Learning

**Project Type:** Applied Machine Learning | Predictive Analytics | AI Product Development  
**Role:** AI/ML Developer & Product Builder  
**Technologies:** Python, Pandas, Scikit-learn, Streamlit, Supabase, GitHub  
**Model:** Logistic Regression  
**Application:** [GymPulse AI](https://gympulse-ai.streamlit.app/)  
**GitHub:** Private repository

## 1. Project Overview

GymPulse AI is a machine-learning-powered gym membership retention application designed to help gym owners identify members who may be at risk of cancelling their memberships.

Gym businesses depend on recurring membership revenue. However, when members gradually reduce their attendance or approach their renewal dates without engaging consistently, gym owners may not recognize the warning signs until a membership is cancelled.

I developed GymPulse AI to explore how machine learning could turn membership and attendance data into actionable retention insights.

The application allows gym operators to assess individual members, evaluate groups of members, and use predicted cancellation risk to prioritize retention efforts.

Rather than building a model purely to demonstrate technical capability, I approached the project as a business problem: **How can a gym use its available data to identify at-risk members early enough to take action?**

## 2. The Business Problem

Member cancellations can reduce recurring revenue and make business growth more difficult.

Gym owners may have access to membership records, attendance histories, renewal dates, and personal training records, but these data points do not automatically tell them which members need attention.

The project focused on three practical challenges:

- **Risk identification:** Recognizing members whose engagement patterns may indicate an increased cancellation risk.
- **Prioritization:** Helping gym operators decide which members should receive attention first.
- **Actionable insights:** Presenting model predictions in a dashboard that supports practical business decisions.

The objective was to develop a predictive decision-support tool rather than simply produce a cancellation prediction.

## 3. My Approach

I approached the project through five stages.

### Stage 1: Data Preparation

I developed a synthetic dataset representing gym membership and engagement behaviour.

The dataset contained 1,000 simulated member records, with 649 non-cancellation cases and 351 cancellation cases.

The target distribution was therefore:

- Non-cancellation: 64.9%
- Cancellation: 35.1%

The dataset was designed to support experimentation with membership characteristics, attendance behaviour, renewal timing, and personal training engagement.

Because the dataset was synthetic, these figures represent the simulated environment rather than observed cancellation rates from a real gym.

### Stage 2: Feature Selection

I selected five input features for the predictive model:

| Feature | Business relevance |
|---|---|
| Membership type | Represents differences between annual, monthly, and quarterly memberships. |
| Days since last visit | Measures how long a member has been away from the gym. |
| Attendance change | Captures changes in member engagement. |
| Days until renewal | Represents the timing of an upcoming membership renewal. |
| Personal training sessions attended | Provides an additional indicator of engagement with gym services. |

The purpose of feature selection was to focus on information that could plausibly help identify changing member behaviour.

These features were selected as predictive signals, not as proof that any particular behaviour causes cancellation.

### Stage 3: Model Development

I implemented a Logistic Regression classifier using Scikit-learn.

I chose Logistic Regression as an interpretable baseline model for a binary classification problem.

The model estimates the likelihood of cancellation based on the selected input features. Its predictions can then be used to categorize members according to their estimated risk.

This approach provided a foundation for evaluating the predictive value of the available features before exploring more complex algorithms.

### Stage 4: Model Evaluation

I evaluated the model using several classification metrics rather than relying on accuracy alone.

The initial evaluation produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 65.0% |
| Precision | 50.0% |
| Recall | 27.1% |
| F1-score | 35.2% |
| ROC-AUC | 0.703 |

These results revealed an important limitation.

Although the model achieved 65% accuracy, its recall was only 27.1%. At the default classification threshold, it identified a relatively small proportion of actual cancellation cases.

For a retention-focused application, this matters because missing an at-risk member may represent a missed opportunity for intervention.

I therefore investigated the effect of lowering the classification threshold.

### Stage 5: Threshold Optimization

Instead of accepting the default threshold, I tested a threshold of 0.30 to investigate the trade-off between identifying more at-risk members and generating additional false positives.

The results on the 200-record test set were:

| Metric | Default threshold | Threshold of 0.30 |
|---|---:|---:|
| Accuracy | 65.0% | 59.0% |
| Precision | 50.0% | 44.9% |
| Recall | 27.1% | 75.7% |
| F1-score | 35.2% | 56.4% |
| ROC-AUC | 0.703 | 0.651 |

**Important evaluation note:** ROC-AUC is normally a threshold-independent ranking metric. Changing only the classification threshold should not change ROC-AUC when the underlying model scores and evaluation dataset remain identical. The reported ROC-AUC values therefore require verification against the actual evaluation code before these results are published as a definitive comparison.

The threshold comparison illustrates the decision trade-off I investigated: increasing recall can help a retention team identify more potential cancellations, but it also increases the number of members flagged for review.

At the 0.30 threshold, the model flagged 118 of the 200 test-set members for assessment.

That is a substantial workload for a gym owner, so the threshold must be evaluated against the business's available time, intervention costs, and tolerance for missed cancellations.

The goal is not to flag everyone. It is to identify a useful group of members whose risk warrants further attention.

## 4. Turning the Model into a Business Application

A predictive model is more useful when its outputs can be incorporated into an actual workflow.

I built GymPulse AI as a Streamlit application to make cancellation-risk assessment accessible through a user interface.

The application includes the following areas:

**Dashboard**

Provides an overview of membership retention risk and helps operators understand where attention may be needed.

**Member Management**

Supports the management of member information used by the application.

**Member Assessment**

Allows an operator to assess an individual member's cancellation risk.

**Batch Assessment**

Supports assessment of multiple member records, making it possible to evaluate groups rather than working through every member individually.

**Model Insights**

Presents information about the model and its predictive performance.

**About the Model**

Documents the underlying approach, algorithm, and selected features.
Application Screenshots

1. GymPulse AI Dashboard

The dashboard provides an overview of the application and its member-retention workflow.



2. Member Assessment

The Member Assessment interface allows gym operators to enter membership and engagement information before generating an individual cancellation-risk assessment.



3. Model Insights

The Model Insights interface lets users explore how the model's estimated outputs respond to changes in member-related inputs.



4. Dashboard — Additional View

An additional view of the GymPulse AI interface.



The application is available at:

https://gympulse-ai.streamlit.app/

## 5. Database Design and Data Security

Because the application is intended for use by different gym businesses, I also worked on separating each gym's data.

I integrated Supabase for authentication and database functionality.

The database includes three principal tables:

- **Gyms:** Stores gym business records and ownership information.
- **Members:** Stores member records associated with individual gyms.
- **Profiles:** Stores user profile and account information.

I configured Row Level Security (RLS) policies to restrict access according to authenticated ownership relationships.

I also reviewed database privileges and tested the application with two separate user accounts to check that one account could not see the other account's member records through the tested application workflow.

This work reinforced an important product-development lesson: building an AI application requires attention to data access, authentication, and the boundaries between customers—not just model performance.

The testing performed provides evidence of account separation in the tested scenarios, but it should not be interpreted as a comprehensive security audit.

## 6. Key Results

The project produced several practical outcomes.

**A working AI application**

I developed a deployed Streamlit application that connects predictive modelling with a usable interface.

**A measurable machine-learning baseline**

The initial Logistic Regression model achieved an ROC-AUC of 0.703, providing a starting point for evaluating whether the selected features contained useful predictive information.

**A more recall-oriented classification strategy**

Testing a lower classification threshold increased reported recall from 27.1% to 75.7%, while reducing reported precision from 50.0% to 44.9%.

The ROC-AUC discrepancy identified above must be resolved before treating the threshold comparison as a fully validated evaluation.

**A business-oriented workflow**

The application supports individual and batch member assessment, helping translate model outputs into a potential retention workflow.

**A multi-user application architecture**

I implemented authentication, relational database relationships, and Row Level Security policies, then tested separation between two accounts.

## 7. Challenges and Lessons Learned

### Accuracy is not the whole story

One of the most important lessons was that a model's accuracy does not necessarily reflect its usefulness for a specific business problem.

In cancellation prediction, recall and precision influence how effectively a gym can identify potential cancellations and how many members require additional review.

The correct balance depends on the business cost of missed cancellations compared with the cost of unnecessary interventions.

### Model predictions need business context

A member flagged as high risk is not guaranteed to cancel.

Predictions should be treated as signals for further assessment rather than definitive conclusions about a person's future behaviour.

Gym owners should combine these signals with relevant context and appropriate human judgment.

### Synthetic data limits real-world conclusions

The project uses synthetic data. Consequently, its evaluation metrics demonstrate the model's behaviour on the simulated dataset, not its expected performance on actual gym customers.

Before using the model to support real retention decisions, I would need to evaluate it on representative historical gym data, with appropriate consent and privacy safeguards.

### Security is part of building AI products

Separating customer data is essential for a multi-tenant application.

This project gave me practical experience with authentication, relational data modelling, Row Level Security, and account-isolation testing.

## 8. What I Would Improve Next

My next development priorities would be:

1. **Validate the evaluation pipeline:** Reproduce all reported metrics from the same held-out test set and investigate the ROC-AUC discrepancy.
2. **Improve data realism:** Evaluate the model on representative historical gym data and investigate potential data leakage.
3. **Compare algorithms:** Benchmark Logistic Regression against alternatives such as Random Forest and gradient-boosted trees.
4. **Calibrate predicted probabilities:** Assess whether predicted cancellation probabilities correspond to observed cancellation frequencies.
5. **Optimize for business value:** Select thresholds using the cost of missed cancellations, intervention costs, and available staff capacity.
6. **Track performance over time:** Monitor model drift and evaluate whether prediction quality changes as member behaviour and gym operations evolve.
7. **Measure retention outcomes:** Run a controlled evaluation to determine whether acting on predictions actually improves membership retention.

The ultimate goal would be to move from predicting cancellation risk to demonstrating measurable improvements in retention and recurring revenue.

## 9. What This Project Demonstrates

GymPulse AI demonstrates my ability to work across multiple stages of applied machine learning:

- Problem definition and business analysis.
- Synthetic data generation and preparation.
- Feature selection and binary classification.
- Logistic Regression modelling with Scikit-learn.
- Evaluation using precision, recall, F1-score, accuracy, and ROC-AUC.
- Classification-threshold analysis.
- Streamlit application development.
- Supabase authentication and relational database design.
- Row Level Security and account-isolation testing.
- Deployment, debugging, and iterative product development.

More importantly, the project demonstrates my approach to AI engineering: start with a real problem, establish a measurable baseline, investigate the limitations, and build a system that could eventually support a practical business decision.

## 10. Conclusion

GymPulse AI began as an investigation into how machine learning could help gym businesses recognize potential membership cancellations.

It developed into a deployed application combining predictive analytics, member assessment workflows, authentication, and data-access controls.

The project also highlighted the difference between building a working model and establishing that a model delivers reliable business value.

My next step is to strengthen the evaluation methodology, validate the model on real-world data, and measure whether prediction-driven interventions improve retention outcomes.

**My objective is to build AI systems that do more than generate predictions: they should help people make better-informed decisions.**

###  ProjectS Repository

[View my FlyRank ML Internship repository](https://github.com/emazarchie-cpu/Emmanuel-Archibong-ML-Intern)

[View my Personal ML repository](https://github.com/emazarchie-cpu/emazarchie.github.io)

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

- **GitHub:** [Explore my FLyrank -ML repositories](https://github.com/emazarchie-cpu/Emmanuel-Archibong-ML-Intern)
- [Explore my Personal -ML repositories](https://github.com/emazarchie-cpu/emazarchie.github.io)

- **LinkedIn:** www.linkedin.com/in/emmanuel-archibong-342120212
- **Email:** emazarchie@gmail.com

I'm open to learning opportunities, collaboration, and conversations about machine learning, artificial intelligence, and responsible AI.
