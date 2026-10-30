# Development and Evaluation of an Interpretable TF-IDF and Logistic Regression Multi-Class Email Threat Classification and Decision-Support System Using Heterogeneous Public Datasets for Application in Nigerian Oil and Gas Operations

A configurable, multi-stage machine-learning pipeline for email acquisition, preprocessing, deduplication, dataset partitioning, model training, calibration, evaluation and diagnostic analysis.

## 1. Project Overview

This project implements a reproducible machine-learning pipeline for detecting and classifying email messages into three categories:

| Class | Label | Description |
|---|---:|---|
| Legitimate | 0 | Genuine, non-malicious email messages |
| Spam | 1 | Unsolicited or unwanted email messages |
| Phishing | 2 | Fraudulent messages intended to deceive recipients or obtain sensitive information |

The system is designed to support systematic experimentation, dataset quality assessment, model development and performance evaluation.

It incorporates multiple processing and validation stages to improve data quality, reduce duplication, maintain dataset integrity and support reliable model evaluation.

The pipeline uses configurable YAML files, allowing different experimental revisions to be executed with distinct parameters without modifying the underlying source code.

## 2. Key Features

- **Multi-class classification:** Distinguishes legitimate, spam and phishing email messages.
- **Dataset acquisition:** Supports the acquisition and processing of email datasets.
- **Email preprocessing:** Extracts and prepares email records for machine-learning operations.
- **Deduplication:** Identifies duplicate records and maintains duplicate references.
- **Dataset partitioning:** Organises records into appropriate development and evaluation datasets.
- **Machine-learning training:** Supports the training of multiple classification algorithms.
- **Model calibration:** Provides a dedicated calibration stage for model confidence and probability estimates.
- **Model evaluation:** Assesses classification performance using evaluation procedures.
- **Diagnostic analysis:** Examines pipeline outputs and potential performance issues.
- **Error review:** Supports the investigation of incorrectly classified messages.
- **Configurable experiments:** Uses YAML configuration files to manage different experimental revisions.
- **Reproducibility:** Uses configurable random seeds and organised experiment outputs.

## 3. Project Architecture

The pipeline follows a modular processing architecture.

```text
Email Datasets
      |
      v
Dataset Acquisition
      |
      v
Manifest Construction
      |
      v
Email Preprocessing
      |
      v
Deduplication
      |
      v
Dataset Splitting
      |
      v
Model Training
      |
      v
Model Audit
      |
      v
Model Calibration
      |
      v
Model Evaluation
      |
      v
Diagnostics
      |
      v
Error Review
      |
      v
Performance Analysis
```

Each stage performs a specific operation and produces outputs required by subsequent stages.

The modular design allows individual stages to be inspected, executed and improved without restructuring the entire pipeline.

## 4. Dataset Sources

The project incorporates publicly available email datasets for developing and evaluating email classification models.

### Enron Email Dataset

The Enron email dataset provides legitimate email messages that can support the legitimate-email classification category.

### Nazario Phishing Corpus

The Nazario phishing corpus provides phishing email samples for developing and evaluating phishing detection capabilities.

The project configuration identifies the following Nazario mailbox files:

- phishing0.mbox
- phishing1.mbox
- phishing2.mbox
- phishing3.mbox
- 20051114.mbox

The private-phishing4.mbox file is excluded in the current configuration.

**Dataset note:** The original datasets are maintained separately from the source-code repository. Users must obtain the datasets from their respective authorised sources and configure their locations before running the pipeline.

## 5. Technology Stack

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and processing |
| Scikit-learn | Machine-learning algorithms and evaluation |
| NumPy | Numerical operations |
| YAML | Experiment configuration |
| Joblib | Model and processing artifact persistence |
| Git | Version control |

Additional dependencies should be installed using the project's dependency specification.

## 6. Repository Structure

The repository is organised into source code, configuration files, datasets, model artifacts and documentation.

```text
code_rev2_complete/
│
├── src/
│   ├── acquire.py
│   ├── build_manifest.py
│   ├── external.py
│   ├── dedup.py
│   ├── split.py
│   ├── train.py
│   ├── audit.py
│   ├── calibrate.py
│   ├── evaluate.py
│   ├── diagnose.py
│   └── error_review.py
│
├── configs/
│   └── decision_table.yaml
│
├── data/
│   ├── raw/
│   │   ├── enron_spam/
│   │   └── nazario/
│   │
│   └── processed/
│
├── models/
│
├── results/
│
├── config_rev2.yaml
├── config_rev3.yaml
├── config_rev4.yaml
├── config_rev5.yaml
│
├── run_pipeline.py
├── README.md
├── README_RUN.md
├── .gitignore
└── requirements.txt
```

*Note: This is a logical representation of the project structure. Actual filenames and directory organisation may differ between revisions.*

## 7. Installation and Setup

### Prerequisites

- Python 3.x
- Git
- Windows, Linux or macOS
- Access to the required email datasets

### Clone the Repository

```bash
git clone https://github.com/dewordman/finalproject.git
cd finalproject
```

### Create a Virtual Environment

On Windows:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a requirements.txt file is not yet available, create one containing the project's actual dependencies.

## 8. Configuration

The pipeline uses YAML configuration files to control experimental settings.

Available configuration revisions include:

- config_rev2.yaml
- config_rev3.yaml
- config_rev4.yaml
- config_rev5.yaml

These configurations allow different experimental settings to be maintained independently.

An example configuration structure is:

```yaml
seed: 42

classes:
  Legitimate: 0
  Spam: 1
  Phishing: 2

paths:
  raw_dir: data/raw
  processed_dir: data/processed
  results_dir: results
  models_dir: models
```

The actual configuration files should be consulted for the complete list of supported parameters.

Before execution, ensure that:

1. The required datasets are available.
2. Dataset paths point to the correct locations.
3. The selected configuration file is valid.
4. Required dependencies are installed.
5. Output directories are writable.

## 9. Running the Pipeline

The main execution script is:

```text
run_pipeline.py
```

To execute the default configuration, use:

```powershell
python run_pipeline.py
```

If the script supports an explicit configuration argument, select a revision using its documented command-line option.

For example, the intended configuration selection may look like:

```powershell
python run_pipeline.py --config config_rev5.yaml
```

Confirm the supported command-line arguments in the script before using this example.

## 10. Experimental Revisions

The project maintains multiple configuration revisions to support iterative development and experimentation.

| Revision | Configuration |
|---|---|
| Revision 2 | config_rev2.yaml |
| Revision 3 | config_rev3.yaml |
| Revision 4 | config_rev4.yaml |
| Revision 5 | config_rev5.yaml |

Each revision should be evaluated using its corresponding configuration and generated outputs.

For meaningful comparisons, experimental conditions, dataset partitions and random seeds should be documented.

## 11. Model Evaluation

The evaluation stage is intended to assess the effectiveness of the trained classifiers.

Relevant classification metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class performance
- Probability calibration

Particular attention should be given to phishing detection, where false negatives may allow malicious messages to be classified as legitimate.

Evaluation should therefore consider individual class performance rather than relying exclusively on overall accuracy.

Actual performance figures should be reported from the generated evaluation results.

## 12. Output and Generated Artifacts

The pipeline generates intermediate processing files, trained models and experiment results.

Examples include:

| Artifact | Purpose |
|---|---|
| records_parsed.pkl | Parsed email records |
| records_grouped.pkl | Grouped email records |
| records_partitioned.pkl | Partitioned dataset |
| records_development.pkl | Development dataset |
| duplicate_reference.joblib | Duplicate-reference information |
| svm.joblib | Trained SVM model |
| logreg.joblib | Trained logistic regression model |
| nb.joblib | Trained Naive Bayes model |

These artifacts are generated during execution and are not included in the GitHub source-code repository.

They can be recreated by executing the appropriate pipeline stages with the required datasets and configuration files.

## 13. Version Control and Data Management

To maintain a manageable repository, large datasets, intermediate processing files and generated machine-learning models are excluded from Git version control.

The repository primarily contains:

- Source code
- Configuration files
- Documentation
- Dependency specifications
- Supporting project resources

This separation reduces repository size while preserving the reproducibility of the experimental workflow.

## 14. Limitations and Considerations

The effectiveness of the classification system depends on dataset quality, class distribution, preprocessing decisions and the representativeness of the training data.

Potential limitations include:

- Dataset imbalance.
- Duplicate or near-duplicate email messages.
- Differences between historical datasets and contemporary phishing techniques.
- Misclassification of legitimate messages as malicious.
- Previously unseen phishing patterns.
- Model generalisation to external datasets.

Results should therefore be interpreted in the context of the datasets and experimental conditions used.

## 15. Future Improvements

Potential areas for further development include:

- Incorporating additional contemporary phishing datasets.
- Improving feature extraction and email representation.
- Exploring advanced ensemble-learning approaches.
- Evaluating transformer-based email classification models.
- Improving detection of previously unseen phishing patterns.
- Conducting cross-dataset validation.
- Improving model interpretability and error analysis.
- Developing a real-time email classification interface.

## 16. Research and Academic Use

This project is intended to support research and experimentation in machine-learning-based email spam and phishing detection.

Researchers using the project should document their datasets, configuration revisions, preprocessing procedures, experimental conditions and evaluation results to support reproducibility.

Dataset owners and original researchers should be appropriately acknowledged when their datasets are used.

## 17. Author and Repository

**Repository:** https://github.com/dewordman/finalproject

**Project:** Development and Evaluation of an Interpretable TF-IDF and Logistic Regression Multi-Class Email Threat Classification and Decision-Support System Using Heterogeneous Public Datasets for Application in Nigerian Oil and Gas Operations

**Author:** Babatunde Ajibola

---

*This repository is a research and development project. It is not intended to replace dedicated email security systems or enterprise anti-phishing solutions.*
