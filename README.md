# Predicting Molecular Properties: logS and logP
## Overview
This repository contains the code and resources for a project submitted to the ChemLLMathon Challenge. The project focuses on predicting molecular properties, specifically logS (aqueous solubility) and logP (lipophilicity), using fine-tuned machine learning models.

## Motivation
Accurate prediction of logS and logP is critical in drug discovery, as these properties influence bioavailability, ADMET profiles, and overall efficacy. Our solution leverages pre-trained large language models and state-of-the-art tools to enhance the accuracy and scalability of molecular property predictions.

## Features
- Data Preprocessing: Normalizes and partitions data based on LogP-LogS thresholds.
- Fine-Tuning ChemBERT: Customizes the ChemBERT model to improve prediction accuracy.
- Molecule Generation: Uses ChemGPT to generate new molecules with desirable properties.
- Predictive Analysis: Outputs accurate predictions for molecular properties using the fine-tuned model.

## Requirements
- Python 3.8 or higher
- Anaconda3
### Libraries:
- Pandas
- Numpy
- RDKit
- PyTorch
- Scikit-learn
- Transformers
- Plotly
- cv2
- SELFIES

## Installation
- Clone the repository:
	```
	git clone https://github.com/yourusername/ChemLLMathon_PredictMolecularProps.git
	cd ChemLLMathon_PredictMolecularProps
	```
- Run **ChemLLMathon_LogP-LogS_Prediction.ipynb** notebook through Jupyter Notebooks.
## Project Workflow
- Data Import and Normalization
- Partitioning based on LogP-LogS values
- Tanimoto Similarity Analysis
- ChemBERT Fine-Tuning
- Molecule Generation with ChemGPT
- Property Prediction and Evaluation

## Evaluation
### Metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

## Results: 
- The basic machine learning techniques with the generated fingerprints outperformed the fine-tuned ChemBERT.

## Limitations
- Limited dataset size may impact generalization.
- Fine-tuning large models can be computationally intensive.

## Future Work
- Incorporating additional datasets for broader applicability.
- Optimizing the fine-tuning process to reduce computational costs.
- Extending the model to predict other molecular properties such as pKa and LogD.

## Acknowledgments
This project was developed as part of the ChemLLMathon Challenge. Special thanks to the organizers and mentors for their guidance and support.