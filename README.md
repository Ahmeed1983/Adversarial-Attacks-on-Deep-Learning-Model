# Adversarial Attacks on DogBreedClassifier

## Introduction
"Adversarial Attacks on DogBreedClassifier" is a comprehensive project focusing on the impact of adversarial attacks on deep learning models. The project employs a DenseNet161-based model trained on the Stanford Dogs Dataset for dog breed classification and investigates the effects of adversarial attacks, specifically the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), on the model's performance.

## Experiment Description

### Purpose of the Project
The primary goal is to explore and demonstrate the influence of adversarial attacks on a pre-trained machine learning model, revealing how slight modifications in input data can lead to misclassifications and expose vulnerabilities in neural networks.

### Why Adversarial Attacks Matter
Adversarial attacks are critical in machine learning and cybersecurity for:
- **Robustness Testing**: They serve as rigorous tests of a model's robustness, revealing vulnerabilities that may not be apparent during standard evaluation.
- **Safety Concerns**: In critical domains such as autonomous driving and medical diagnostics, the reliability of AI systems is paramount.
- **Security Implications**: Recognizing the existence of adversarial examples is key for designing secure AI systems.
- **Ethical Considerations**: They raise important questions about the potential misuse of AI technology.

## Motivations

### Current ML Challenges Addressed
- **Security Concerns**: Highlighting the need for robust AI in critical systems.
- **Real-World Generalization**: Emphasizing the importance of models that perform well under adversarial conditions.
- **Responsible AI Development**: Focusing on ethical aspects and potential harm of AI technology.

### Importance of Model Robustness
- **Trustworthiness**: Ensuring reliable performance in AI systems.
- **Safety-Critical Applications**: Addressing the need for robustness in domains where errors could have severe consequences.
- **Fairness and Bias**: Reducing discriminatory behavior in AI systems through robustness.

## Environment Setup and Installation

### Prerequisites
- Conda environment manager (recommended)
- Python 3.6 or higher
- Pytorch
- Other dependencies listed in 'requirements.txt'

### Installation Steps
1. Clone the repository:
   ```
   git clone https://github.com/Ahmeed1983/Adversarial-Attacks-on-Deep-Learning-Model.git
   ```
2. Navigate to the project directory:
   ```
   cd Adversarial-Attacks-on-Deep-Learning-Model
   ```
3. If using conda, create an environment from the `environment.yml` file:
   ```
   conda env create -f environment.yml
   ```
   Activate the environment:
   ```
   conda activate myenv
   ```
4. Alternatively, install dependencies using pip:
   ```
   pip install -r requirements.txt
   ```

## Data Loading and Preprocessing

Run the `data_loading_and_preprocessing.py` script located in the `data/` directory to download and prepare the dataset.

## Using the Pre-trained Models

Download the model weights and load them as detailed in the "Using the Pre-trained Dog Breed Classifier Models" section.
Downloading the Models
-Standard Model Weights:
        -Dog Breed Classifier Weights - Standard
-Adversarial Training Model Weights:
        -Dog Breed Classifier Weights - Adversarial

## Running the Notebook

The Jupyter notebook included in the `notebooks/` directory contains all the code for model training, evaluation, and adversarial attack analysis. To run the notebook:
1. Ensure Jupyter is installed, which is included in the `requirements.txt`.
2. Navigate to the `notebooks/` directory.
3. Run Jupyter Notebook or JupyterLab:
   ```
   jupyter notebook
   ```
   Or for JupyterLab:
   ```
   jupyter lab
   ```
4. Open `AdversarialAttacksonDeepModel.ipynb`.
5. Execute the cells in sequence to observe the effects of adversarial attacks.

## Conclusion

Through this project, we have contributed to the vital conversation on AI security and robustness. Our investigations into adversarial attacks underscore the need for continued advancements in creating AI systems resilient to such manipulations. The insights from this work aim to inform and inspire the development of more secure, reliable, and ethical AI applications.
