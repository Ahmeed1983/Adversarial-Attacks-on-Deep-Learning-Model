# Adversarial Attacks on DogBreedClassifier

**Table of Contents**

1. **Introduction**
    - Overview of the Project
    - Objectives and Goals

2. **Experiment Description**
    - Purpose of the Project
    - Importance of Adversarial Attacks

3. **Motivations**
    - Addressing Current ML Challenges
    - The Need for Model Robustness

4. **Environment Setup and Installation**
    - Prerequisites
    - Installation Steps

5. **Data Resources**
    - Full Stanford Dogs Dataset
    - Dataset Subsets
        - Original Subset (10 Images per Class)
        - Enhanced Subset (60 Images per Class)

6. **Data Loading and Preprocessing**
    - Instructions for Data Preparation

7. **Using the Pre-trained Models**
    - Standard Model Weights
    - Adversarially Trained Model Weights

8. **Running the Notebook**
    - Steps to Execute the Notebook

9. **Conclusion**
    - Summary and Final Thoughts

10. **FAQ**
    - Frequently Asked Questions

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
## Data Resources

The Stanford Dogs Dataset is accessible in its complete form and in two subsets to suit various computational requirements and experimental scopes.

### Full Stanford Dogs Dataset
For exhaustive analysis and training, the full dataset is available:

- [Download Full Stanford Dogs Dataset](https://mega.nz/file/8GclxSDb#wrQiLQSvp-iJJuqllMsQnfFq-on-OO0k5IzyzNOyaY4)

### Dataset Subsets
Choose between two subsets of the dataset based on your needs:

#### Original Subset (10 Images per Class)
Optimized for rapid testing and low-resource scenarios, this subset contains 10 images per class:

- [Download Original Subset (10 Images/Class)](https://mega.nz/file/xDU2gD5I#83LpT9iUjzUlVy_-6ufHP3wSVHJm_HDsHcCWUBdbdhQ)

#### Enhanced Subset (60 Images per Class)
For broader experimentation without utilizing the full dataset, this larger subset provides 60 images per class:

- [Download Enhanced Subset (60 Images/Class)](https://mega.nz/file/MfllXAhS#aEXSMA868HcAlgbsqC9hjeRGhr237113tDAA4YDU_0k)


## Data Loading and Preprocessing

Run the `data_loading_and_preprocessing.py` script located in the `data/` directory to download and prepare the dataset.

## Using the Pre-trained Models

Download the model weights and load them as detailed in the "Using the Pre-trained Dog Breed Classifier Models" section.
## Downloading the Models

### Standard Model Weights:
- [Dog Breed Classifier Weights - Standard](https://mega.nz/file/hf8xAYjQ#VhB_DLv1dWXa5o9KqKuB3y0iHbF3mlf9wMCKVeWUMBA)

### Adversarial Training Model Weights:
- [Dog Breed Classifier Weights - Adversarial](https://mega.nz/file/QSlkWZiJ#zV5epBI12TtsJ1LnuhVSQYLyE8ArYYRSO7Y-fKUnDts)


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
4. Open `AdversarialAttacksonDeepModel.ipynb` for the full Stanford Dog Image Dataset. OR
5. Open `AdversarialAttacksonDeepModelSubset.ipynb` for the Subset Stanford Dog Image Dataset.
6. Execute the cells in sequence to observe the effects of adversarial attacks.


## Conclusion

Through this project, I have contributed to the vital conversation on AI security and robustness. Our investigations into adversarial attacks underscore the need for continued advancements in creating AI systems resilient to such manipulations. The insights from this work aim to inform and inspire the development of more secure, reliable, and ethical AI applications.


## FAQ

### Q1: What is the purpose of this project?
**A1:** The project aims to demonstrate the impact of adversarial attacks, such as the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), on a pre-trained DenseNet161 model used for dog breed classification. It highlights how slight modifications in input data can lead to misclassifications and expose vulnerabilities in neural networks.

### Q2: Do I need to download the entire Stanford Dogs Dataset to use this project?
**A2:** No, you have options. While the full dataset is available for comprehensive analysis, we also provide two subsets of the dataset. The original subset contains 10 images per class, and the enhanced subset contains 60 images per class. These subsets are suitable for quicker experimentation and testing.

### Q3: How can I use the pre-trained models provided in the project?
**A3:** We've provided download links for two sets of model weights: standard and adversarially trained. Once downloaded, you can load these weights into the DenseNet161 model structure provided in the project's scripts. Detailed instructions for loading the models are included in the "Using the Pre-trained Models" section of the README.

### Q4: Can I run the project in Google Colab?
**A4:** Yes, the project is compatible with Google Colab. You can upload the Jupyter notebook provided in the `notebooks/` directory to Colab and follow the instructions within to run the experiments. Just make sure to upload the dataset or model weights as necessary or modify the paths in the notebook to point to the right locations.

### Q5: What should I do if I encounter an error while running the project?
**A5:** First, ensure you have followed all installation and setup instructions correctly, including installing all dependencies. If the error persists, please document the error message and steps leading to it, and feel free to raise an issue in the GitHub repository for further assistance.


