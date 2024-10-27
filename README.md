# Multimodal Diagnostic Framework for Dementia: Integrating Textual and Acoustic Data for Enhanced Classification Accuracy

## Author
**Shreya Murigendra Pattanashetti**  
MSc Artificial Intelligence, FT  
Supervised by: Dr. Matthew Purver

---

## Project Overview
This repository presents a multimodal diagnostic framework for dementia classification, integrating textual and acoustic data to improve diagnostic accuracy. This study specifically focuses on leveraging the Pitt Corpus to compare single-modal (text or audio) and multimodal approaches using deep learning architectures such as BioBERT, Clinical BERT, RoBERTa, and DistilBERT. Through an advanced approach that combines linguistic and acoustic feature extraction, this framework contributes to more accurate dementia diagnostics, focusing on Alzheimer’s Disease (AD), Mild Cognitive Impairment (MCI), and other related conditions.

## Abstract
Dementia diagnostics can be improved with multimodal techniques that incorporate both linguistic and acoustic markers from patient interactions. This project evaluates deep learning models trained on text-only, audio-only, and combined multimodal data for classifying dementia subtypes and stages. The findings underscore the superiority of multimodal approaches, with BioBERT showing enhanced diagnostic accuracy when optimized with specific dropout and regularization strategies. 

---

## Project Structure
- **Introduction**: Provides background information on dementia, highlighting the potential of multimodal data in diagnostic enhancement.
- **Related Work**: Summarizes significant studies in text-based, audio-based, and multimodal dementia diagnosis.
- **Methodology**:
  - **Research Objective**: Outlines the goal of developing multimodal diagnostic models.
  - **Significance**: Discusses the potential impact of integrating text and audio data in dementia diagnostics.
- **Data Preparation**:
  - **Corpus Description**: Details the Pitt Corpus from TalkBank, focusing on interactions with dementia patients.
  - **Data Cleaning & Preprocessing**: Covers steps like noise reduction, tokenization, normalization, and data augmentation.
- **Feature Extraction**: Explains linguistic and acoustic feature extraction using models such as BioBERT, Clinical BERT, and Mel-frequency cepstral coefficients (MFCC) for audio.
- **Model Selection & Architecture**:
  - **Multimodal Architecture**: Integrates both audio and text features.
  - **Text-only & Audio-only Models**: Describes architectures for each single modality.
- **Training the Model**: Details hyperparameter tuning, optimization techniques, and cross-validation strategies.
- **Evaluation**: Presents performance metrics such as accuracy, precision, recall, ROC-AUC, and statistical validation.
- **Results**: Analyzes model performance across dementia types and stages, underscoring the advantages of multimodal approaches.
- **References**: Includes sources cited for related work, methodologies, and data.

---

## Installation & Requirements
This project uses Python with machine learning and audio-processing libraries.  
Ensure you have the following installed:

- Python 3.8 or higher
- Libraries:
  - TensorFlow
  - PyTorch
  - Numpy
  - Pandas
  - Scikit-learn
  - Librosa (for audio feature extraction)
  - Hugging Face Transformers (for BioBERT, Clinical BERT, RoBERTa, and DistilBERT)

## Dataset
- **Pitt Corpus**: Available from [TalkBank](https://dementia.talkbank.org/access/English/Pitt.html). This dataset includes audio recordings and transcriptions, formatted to capture linguistic and acoustic markers indicative of cognitive impairments.

## Usage
1. **Data Preparation**: Follow scripts in `data_preprocessing/` to clean, normalize, and preprocess audio and text data.
2. **Feature Extraction**: Use `feature_extraction/` to derive linguistic and acoustic features.
3. **Model Training**:
   - Run `train_text_model.py` for text-only models.
   - Run `train_audio_model.py` for audio-only models.
   - Run `train_multimodal_model.py` for multimodal models.
4. **Evaluation**: Use `evaluate_model.py` to assess model performance and generate classification reports.

## Results
Multimodal models achieved a notable improvement in classification accuracy, precision, and recall, particularly using BioBERT in combination with optimized dropout and regularization settings. Detailed results, including per-class analyses and comparisons across models, are available in `results/`.

## Acknowledgments
Special thanks to Dr. Matthew Purver for guidance, and to the TalkBank team for access to the Pitt Corpus. This project is part of the MSc Artificial Intelligence program at University of London - Queen Mary.

## Citation
If you use this project in your research, please cite:

```bibtex
@misc{pattanashetti2024multimodal,
  author = {Shreya Murigendra Pattanashetti},
  title = {Multimodal Diagnostic Framework for Dementia: Integrating Textual and Acoustic Data for Enhanced Classification Accuracy},
  year = {2024}
}
