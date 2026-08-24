# Brain Tumor MRI Classification

Brain tumor MRI classification with transfer learning on Xception for four classes: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary**.

## Project Overview

This project trains an image classification model to predict the tumor category from brain MRI scans. The pipeline uses transfer learning with **Xception** as the backbone, then fine-tunes the last 30 layers to adapt the pretrained features to the target MRI dataset.

The final model reports the following test performance on **800 test images**:

- **Accuracy:** 95.63%
- **Precision:** 95.61%
- **Recall:** 95.38%

## Medical Disclaimer

This repository is for research, educational, and experimental purposes only. It is **not** a medical device and must **not** be used as a substitute for professional diagnosis, treatment, or clinical decision-making. Always consult qualified medical personnel for medical interpretation and patient care.

## Dataset

- **Dataset URL:** `https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`


## Preprocessing

Images are preprocessed to match the model input and improve generalization:

- Resize to **299 x 299 x 3**
- Rotation augmentation
- Width shifts
- Height shifts
- Zoom augmentation
- Brightness augmentation
- Horizontal flip

## Model Architecture

The model uses **Xception** pretrained on **ImageNet** as the feature extractor, followed by a compact classification head:

1. **Global Max Pooling**
2. **Dropout (0.30)**
3. **Dense 128, ReLU**
4. **Dropout (0.25)**
5. **Dense 4, Softmax**

## Transfer Learning and Fine-Tuning

Training follows a two-stage approach:

1. Train the custom classification head while keeping the Xception backbone frozen.
2. Fine-tune the **last 30 Xception layers** to improve task-specific performance.

## Results

Evaluated on **800 test images**:

- **Accuracy:** 95.63%
- **Precision:** 95.61%
- **Recall:** 95.38%

### Confusion Matrix

**Class order used for rows and columns:**

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

**Exact confusion matrix:**

[[166  20  13   1]
 [  1 199   0   0]
 [  0   0 200   0]
 [  0   0   0 200]]

## Error Analysis

Observed errors were concentrated in the **Glioma** class:

- **20** Glioma samples misclassified as **Meningioma**
- **13** Glioma samples misclassified as **No Tumor**
- **1** Glioma sample misclassified as **Pituitary**
- **Glioma recall:** approximately **83%**

## Feature Analysis

The representation learned by the network was analyzed using embedding and dimensionality reduction methods:

- **2048 features** from the backbone representation
- **PCA to 50 components** explaining approximately **81% variance**
- **t-SNE** for two-dimensional visualization
- **Confidence analysis** to inspect prediction certainty and ambiguous cases

## Grad-CAM

Grad-CAM visualizations are used to inspect which regions of the MRI images contribute most strongly to predictions. This helps with qualitative analysis and error inspection, but it does not provide clinical interpretability on its own.

## Image References

- `results/confusion_matrix.png`
- `results/training_curves.png`
- `results/pca.png`
- `results/tsne.png`


## Project Structure

```text
.
├── README.md
├── results/
│   ├── Confusion_matrix.png
│   ├── Accuracy_Loss Curves.png
|   ├── Glioma Errors in feature Space.png
│   └── t-SNE of Xception Feature Space.png
├── src/
├── data/
├── notebooks/
└── models/
```

## Installation

```bash
git clone https://github.com/mehrmandi/Brain_tumor_xception.git 
cd Brain_tumor_xception 
pip install -r requirements.tx
```

## Usage

Train the model
```bash
python train.py
```
Evaluate the model
```bash
python evaluate.py
```
Visualize results
```bash
python visualize.py
```

## Requirements

```text
tensorflow
keras
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
```

## Limitations

- The model is intended for classification research, not clinical use.
- Performance depends on dataset quality, class balance, and preprocessing consistency.
- Grad-CAM and feature visualizations are helpful for inspection, but they do not replace expert medical review.
- The reported metrics should be interpreted in the context of the specific dataset split used for evaluation.

## Future Work

- Validate on external datasets from different acquisition sources.
- Compare Xception with additional pretrained backbones.
- Improve interpretability with more robust saliency and uncertainty methods.
- Evaluate calibration and confidence thresholds for safer downstream use.
- Expand class-wise analysis and experiment tracking.

## Technologies

- Python
- TensorFlow / Keras
- Xception
- Scikit-learn
- Matplotlib
- Seaborn
- OpenCV

## Author

- **Name:** `Mehrnoosh Mandipour`
- **Email:** `mehrnooshmandipour@gmail.com`
- **Affiliation:** `Independent Researcher, Iran`

## License

The source code in this repository is released under the MIT License.

The Brain Tumor MRI Dataset used in this project is not included in this repository and is subject to its original license (CC BY 4.0). Please refer to the original dataset source for terms of use and attribution.