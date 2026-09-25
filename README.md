# Animal Image Classification using Support Vector Machine

This project was completed as part of the **EGE352 Data Analytic & Machine Learning** module.

The objective was to build a machine learning model that classifies grayscale images into three animal categories:

- Cow
- Dog
- Panda

The project uses a **Support Vector Machine (SVM)** classifier with an RBF kernel. The model was later fine-tuned using **GridSearchCV** to improve classification performance.

## Project Workflow

1. Loaded grayscale animal images from separate training folders.
2. Converted the images into NumPy arrays.
3. Reshaped each image into a 150 × 150 pixel representation.
4. Normalised the image values by scaling pixel values between 0 and 1.
5. Split the dataset into training and testing sets using a 70/30 split.
6. Trained an initial SVM classifier using an RBF kernel.
7. Evaluated the model using a confusion matrix and accuracy score.
8. Used GridSearchCV to tune the SVM `C` and `gamma` parameters.
9. Used the optimised classifier to predict unseen images from the `proj1_test` dataset.
10. Displayed the test images together with their predicted animal classes.

## Model Performance

The initial SVM model achieved an accuracy of approximately **38.1%**.

GridSearchCV was then used to search across different combinations of `C` and `gamma`.

The selected SVM model used:
C = 10
gamma = 0.0001
kernel = RBF

After parameter tuning, the model achieved an accuracy of approximately 83.8% on the test dataset.
