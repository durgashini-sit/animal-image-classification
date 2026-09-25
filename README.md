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
`C = 10, gamma = 0.0001, kernel = RBF`

After parameter tuning, the model achieved an accuracy of approximately **83.8%** on the test dataset.

## Dataset

The repository contains four image folders:
Data/

├── CowHead_gray/

├── DogHead_gray/

├── PandaHead_gray/

└── proj1_test/

The training dataset contains grayscale images of cows, dogs and pandas. The images were provided already pre-processed to grayscale and a fixed resolution for the project.
The proj1_test folder contains unseen images used to test the final trained classifier.

## Technologies Used

- Python
- Google Colab
- NumPy
- Pandas
- Matplotlib
- PIL / Pillow
- Scikit-learn
- Support Vector Machine (SVM)
- GridSearchCV

## Path Note

The original notebook was developed in Google Colab and accessed the dataset through Google Drive using:
from google.colab import drivedrive.mount('/content/gdrive')

The dataset has also been included in this GitHub repository so that the complete project files are stored together for reference.
Some dataset paths in the repository version may have been updated to reflect the included Data folder structure, while the original Google Drive mounting code has been retained for reference.

## Skills Demonstrated

- Machine Learning
- Python Programming
- Image Classification
- Data Preprocessing
- NumPy
- Support Vector Machines
- Hyperparameter Tuning
- GridSearchCV
- Model Evaluation
- Confusion Matrix
- Data Visualisation
