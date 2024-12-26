# energy_efficiency_predictor
This project aims to use machine learning regression models to predict energy efficiency metrics for buildings based on input features such as building characteristics and environmental factors. 

The project uses machine learning techniques, data preprocessing, and MLflow for model tracking and experiment management.

## Table of Contents
1. [Overview]()
2. [Project Features]()
3. [Technologies Used]()
4. [Installation and Setup]()
5. [How to Run]()
6. [Future Improvements]()
7. [License]()
8. [Contributing]()
9. [Credits]()

## Overview
Energy efficiency is a critical consideration in modern buildings.
[The dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency) used comes from [UC Irvine Machine learning Repository](https://archive.ics.uci.edu/).
Leveraging machine learning, this project trains, evaluates, and logs regression models to improve prediction for energy efficiency.

Tasks include:
- Predicting energy efficiency using features like noise levels and building properties.
- Evaluating the models based on metrics such as Mean Squared Error (MSE) and R-squared (R²).
- Logging experiment data and metrics with **MLflow** for version control and reproducibility.

## Project Features
- **Data Preprocessing**: Clean and structure train/test sets for model training.
- **Machine Learning Models**: Regularized PCA regression, Decision tree regressor.
- **Multi-Tracking with MLflow**: Tracks experiments, model metrics, and artifacts.
- **Cross-validation**: Evaluates model performance using K-fold validation.

## Technologies Used
- Python (3.12.8)
- Key Libraries:
    - **scikit-learn**: For building machine learning models and evaluate them.
    - **pandas/numpy**: Data handling and numerical computations.
    - **MLflow**: Experiment tracking and model management.
    - **seaborn/matplotlib**: For visualizing model results.

## Installation and Setup
Ensure you have Python 3.12+ installed. Follow these steps to set up the development environment:
1. Clone the repository:

    `git clone git@github.com:pfacouetey/energy_efficiency_predictor.git`

    `cd energy_efficiency_predictor`

2. Create a virtual environment `venv`:

   - For Windows users : `python -m venv venv`

   - For Mac or Linux users : `python3 -m venv venv`

4. Activate the newly created virtual environment `venv`:
    
   - For Windows users : `venv\Scripts\Activate.ps1` (with Powershell) or `venv\Scripts\activate.bat` (with Command Prompt)

   - For Mac or Linux users : `source venv/bin/activate`
   
2. Install `poetry` in the newly created virtual environment `venv`:

    `pip install poetry`

3. Install project dependencies in the newly created virtual environment `venv` using `poetry`:

    `poetry install --no-root`

Feel free to create your virtual envrionment using your own ways, activate it, install `poetry` in it, then use `poetry` to instqll in it the necessary packages.

With Poetry, your dependencies are locked in the `poetry.lock` file, ensuring consistency across environments.
Let me know if you'd like me to enrich this README further or include any other project-specific details!

## How to Run
Open the `notebooks folder`, and execute the `main_copy.ipynb` notebook cells step by step.
Don't forget to use the kernel associated to the virtual environment managed by `poetry`.

Trying to run `main.ipynb` file will result in a failure, since you need a `data folder` you can't access.

The data folder has been created to track any change on the dataset done by [UC Irvine Machine learning Repository](https://archive.ics.uci.edu/).

## Future Improvements
- Enhance preprocessing pipelines to handle outliers detection.
- Include visualization dashboards for energy efficiency predictions.
- Add deployment pipelines (e.g., Flask or FastAPI) for serving the final model retained.
- Compare performance with additional machine learning models (e.g., Random Forest, Gradient Boosting), and even test some deep learning models.

## License
This project is registered under the terms of the MIT license. For any further details, consult the `LICENSE` file.

## Contributing
Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## Credits
`energy_efficiency_predictor` was created by Prince Foli Acouetey, and was based on data originated from [UC Irvine Machine learning Repository](https://archive.ics.uci.edu/).
