# Titanic Survival Prediction with Python, R, and Docker

## 1. Project Overview

For this homework, I used Python and R to analyze the Titanic dataset and build logistic regression models to predict passenger survival.

I also used Docker to make both programs easier to run. Instead of installing Python, R, and their packages separately, users can run the project directly through Docker.

The main steps in this project are:

- Load and explore the Titanic data.
- Clean missing values and prepare the features.
- Train logistic regression models.
- Calculate training and held-out test accuracy.
- Predict survival for passengers in the Kaggle test dataset.

Both the Python and R scripts print their results in the terminal.

## 2. Project Structure

The GitHub repository is organized as follows:

```text
titanic-survival-docker/
├── README.md
├── .gitignore
└── src/
    ├── data/
    │   └── .gitkeep
    ├── python/
    │   ├── Dockerfile
    │   ├── main.py
    │   └── requirements.txt
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── main.R
```

The `python` and `r` folders contain the code and Dockerfiles for each implementation.

The `data` folder is where the Titanic CSV files should be placed. The dataset is not included in GitHub because it must be downloaded separately.

## 3. Setup and Download the Data

### Step 1: Install Git and Docker

To run this project, you need:

- Git: https://git-scm.com/downloads
- Docker Desktop: https://www.docker.com/products/docker-desktop/
- A Kaggle account to download the data

Make sure Docker Desktop is running.

You can check your Docker installation with:

```bash
docker --version
docker run hello-world
```

If Docker is working, the second command should print `Hello from Docker!`.

### Step 2: Clone the GitHub repository

Open a terminal and run:

```bash
git clone https://github.com/YufanCheng-lizzzz/titanic-survival-docker.git
```

Then move into the project folder:

```bash
cd titanic-survival-docker
```

All commands below should be run from this folder.

### Step 3: Download the Titanic dataset

Go to the Kaggle Titanic competition:

https://www.kaggle.com/competitions/titanic/data

1. Sign in to Kaggle.
2. Join the competition and accept the rules if needed.
3. Click **Download All**.
4. Unzip the downloaded file if necessary.
5. Find `train.csv` and `test.csv`.

Copy both CSV files into the `src/data` folder.

After downloading the data, the folder should look like this:

```text
src/data/
├── .gitkeep
├── train.csv
└── test.csv
```

You can check the files in the terminal:

```bash
ls src/data
```

The `train.csv` file contains 891 passengers and their survival labels. The `test.csv` file contains 418 passengers but does not include the survival labels.

We do not need `gender_submission.csv` for this project.

**Important:** Do not upload the CSV files to GitHub. The `.gitignore` file is set up to exclude them.

## 4. Run the Python Model

The Python version uses pandas and scikit-learn.

### Step 1: Build the Docker image

Run this command from the project root folder:

```bash
docker build -t titanic-python ./src/python
```

This uses the Dockerfile in `src/python` to create a Python 3.11 environment.

The required packages are listed in `requirements.txt`:

- pandas 2.2.3
- scikit-learn 1.6.1

Docker installs these packages automatically when building the image.

### Step 2: Run the container

On macOS or Linux, run:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data:ro" titanic-python
```

This command lets the container read the local Titanic data from `src/data`.

The data is mounted at `/app/data` inside the container. The `:ro` option means read-only, and `--rm` removes the container after it finishes.

The dataset is not copied into the Docker image.

### Step 3: Check the results

The Python script will print the dataset information, missing values, survival rates, cleaned data, model accuracy, and test predictions.

Here are some results from my run:

```text
Training data shape:
(891, 12)

Logistic Regression Results:
Training accuracy: 0.8002

Model Validation Results:
Training accuracy: 0.8076
Held-out test accuracy (20% of train.csv): 0.8045

Test data shape:
(418, 11)

Predicted survival counts:
0    262
1    156
```

The full dataset model achieved about 80% training accuracy. The model trained on the 80% split achieved about 80.45% accuracy on the held-out data.

## 5. Run the R Model

I also implemented the same general process in R.

The R version uses base R functions, including `glm()` for logistic regression. No additional R packages are needed.

### Step 1: Build the R Docker image

Run:

```bash
docker build -t titanic-r ./src/r
```

This creates an R 4.4.3 environment and runs `install_packages.R`.

### Step 2: Run the R container

On macOS or Linux, run:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data:ro" titanic-r
```

The R container uses the same `train.csv` and `test.csv` files as the Python container.

### Step 3: Check the results

The R script also prints the data exploration, cleaning, model accuracy, and test predictions.

My results were:

```text
Training data shape:
[1] 891 12

Logistic Regression Results:
Training accuracy: 0.7980

Model Validation Results:
Training accuracy: 0.8062
Held-out test accuracy (20% of train.csv): 0.7598

Test data shape:
[1] 418 11

Predicted survival counts:
  0   1
262 156
```

The R model achieved 79.80% training accuracy and 75.98% held-out test accuracy.

## 6. Data Cleaning and Model Results

### Data exploration

Before training the models, I checked the missing values and compared survival rates across different groups.

The main missing values in `train.csv` were:

| Column | Missing values |
|---|---:|
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |

I also found that female passengers had a survival rate of about 74.2%, compared with 18.9% for male passengers.

For passenger class, the survival rates were about 63.0% for first class, 47.3% for second class, and 24.2% for third class.

These results suggest that sex and passenger class are useful features for predicting survival.

### Data cleaning

I used seven features in the models:

`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.

For missing values:

- I filled missing `Age` values with the median age.
- I filled missing `Embarked` values with the most common category.
- I filled missing `Fare` values with the median fare when needed.
- I did not use `Cabin` because it has many missing values.

I also converted categorical variables so they could be used in the models. Python uses one-hot encoding, while R uses factors.

For the held-out evaluation, I calculated the missing-value replacements from the training portion only, instead of using information from the test portion.

After cleaning, there were no missing values in the selected model features.

### Model evaluation

I used logistic regression in both Python and R.

First, I trained a model on the full `train.csv` dataset and calculated training accuracy.

Then I split the labeled data into approximately 80% training and 20% testing. This gave me a way to check how the model performs on data it did not train on.

Here is a summary of my results:

| Metric | Python | R |
|---|---:|---:|
| Full training accuracy | 80.02% | 79.80% |
| Training accuracy on 80% split | 80.76% | 80.62% |
| Held-out test accuracy | 80.45% | 75.98% |

The two implementations use different random splitting methods, so their held-out test sets are not exactly the same. This means the accuracy difference does not necessarily show that one implementation is better.

### Kaggle test data

Both scripts also predict survival for the 418 passengers in Kaggle's `test.csv`.

However, this file does not contain the actual `Survived` values. Because of this, I cannot calculate accuracy directly on the Kaggle test file.

Instead, I report the accuracy from the held-out 20% of `train.csv`, where the actual survival labels are available.

I did not use `gender_submission.csv` as the true labels because it is only an example submission.

## 7. Notes and Troubleshooting

Here are a few things to check if the code does not run.

**1. Docker is not found**

Make sure Docker Desktop is installed and running. Try:

```bash
docker --version
```

**2. The CSV files cannot be found**

Check that `train.csv` and `test.csv` are inside `src/data`, not inside the `python` or `r` folders.

Also, make sure the terminal is in the main `titanic-survival-docker` directory before running the Docker commands.

**3. The image does not exist**

Build the image first using `docker build`, then run it using `docker run`.

**4. Changes to the code do not appear**

If you edit `main.py` or `main.R`, you need to rebuild the corresponding Docker image before running it again.

**5. Running on Windows**

The commands above were tested on macOS. For Windows PowerShell, the data mount can be written using `${PWD}` instead of `$(pwd)`. For example:

```powershell
docker run --rm -v "${PWD}/src/data:/app/data:ro" titanic-python
```

The R command uses the same format with `titanic-r`.

---

With these steps, both the Python and R models can be run using Docker. The only data files that users need to download separately are `train.csv` and `test.csv`.