# Algerian Forest Fire Regression Project

This repository hosts an end-to-end Machine Learning pipeline that predicts the **Fire Weather Index (FWI)** using meteorological conditions and underlying index markers. The project treats this ecological problem as a regression task, applying data cleaning, Exploratory Data Analysis (EDA), feature engineering, model training, and regularization techniques to achieve highly accurate predictions. Additionally, it features a deployed **Flask Web Application** to provide real-time interactive model inference.

---

## 📌 Project Overview
Forest fires present a major risk to human lives, properties, and biodiversity. Predicting the Fire Weather Index (FWI)-a key indicator used to estimate fire intensity and behavior-can empower forestry departments to deploy preemptive safety measures. This project automates that assessment by training regression algorithms on real-world climatic measurements.

---

## 📊 Dataset Information
The dataset used in this project originates from the **UCI Machine Learning Repository** and captures observations across two specific regions in Algeria:
1. **Bejaia Region** (Northeast Algeria) - 122 instances
2. **Sidi Bel-Abbes Region** (Northwest Algeria) - 122 instances

**Timeline Covered:** June 2012 to September 2012  
**Total Samples:** 244 instances

### Attribute Description

#### Meteorological Variables:
* **Date:** Day, month (`June` to `September`), and Year (`2012`).
* **Temperature (Temp):** Noon max temperature in Celsius degrees (Range: 22 to 42°C).
* **Relative Humidity (RH):** Humidity percentage value (Range: 21% to 90%).
* **Wind Speed (Ws):** Wind velocity measurement in km/h (Range: 6 to 29 km/h).
* **Rain:** Total precipitations of the day in mm (Range: 0 to 16.8 mm).

#### Fire Weather Index (FWI) Components:
* **FFMC (Fine Fuel Moisture Code):** Numeric index evaluating moisture content of litter and ignition potential (28.6 to 92.5).
* **DMC (Duff Moisture Code):** Numeric index denoting moisture content of shallow organic layers (1.1 to 65.9).
* **DC (Drought Code):** Numeric index representing deep, compact organic layers (7 to 220.4).
* **ISI (Initial Spread Index):** Numeric score rating expected rate of fire spread (0 to 18.5).
* **BUI (Buildup Index):** Total amount of fuel available for combustion (1.1 to 68).
* **Classes:** Categorical labels stating status (`fire` vs `not fire`).

#### Target/Dependent Feature:
* **FWI (Fire Weather Index):** Index indicating the structural intensity of potential fires (Continuous variable: 0 to 31.1).

---

## 🛠 Project Execution Architecture & Steps

The pipeline is organized systematically into the following sequentially designed execution phases:

### Phase 1: Data Preprocessing & Cleaning
* **Regional Separation:** The original text dataset lacks a clear regional feature, but switches sections at row index 122. A custom categorical column (`Region`) was engineered, mapping Bejaia as `1` and Sidi Bel-Abbes as `2`.
* **Sanitization:** Stripped out whitespace inconsistencies, handled empty or misplaced header rows generated during raw dataset collation, and resolved structural formatting issues.
* **Type Casting:** Converted string object features into proper numeric datatypes (`int` and `float`) for model compatibility.
* **Categorical Encoding:** Encoded target indicator classification text features (`fire`, `not fire`) into standard numeric states (`1`, `0`).

### Phase 2: Exploratory Data Analysis (EDA)
* Handled the high multi-collinearity existing between structural components (such as dropping overlapping features like `BUI` and `DC` depending on correlation thresholds).
* Plotted statistical charts utilizing `Matplotlib` and `Seaborn` to check the data distributions, seasonal trends across active fire months (identifying peak incidents during August), and correlation matrix heatmaps.

### Phase 3: Model Pipeline Engineering & Training
* Split the processed matrices into training and test datasets with a standard 75/25 distribution split.
* Executed feature standardization mapping using **StandardScaler** to prevent scale imbalances across independent measurements.
* Scaled features were fit across multiple algorithms to find the most resilient model:
  1. **Linear Regression**
  2. **Lasso Regression (L1 Regularization)** - For automated feature selection.
  3. **Ridge Regression (L2 Regularization)** - To protect model weights against overfitting.
  4. **ElasticNet Regression** - For combined hyperparameter penalty control.
* Evaluated models via standard metrics, picking the optimized **Ridge Regression** model based on its superior R² score and low Mean Absolute Error (MAE).

### Phase 4: Serialization & Deployment
* Serialized the finalized preprocessing transformer object (`scaler.pkl`) and optimized model algorithm weights (`ridge.pkl`) using standard `pickle` serialization.
* Constructed a lightweight web user-interface with an interactive **Flask App Engine** (`application.py` / `app.py`) allowing web browser endpoint querying.

---

## 📁 Repository Structure


├── models/
│   ├── scaler.pkl            # Serialized Standard Scaler object for input standardization
│   └── ridge.pkl             # Trained Ridge Regression Machine Learning model weights
├── notebooks/
│   ├── EDA_Notebook.ipynb    # Jupyter Notebook containing data exploration and preprocessing
│   └── Model_Training.ipynb  # Jupyter Notebook containing feature engineering and model training
├── templates/
│   ├── index.html            # Landing dashboard page design file
│   └── home.html             # Parameter submission web form interface
├── application.py            # Core Flask Web server script entrypoint
├── requirements.txt          # File containing library dependencies to run the project
└── README.md                 # Project configuration manual and overview


## 🚀 Step-by-Step Local Setup & Execution Instructions
Follow these instructions exactly to pull down, set up, and evaluate the algorithm pipeline inside your environment:

Prerequisites
Ensure your local environment has Python 3.8+ installed.

Step 1: Clone the Repository
Bash
git clone [https://github.com/KeshavPal06/Algerian-Forest-Fire-Regression.git](https://github.com/KeshavPal06/Algerian-Forest-Fire-Regression.git)
cd Algerian-Forest-Fire-Regression


Step 2: Establish an Isolated Virtual Environment (Recommended)
Bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Step 3: Install Required Dependencies
Install the required packages list specified within the repository requirements sheet:

Bash
pip install -r requirements.txt


Step 4: Run the Flask Web Server
Kick off the local server runtime tracking mechanism with the execution script command:

Bash
python application.py
(Alternatively, try python app.py depending on the exact script naming configuration).

Step 5: Test the Model Endpoint
Once the development console server begins running locally, navigate to the following URL in your web browser:


Plaintext
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)
Fill out the input feature parameters (Temperature, RH, Wind Speed, Rain, FFMC, DMC, ISI, etc.) within the web page forms UI dashboard and submit to receive instant continuous FWI rating index estimations back.


## 📈 Performance Summary
Selected Model: Ridge Regression

R² Score: ~0.984 (Explaining over 98% of target variation during evaluations)

Mean Absolute Error (MAE): ~0.56


## 🧰 Technologies & Toolkits Used
Core Language: Python

Data Wrangling & Visualizations: Pandas, NumPy, Matplotlib, Seaborn

Model Pipeline Training: Scikit-Learn (sklearn)

API Engine & Web Server Framework: Flask

Serialization Utilities: Pickle
