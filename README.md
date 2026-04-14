# AI-Powered Predictive Maintenance for IoT Devices 🛠️📈

A machine learning solution designed to analyze real-time IoT sensor streams (temperature, vibration, pressure) to predict equipment failure before it occurs.

## 🌟 Key Features
- **Early Warning System:** Predicts mechanical failures based on multivariate sensor thresholds.
- **Simulated IoT Pipeline:** Generates synthetic time-series data to model industrial machine degradation.
- **Ensemble Learning:** Uses a Random Forest Classifier to achieve high recall, ensuring no critical failures are missed.
- **Actionable Alerts:** Built-in logic to trigger maintenance schedules when anomaly probability exceeds 80%.

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **Libraries:** Scikit-Learn, Pandas, NumPy, Seaborn
- **Algorithm:** Random Forest (Classification)

## 📊 Results & Visualization
- **Accuracy:** ~9X%
- **Visualization:** See `outputs/sensor_trends.png` for the correlation between machine age, temperature spikes, and failure events.

## 📂 Project Structure
- `src/processing.py`: Handles data cleaning and feature engineering.
- `main.py`: The main execution hub for training and testing the AI model.
- `models/`: Stores the pre-trained `.pkl` model for deployment.
