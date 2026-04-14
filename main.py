import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# --- 1. SIMULATE IOT SENSOR DATA ---
def generate_iot_data():
    np.random.seed(42)
    rows = 2000
    # Sensors: Temperature (C), Vibration (mm/s), Pressure (psi)
    data = {
        'temp': np.random.normal(70, 5, rows),
        'vibration': np.random.normal(2, 0.5, rows),
        'pressure': np.random.normal(50, 10, rows),
        'age_hours': np.arange(rows)
    }
    df = pd.DataFrame(data)
    
    # Logic: If Temp > 85 and Vibration > 3.5, it's a 'Failure' (Label 1)
    # We add some noise to make it realistic for AI
    df['failure'] = ((df['temp'] * 0.4 + df['vibration'] * 15) > 75).astype(int)
    return df

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    print("📡 Ingesting Virtual IoT Sensor Streams...")
    df = generate_iot_data()

    # --- 2. PREPROCESSING & SPLITTING ---
    X = df[['temp', 'vibration', 'pressure', 'age_hours']]
    y = df['failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. MODEL TRAINING ---
    print("🧠 Training Predictive Maintenance Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- 4. EVALUATION ---
    y_pred = model.predict(X_test)
    print("\n📊 Maintenance Prediction Report:")
    print(classification_report(y_test, y_pred))

    # --- 5. VISUALIZATION (THE PROOF) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['age_hours'], y=df['temp'], hue=df['failure'], palette='coolwarm')
    plt.title('IoT Sensor Monitoring: Temperature vs Age (Failure Points in Red)')
    plt.savefig('outputs/sensor_trends.png')
    
    # Save Model
    joblib.dump(model, 'models/maintenance_model.pkl')
    print("✅ System Ready. Model saved to models/")

    # --- 6. VIRTUAL SIMULATION TEST ---
    sample_sensor = np.array([[90, 4.5, 55, 1999]]) # High temp, high vibration
    prediction = model.predict(sample_sensor)
    if prediction[0] == 1:
        print("\n⚠️ ALERT: High risk of failure detected! Schedule immediate maintenance.")
    else:
        print("\n✅ Machine Status: Healthy.")

if __name__ == "__main__":
    main()