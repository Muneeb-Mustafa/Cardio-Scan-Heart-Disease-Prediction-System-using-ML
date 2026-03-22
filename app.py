from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import json
from datetime import datetime

app = Flask(__name__)

# ── Load model & data ──────────────────────────────────────────────────────────
data = pickle.load(open('./models/data.pkl', 'rb'))
pipe = pickle.load(open('./models/model.pkl', 'rb'))

# ── In-memory patient history (resets on restart) ─────────────────────────────
patient_history = []

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURE_KEYS  = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']
FEATURE_NAMES = ['Age','Sex','Chest Pain','Blood Pressure','Cholesterol',
                 'Fasting Sugar','ECG','Max Heart Rate','Exercise Angina',
                 'ST Depression','Slope','Major Vessels','Thalassemia']


def get_options():
    return {
        'sexs':     [int(x) for x in sorted(data['sex'].unique())],
        'cps':      [int(x) for x in sorted(data['cp'].unique())],
        'fbss':     [int(x) for x in sorted(data['fbs'].unique())],
        'restecgs': [int(x) for x in sorted(data['restecg'].unique())],
        'exangs':   [int(x) for x in sorted(data['exang'].unique())],
        'slopes':   [int(x) for x in sorted(data['slope'].unique())],
        'cas':      [int(x) for x in sorted(data['ca'].unique())],
        'thals':    [int(x) for x in sorted(data['thal'].unique())],
    }


def feature_importance_data():
    importances = pipe.feature_importances_.tolist()
    paired = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in paired]
    values = [round(p[1] * 100, 1) for p in paired]
    return json.dumps(labels), json.dumps(values)


def validate_inputs(age, trestbps, chol, thalach, oldpeak, bmi):
    warnings = []
    if trestbps >= 180:
        warnings.append("Blood pressure >= 180 mmHg — critically high")
    elif trestbps >= 140:
        warnings.append("Blood pressure >= 140 mmHg — elevated (Stage 2)")
    if chol >= 240:
        warnings.append("Cholesterol >= 240 mg/dl — high risk range")
    elif chol >= 200:
        warnings.append("Cholesterol 200–239 mg/dl — borderline high")
    if thalach < 80:
        warnings.append("Max heart rate < 80 bpm — unusually low")
    if oldpeak > 4.0:
        warnings.append("ST depression > 4.0 — significant finding")
    if age >= 70:
        warnings.append("Patient age >= 70 — risk factors may be amplified")
    if bmi and bmi >= 30:
        warnings.append(f"BMI {bmi} — obese range (>= 30)")
    elif bmi and bmi >= 25:
        warnings.append(f"BMI {bmi} — overweight range (25–29.9)")
    return warnings


@app.route('/', methods=['GET'])
def index():
    feat_labels, feat_values = feature_importance_data()
    return render_template('index.html',
        **get_options(),
        age=None, sex=None, cp=None, trestbps=None, chol=None,
        fbs=None, restecg=None, thalach=None, exang=None,
        oldpeak=None, slope=None, ca=None, thal=None,
        height=None, weight=None,
        prediction=None, risk_pct=None, bmi=None, warnings=[],
        feat_labels=feat_labels, feat_values=feat_values,
        history=list(reversed(patient_history[-10:])),
    )


@app.route('/predict', methods=['POST'])
def predict():
    age      = int(request.form['age'])
    sex      = int(request.form['sex'])
    cp       = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol     = int(request.form['chol'])
    fbs      = int(request.form['fbs'])
    restecg  = int(request.form['restecg'])
    thalach  = int(request.form['thalach'])
    exang    = int(request.form['exang'])
    oldpeak  = float(request.form['oldpeak'])
    slope    = int(request.form['slope'])
    ca       = int(request.form['ca'])
    thal     = int(request.form['thal'])
    height   = request.form.get('height', '').strip()
    weight   = request.form.get('weight', '').strip()

    bmi = None
    if height and weight:
        try:
            h_m = float(height) / 100
            bmi = round(float(weight) / (h_m ** 2), 1)
        except (ValueError, ZeroDivisionError):
            bmi = None

    query = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
        columns=FEATURE_KEYS
    )
    prediction = int(pipe.predict(query)[0])
    risk_pct   = round(float(pipe.predict_proba(query)[0][1]) * 100, 1)
    warnings   = validate_inputs(age, trestbps, chol, thalach, oldpeak, bmi)

    patient_history.append({
        'time':   datetime.now().strftime('%d %b %H:%M'),
        'age':    age,
        'sex':    'Male' if sex == 1 else 'Female',
        'risk':   risk_pct,
        'result': 'Disease' if prediction == 1 else 'No Disease',
    })
    if len(patient_history) > 50:
        patient_history.pop(0)

    feat_labels, feat_values = feature_importance_data()

    return render_template('index.html',
        **get_options(),
        prediction=prediction, risk_pct=risk_pct, bmi=bmi,
        warnings=warnings,
        age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
        fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
        oldpeak=oldpeak, slope=slope, ca=ca, thal=thal,
        height=height, weight=weight,
        feat_labels=feat_labels, feat_values=feat_values,
        history=list(reversed(patient_history[-10:])),
    )


@app.route('/clear-history', methods=['POST'])
def clear_history():
    patient_history.clear()
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)