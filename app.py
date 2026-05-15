from flask import Flask, request, render_template, jsonify, make_response
import pickle
import pandas as pd
import json
from datetime import datetime
from io import BytesIO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)

# ── Load model & data ──────────────────────────────────────────────────────────
data = pickle.load(open(os.path.join(BASE_DIR, 'models', 'data.pkl'), 'rb'))
pipe = pickle.load(open(os.path.join(BASE_DIR, 'models', 'model.pkl'), 'rb'))

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
        warnings.append("Cholesterol 200-239 mg/dl — borderline high")
    if thalach < 80:
        warnings.append("Max heart rate < 80 bpm — unusually low")
    if oldpeak > 4.0:
        warnings.append("ST depression > 4.0 — significant finding")
    if age >= 70:
        warnings.append("Patient age >= 70 — risk factors may be amplified")
    if bmi and bmi >= 30:
        warnings.append(f"BMI {bmi} — obese range (>= 30)")
    elif bmi and bmi >= 25:
        warnings.append(f"BMI {bmi} — overweight range (25-29.9)")
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
        patient_name='',
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
    patient_name = request.form.get('patient_name', '').strip()

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
        'time':     datetime.now().strftime('%d %b %H:%M'),
        'age':      age,
        'sex':      'Male' if sex == 1 else 'Female',
        'risk':     risk_pct,
        'result':   'Disease' if prediction == 1 else 'No Disease',
        # store full data for PDF export
        'sex_raw':  sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,
        'height': height, 'weight': weight, 'bmi': bmi,
        'warnings': warnings, 'prediction': prediction,
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
        patient_name=patient_name,
        feat_labels=feat_labels, feat_values=feat_values,
        history=list(reversed(patient_history[-10:])),
    )


@app.route('/clear-history', methods=['POST'])
def clear_history():
    patient_history.clear()
    return jsonify({'status': 'ok'})


# ── PDF Export ─────────────────────────────────────────────────────────────────
@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    # Read form data
    age      = request.form.get('age', 'N/A')
    sex_raw  = request.form.get('sex', '')
    sex      = 'Male' if sex_raw == '1' else 'Female'
    cp_map   = {'0': 'Typical Angina', '1': 'Atypical Angina', '2': 'Non-anginal Pain', '3': 'Asymptomatic'}
    cp       = cp_map.get(request.form.get('cp', ''), 'N/A')
    trestbps = request.form.get('trestbps', 'N/A')
    chol     = request.form.get('chol', 'N/A')
    fbs      = 'True (>=120)' if request.form.get('fbs') == '1' else 'False (<120)'
    ecg_map  = {'0': 'Normal', '1': 'ST-T Wave Abnormality', '2': 'LV Hypertrophy'}
    restecg  = ecg_map.get(request.form.get('restecg', ''), 'N/A')
    thalach  = request.form.get('thalach', 'N/A')
    exang    = 'Yes' if request.form.get('exang') == '1' else 'No'
    oldpeak  = request.form.get('oldpeak', 'N/A')
    slope_map = {'0': 'Upsloping', '1': 'Flat', '2': 'Downsloping'}
    slope    = slope_map.get(request.form.get('slope', ''), 'N/A')
    ca       = request.form.get('ca', 'N/A')
    thal_map = {'0': 'Normal', '1': 'Fixed Defect', '2': 'Reversible Defect'}
    thal     = thal_map.get(request.form.get('thal', ''), 'N/A')
    height   = request.form.get('height', '')
    weight   = request.form.get('weight', '')
    bmi      = request.form.get('bmi', '')
    risk_pct = request.form.get('risk_pct', 'N/A')
    prediction = request.form.get('prediction', '0')
    warnings_raw = request.form.get('warnings', '')
    warnings = [w.strip() for w in warnings_raw.split('|') if w.strip()]
    report_time = datetime.now().strftime('%d %B %Y, %H:%M')
    patient_name = request.form.get('patient_name', '').strip() or 'Patient'

    # ── Build PDF in memory ───────────────────────────────────────────────────
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm,
        title=f'CardioScan Report - {patient_name}',
        author='CardioScan',
        subject='Cardiac Risk Assessment Report',
    )

    # Colour palette
    NAVY      = colors.HexColor('#0d1b2a')
    TEAL      = colors.HexColor('#00c4cc')
    TEAL_DARK = colors.HexColor('#007b8a')
    RED       = colors.HexColor('#ff4757')
    GREEN     = colors.HexColor('#2ed573')
    YELLOW    = colors.HexColor('#ffc107')
    WHITE     = colors.white
    LIGHT     = colors.HexColor('#e8f4fd')
    GREY      = colors.HexColor('#8ba5c0')
    CARD_BG   = colors.HexColor('#16293d')

    styles = getSampleStyleSheet()

    def style(name, **kw):
        s = ParagraphStyle(name, **kw)
        return s

    title_style = style('Title2',
        fontSize=22, textColor=WHITE, alignment=TA_CENTER,
        fontName='Helvetica-Bold', spaceAfter=2)

    sub_style = style('Sub',
        fontSize=9, textColor=GREY, alignment=TA_CENTER,
        fontName='Helvetica', spaceAfter=4)

    section_style = style('Section',
        fontSize=10, textColor=TEAL, fontName='Helvetica-Bold',
        spaceBefore=10, spaceAfter=6, leftIndent=0)

    cell_label = style('CellLabel',
        fontSize=8, textColor=GREY, fontName='Helvetica-Bold')

    cell_value = style('CellValue',
        fontSize=10, textColor=LIGHT, fontName='Helvetica')

    disclaimer_style = style('Disclaimer',
        fontSize=7.5, textColor=GREY, alignment=TA_CENTER,
        fontName='Helvetica-Oblique', spaceBefore=6)

    story = []

    # ── Header block ──────────────────────────────────────────────────────────
     
    # Stack vertically in one column 
    header_table = Table(
        [[Paragraph('CardioScan', style('H', fontSize=24, textColor=WHITE,
                    fontName='Helvetica-Bold', alignment=TA_CENTER))],
         [Paragraph('AI-Powered Cardiac Risk Report', style('HS', fontSize=10,
                    textColor=TEAL, fontName='Helvetica', alignment=TA_CENTER))],
         [Paragraph(f'Generated: {report_time}', style('HT', fontSize=8,
                    textColor=GREY, fontName='Helvetica', alignment=TA_CENTER))]],
        colWidths=[170*mm]
    )
    header_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), NAVY),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 0),
        ('RIGHTPADDING',  (0,0), (-1,-1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8*mm))

    # ── Result banner ─────────────────────────────────────────────────────────
    is_disease = prediction == '1'
    result_color  = RED   if is_disease else GREEN
    result_bg     = colors.HexColor('#2a0d0f') if is_disease else colors.HexColor('#0d2a14')
    result_border = colors.HexColor('#ff4757') if is_disease else colors.HexColor('#2ed573')
    result_text   = 'DISEASE DETECTED' if is_disease else 'NO DISEASE DETECTED'
    result_sub    = ('High cardiac risk indicators found. Consult a physician immediately.'
                     if is_disease else
                     'No significant cardiac risk indicators detected.')

    result_table = Table(
        [[Paragraph(result_text, style('RT', fontSize=18, textColor=result_color,
                    fontName='Helvetica-Bold', alignment=TA_CENTER))],
         [Paragraph(result_sub, style('RS', fontSize=9, textColor=GREY,
                    fontName='Helvetica', alignment=TA_CENTER))],
         [Paragraph(f'Risk Score: {risk_pct}%', style('RR', fontSize=13,
                    textColor=result_color, fontName='Helvetica-Bold', alignment=TA_CENTER))]],
        colWidths=[170*mm]
    )
    result_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), result_bg),
        ('LINEABOVE',     (0,0), (-1,0),  2, result_border),
        ('LINEBELOW',     (0,-1),(-1,-1), 2, result_border),
        ('LINEBEFORE',    (0,0), (0,-1),  2, result_border),
        ('LINEAFTER',     (-1,0),(-1,-1), 2, result_border),
        ('TOPPADDING',    (0,0), (-1,0),  10),
        ('BOTTOMPADDING', (0,-1),(-1,-1), 10),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('RIGHTPADDING',  (0,0), (-1,-1), 12),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 6*mm))

    # ── Patient details ───────────────────────────────────────────────────────
    story.append(Paragraph('Patient Details', section_style))
    story.append(HRFlowable(width='100%', thickness=1, color=TEAL, spaceAfter=4))

    def make_row(label, value):
        return [
            Paragraph(label, cell_label),
            Paragraph(str(value), cell_value),
        ]

    bmi_display = bmi if bmi else 'Not provided'
    hw_display  = f'{height} cm / {weight} kg' if height and weight else 'Not provided'

    demo_data = [
        make_row('Age', f'{age} years'),
        make_row('Biological Sex', sex),
        make_row('Height / Weight', hw_display),
        make_row('BMI', bmi_display),
    ]
    demo_table = Table(demo_data, colWidths=[55*mm, 115*mm])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), CARD_BG),
        ('ROWBACKGROUNDS',(0,0), (-1,-1), [CARD_BG, colors.HexColor('#1a2d42')]),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 5*mm))

    # ── Clinical readings ─────────────────────────────────────────────────────
    story.append(Paragraph('Clinical Readings', section_style))
    story.append(HRFlowable(width='100%', thickness=1, color=TEAL, spaceAfter=4))

    clinical_data = [
        make_row('Chest Pain Type',        cp),
        make_row('Resting Blood Pressure', f'{trestbps} mmHg'),
        make_row('Cholesterol',            f'{chol} mg/dl'),
        make_row('Fasting Blood Sugar',    fbs),
        make_row('Resting ECG',            restecg),
        make_row('Max Heart Rate',         f'{thalach} bpm'),
        make_row('Exercise Induced Angina',exang),
        make_row('ST Depression',          oldpeak),
        make_row('ST Segment Slope',       slope),
        make_row('Major Vessels',          ca),
        make_row('Thalassemia',            thal),
    ]
    clinical_table = Table(clinical_data, colWidths=[75*mm, 95*mm])
    clinical_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), CARD_BG),
        ('ROWBACKGROUNDS',(0,0), (-1,-1), [CARD_BG, colors.HexColor('#1a2d42')]),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(clinical_table)

    # ── Warnings ──────────────────────────────────────────────────────────────
    if warnings:
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph('Clinical Alerts', section_style))
        story.append(HRFlowable(width='100%', thickness=1, color=YELLOW, spaceAfter=4))
        warn_bg = colors.HexColor('#1f1a0a')
        for w in warnings:
            wt = Table(
                [[Paragraph(f'  {w}', style('W', fontSize=9, textColor=YELLOW,
                            fontName='Helvetica'))]],
                colWidths=[170*mm]
            )
            wt.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), warn_bg),
                ('LINEBEFORE',    (0,0),(0,-1),  3, YELLOW),
                ('TOPPADDING',    (0,0),(-1,-1), 5),
                ('BOTTOMPADDING', (0,0),(-1,-1), 5),
                ('LEFTPADDING',   (0,0),(-1,-1), 8),
            ]))
            story.append(wt)
            story.append(Spacer(1, 2))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GREY))
    story.append(Paragraph(
        'This report is generated by the CardioScan AI model (Random Forest, 88.5% accuracy) '
        'for educational purposes only. It is NOT a substitute for professional medical advice, '
        'diagnosis, or treatment. Always consult a qualified physician.',
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    safe_name = patient_name.replace(' ', '_')
    filename = f"cardioscan_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response = make_response(buffer.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
