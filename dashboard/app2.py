import sys
sys.stdout.reconfigure(line_buffering=True)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import time
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import label_binarize

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FL IDS — Smart Energy IoT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

:root {
  --mint:     #C8E6C9; --mint-d:   #81C784;
  --lav:      #EDE7F6; --lav-d:    #B39DDB;
  --peach:    #FFE0B2; --peach-d:  #FFAB76;
  --sky:      #BBDEFB; --sky-d:    #64B5F6;
  --rose:     #FADADD; --rose-d:   #F48FB1;
  --cream:    #F7F4EF; --charcoal: #3D3D3D; --muted: #888;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--cream) !important;
  color: var(--charcoal) !important;
}

.stApp { background: #F7F4EF !important; }
.main .block-container { padding: 2rem 2.5rem !important; max-width: 1300px; }

[data-testid="stSidebar"] {
  background: linear-gradient(160deg,#EDE7F6 0%,#E8F5E9 100%) !important;
  border-right: 1px solid #DDD !important;
}
[data-testid="stSidebar"] * { color: var(--charcoal) !important; }
[data-testid="stSidebar"] hr { border-color: #CCC !important; }

[data-testid="stSidebar"] .stRadio label {
  background: white !important; border-radius: 10px !important;
  padding: 8px 14px !important; margin: 3px 0 !important;
  border: 1px solid #E0E0E0 !important; cursor: pointer !important;
  transition: all 0.2s !important; display: block !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: #EDE7F6 !important; border-color: #B39DDB !important;
}

[data-testid="metric-container"] {
  background: white !important; border-radius: 16px !important;
  padding: 1.2rem !important; border: 1px solid #EEE !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size:13px !important; }
[data-testid="stMetricValue"] { color: var(--charcoal) !important; font-size:26px !important; font-weight:600 !important; }

.stButton > button {
  background: linear-gradient(135deg,#B39DDB,#81C784) !important;
  color: white !important; border: none !important;
  border-radius: 12px !important; padding: 10px 24px !important;
  font-weight: 500 !important; transition: all 0.3s !important;
  box-shadow: 0 4px 15px rgba(179,157,219,0.4) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

[data-testid="stFileUploader"] {
  background: white !important; border-radius: 16px !important;
  border: 2px dashed #B39DDB !important; padding: 1rem !important;
}

.stSuccess { background:#E8F5E9 !important; border-radius:12px !important; border-left:4px solid #81C784 !important; }
.stError   { background:#FADADD !important; border-radius:12px !important; border-left:4px solid #F48FB1 !important; }
.stInfo    { background:#E3F2FD !important; border-radius:12px !important; border-left:4px solid #64B5F6 !important; }
.stWarning { background:#FFF8E1 !important; border-radius:12px !important; border-left:4px solid #FFD54F !important; }

.stSlider [data-baseweb="slider"] { padding: 0 !important; }

.stTabs [data-baseweb="tab-list"] {
  background: white !important; border-radius: 12px !important;
  padding: 4px !important; border: 1px solid #EEE !important;
}
.stTabs [data-baseweb="tab"] { border-radius: 10px !important; font-weight:500 !important; }

hr { border-color: #EEE !important; }

@keyframes fadeInUp {
  from { opacity:0; transform:translateY(20px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes pulse {
  0%,100% { opacity:1; }
  50%      { opacity:0.5; }
}
@keyframes countUp {
  from { opacity:0; }
  to   { opacity:1; }
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────
def card(title, value, subtitle, color, icon=""):
    st.markdown(f"""
    <div style="background:{color};border-radius:16px;padding:20px 22px;
                border:1px solid rgba(0,0,0,0.06);
                box-shadow:0 2px 12px rgba(0,0,0,0.05);
                animation:fadeInUp 0.5s ease">
      <p style="margin:0;font-size:11px;color:#666;text-transform:uppercase;
                letter-spacing:.06em;font-weight:500">{icon} {title}</p>
      <p style="margin:6px 0 4px;font-size:30px;font-weight:600;
                color:#3D3D3D">{value}</p>
      <p style="margin:0;font-size:12px;color:#888">{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def section(icon, title, subtitle=""):
    st.markdown(f"""
    <div style="margin:1.5rem 0 1rem;animation:fadeInUp 0.4s ease">
      <h3 style="margin:0;font-family:'DM Serif Display',serif;
                 font-size:22px;color:#3D3D3D">{icon} {title}</h3>
      {"<p style='margin:4px 0 0;font-size:13px;color:#888'>"+subtitle+"</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)

def page_header(title, subtitle):
    st.markdown(f"""
    <div style="padding:1.5rem 0 1rem;animation:fadeInUp 0.4s ease">
      <h1 style="margin:0 0 8px;font-family:'DM Serif Display',serif;
                 font-size:38px;color:#3D3D3D;line-height:1.1">{title}</h1>
      <p style="margin:0;font-size:14px;color:#888">{subtitle}</p>
    </div>
    <hr style="margin:1rem 0">
    """, unsafe_allow_html=True)

def info_box(text, color="#FFF8E1", border="#FFD54F"):
    st.markdown(f"""
    <div style="background:{color};border-radius:14px;padding:14px 18px;
                border-left:4px solid {border};margin:10px 0;font-size:13px;color:#555">
      {text}</div>""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        m = tf.keras.models.load_model("saved_models/fl_model.h5")
        cn = np.load("saved_models/class_names.npy", allow_pickle=True)
        return m, [str(c) for c in cn]
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, ["DDoS","DoS","Normal","Other"]

@st.cache_data
def load_results():
    try:
        with open("evaluation/results.json") as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_test_data():
    try:
        X = np.load("saved_models/X_test.npy")
        y = np.load("saved_models/y_test.npy")
        return X, y
    except:
        return None, None

model, class_names = load_model()
results            = load_results()
X_test, y_test     = load_test_data()
WINDOW             = 20
NUM_CLASSES        = len(class_names)

PASTEL = {
    "DDoS":   "#BBDEFB",
    "DoS":    "#FADADD",
    "Normal": "#C8E6C9",
    "Other":  "#FFE0B2",
}
PASTEL_DARK = {
    "DDoS":   "#64B5F6",
    "DoS":    "#F48FB1",
    "Normal": "#81C784",
    "Other":  "#FFAB76",
}
SEVERITY = {"Normal":"✅ Safe","DDoS":"🔴 Critical","DoS":"🟠 High","Other":"🟡 Medium"}

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
      <div style="font-size:52px">⚡</div>
      <h2 style="margin:8px 0 4px;font-family:'DM Serif Display',serif;
                 font-size:22px;color:#3D3D3D">FL IDS</h2>
      <p style="margin:0;font-size:12px;color:#888">Smart Energy IoT</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    page = st.radio("", [
        "🏠  Home",
        "🔴  Real-time Monitor",
        "🔍  Live Detection",
        "📊  Evaluation Results",
        "📈  Training History",
        "🆚  Baseline Comparison",
        "🧠  Model Explainability",
        "🗺️  Federation Map",
        "🛡️  Attack Encyclopedia",
        "🏗️  Architecture",
        "ℹ️  About",
    ], label_visibility="collapsed")

    st.divider()
    now = datetime.datetime.now().strftime("%H:%M:%S")
    _sidebar_acc = f"{results['accuracy']*100:.2f}% on test set" if results else "not evaluated yet"
    st.markdown(f"""
    <div style="font-size:12px;color:#999;line-height:1.8">
      <b style="color:#666">Model</b><br>CNN + BiLSTM + Attention<br><br>
      <b style="color:#666">Accuracy</b><br>{_sidebar_acc}<br><br>
      <b style="color:#666">Classes</b><br>{' · '.join(class_names)}<br><br>
      <b style="color:#666">System time</b><br>{now}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown("""
    <div style="padding:2rem 0 1rem;animation:fadeInUp 0.5s ease">
      <p style="margin:0;font-size:13px;color:#B39DDB;font-weight:600;
                letter-spacing:.1em;text-transform:uppercase">Final Year Major Project</p>
      <h1 style="margin:6px 0 10px;font-family:'DM Serif Display',serif;
                 font-size:44px;color:#3D3D3D;line-height:1.1">
        Federated Learning<br>Intrusion Detection System</h1>
      <p style="margin:0;font-size:15px;color:#888;max-width:650px;line-height:1.6">
        A privacy-preserving cyberattack detection framework for Smart Energy IoT
        environments — combining Federated Learning with a hybrid CNN + BiLSTM +
        Attention deep learning model trained on CICIoT2023 and EdgeIIoTset datasets.
      </p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    if results is not None:
        f1s = dict(zip(results["classes"], results["per_f1"]))
        c1,c2,c3,c4 = st.columns(4)
        with c1: card("Test Accuracy", f"{results['accuracy']*100:.2f}%",
                      f"on {results['total_samples']:,} test samples", "#E8F5E9", "🎯")
        with c2: card("Macro F1", f"{results['macro_f1']*100:.2f}%",
                      "avg across all classes", "#EDE7F6", "📊")
        with c3: card("Normal F1", f"{f1s.get('Normal',0)*100:.1f}%",
                      "detection rate", "#E3F2FD", "✅")
        with c4: card("DDoS F1", f"{f1s.get('DDoS',0)*100:.1f}%",
                      "attack detection", "#FFF8E1", "🔴")
    else:
        st.warning("⚠️ No evaluation results found — run `evaluation/evaluate.py` first.")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        section("🌐","Federated Clients","5 smart energy IoT environments")
        clients = [
            ("🏠","Smart Home","Client 1","#E8F5E9"),
            ("🚗","EV Charging Station","Client 2","#E3F2FD"),
            ("⚡","Grid Sensor","Client 3","#FFF8E1"),
            ("☀️","Solar/Wind Controller","Client 4","#EDE7F6"),
            ("🏭","Industrial Energy System","Client 5","#FADADD"),
        ]
        for icon,name,cid,color in clients:
            st.markdown(f"""
            <div style="background:{color};border-radius:12px;padding:11px 16px;
                        margin:5px 0;border:1px solid rgba(0,0,0,0.05);
                        display:flex;align-items:center;gap:12px">
              <span style="font-size:20px">{icon}</span>
              <div>
                <span style="font-weight:500;color:#3D3D3D;font-size:13px">{name}</span>
                <span style="color:#aaa;font-size:11px;margin-left:6px">{cid}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        section("🎯","Attack Classes","Multi-class softmax output")
        for cls in class_names:
            color  = PASTEL.get(cls,"#EEE")
            sev    = SEVERITY.get(cls,"🔵 Info")
            st.markdown(f"""
            <div style="background:{color};border-radius:12px;padding:14px 18px;
                        margin:5px 0;border:1px solid rgba(0,0,0,0.06)">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-weight:600;color:#3D3D3D;font-size:14px">{cls}</span>
                <span style="font-size:12px">{sev}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    with col3:
        section("📋","Project Pipeline","End-to-end workflow")
        steps = [
            ("01","Data Collection","CICIoT2023 + EdgeIIoTset","#E8F5E9"),
            ("02","Preprocessing","Clean, encode, normalize","#EDE7F6"),
            ("03","Windowing","Sliding window sequences","#E3F2FD"),
            ("04","FL Training","FedAvg across 5 clients","#FFF8E1"),
            ("05","Evaluation","Accuracy, F1, ROC-AUC","#FADADD"),
            ("06","Deployment","Streamlit dashboard","#E8F5E9"),
        ]
        for num,title,desc,color in steps:
            st.markdown(f"""
            <div style="background:{color};border-radius:12px;padding:10px 16px;
                        margin:5px 0;border:1px solid rgba(0,0,0,0.05);
                        display:flex;align-items:center;gap:12px">
              <span style="background:white;border-radius:8px;padding:4px 8px;
                           font-weight:700;font-size:11px;color:#666">{num}</span>
              <div>
                <p style="margin:0;font-weight:500;font-size:13px;color:#3D3D3D">{title}</p>
                <p style="margin:0;font-size:11px;color:#888">{desc}</p>
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — REAL-TIME MONITOR
# ══════════════════════════════════════════════════════════════════════
elif "Monitor" in page:
    page_header("🔴 Real-time Attack Monitor",
                "Continuous traffic analysis with live threat level assessment")

    threat_placeholder = st.empty()
    stats_placeholder  = st.empty()
    log_placeholder    = st.empty()

    if model is not None and X_test is not None:
        n_windows  = min(30, len(X_test))
        all_preds, all_confs, log_rows = [], [], []
        attack_count = 0

        for i in range(n_windows):
            seq   = X_test[i:i+1]
            proba = model.predict(seq, verbose=0)[0]
            pred  = class_names[np.argmax(proba)]
            conf  = float(np.max(proba))
            all_preds.append(pred)
            all_confs.append(conf)
            if pred != "Normal":
                attack_count += 1

            attack_pct = attack_count / (i+1) * 100
            if attack_pct < 10:
                threat_color, threat_label, threat_bg = "#81C784","LOW","#E8F5E9"
            elif attack_pct < 30:
                threat_color, threat_label, threat_bg = "#FFD54F","MEDIUM","#FFF8E1"
            elif attack_pct < 60:
                threat_color, threat_label, threat_bg = "#FFAB76","HIGH","#FFE0B2"
            else:
                threat_color, threat_label, threat_bg = "#F48FB1","CRITICAL","#FADADD"

            with threat_placeholder.container():
                st.markdown(f"""
                <div style="background:{threat_bg};border-radius:20px;
                            padding:2rem;text-align:center;
                            border:2px solid {threat_color};margin-bottom:1rem">
                  <p style="margin:0;font-size:12px;color:#666;
                            text-transform:uppercase;letter-spacing:.1em">
                    Threat Level</p>
                  <p style="margin:8px 0;font-size:52px;font-weight:700;
                            color:{threat_color};font-family:'DM Serif Display',serif">
                    {threat_label}</p>
                  <p style="margin:0;font-size:14px;color:#888">
                    {attack_count} attacks in {i+1} windows analyzed
                    · {datetime.datetime.now().strftime("%H:%M:%S")}</p>
                </div>""", unsafe_allow_html=True)

            with stats_placeholder.container():
                s1,s2,s3,s4 = st.columns(4)
                with s1: card("Windows Analyzed",str(i+1),"processed","#EDE7F6","📦")
                with s2: card("Attacks Found",str(attack_count),"flagged","#FADADD","🚨")
                with s3: card("Attack Rate",f"{attack_pct:.1f}%","of windows","#FFE0B2","📈")
                with s4: card("Last Prediction",pred,f"conf: {conf:.1%}",
                              PASTEL.get(pred,"#EEE"),"🧠")

            ts = datetime.datetime.now().strftime("%H:%M:%S")
            log_rows.append({
                "Time":       ts,
                "Window":     i+1,
                "Prediction": pred,
                "Confidence": f"{conf:.2%}",
                "Status":     "🔴 ATTACK" if pred!="Normal" else "✅ Normal"
            })
            with log_placeholder.container():
                section("📜","Live Detection Log","Most recent 10 entries")
                df_log = pd.DataFrame(log_rows[-10:][::-1])
                st.dataframe(df_log, use_container_width=True, hide_index=True)

            time.sleep(0.3)

        st.success(f"✅ Monitoring complete — analyzed {n_windows} windows")
        dist = pd.Series(all_preds).value_counts()
        st.bar_chart(dist)
    else:
        st.warning("Model or test data not found. Run training first.")
        st.markdown("""
        <div style="background:#EDE7F6;border-radius:16px;padding:2rem;text-align:center">
          <div style="font-size:48px">🔴</div>
          <p style="font-weight:600;margin:8px 0 4px">Monitor Ready</p>
          <p style="color:#888;font-size:13px">Load model to start live monitoring</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════
elif "Detection" in page:
    page_header("🔍 Live Traffic Detection",
                "Upload network traffic CSV to detect cyberattacks with confidence scoring")

    conf_threshold = st.slider(
        "🎚️ Confidence Threshold — only flag predictions above this level",
        min_value=0.5, max_value=0.99, value=0.70, step=0.01,
        format="%.0f%%"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if not uploaded:
        st.markdown(f"""
        <div style="background:#EDE7F6;border-radius:20px;padding:3rem;
                    text-align:center;border:2px dashed #B39DDB;margin:1rem 0">
          <div style="font-size:48px;margin-bottom:14px">📂</div>
          <p style="font-weight:600;color:#3D3D3D;font-size:16px;margin:0">
            Drop your network traffic CSV here</p>
          <p style="color:#888;font-size:13px;margin:8px 0 0">
            Any CSV with numeric network traffic features ·
            Confidence threshold: {conf_threshold:.0%}</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧪 Quick Test with Saved Test Data"):
            if model is not None and X_test is not None:
                with st.spinner("Running predictions..."):
                    preds, confs, rows = [], [], []
                    for i in range(0, min(200,len(X_test))-WINDOW, WINDOW):
                        proba = model.predict(X_test[i:i+WINDOW][np.newaxis], verbose=0)[0]
                        c_val = float(np.max(proba))
                        p_val = class_names[np.argmax(proba)]
                        if c_val >= conf_threshold:
                            preds.append(p_val); confs.append(c_val); rows.append(i)

                c1,c2,c3,c4 = st.columns(4)
                attacks = [p for p in preds if p!="Normal"]
                with c1: card("Windows",str(len(preds)),"analyzed","#E8F5E9","📦")
                with c2: card("Attacks",str(len(attacks)),"detected","#FADADD","🚨")
                with c3: card("Safe",str(len(preds)-len(attacks)),"normal","#E3F2FD","✅")
                with c4: card("Avg Conf",f"{np.mean(confs):.1%}" if confs else "N/A",
                              "confidence","#EDE7F6","🎯")

                result_df = pd.DataFrame({
                    "Window Start": rows,
                    "Predicted":    preds,
                    "Confidence":   [f"{c:.2%}" for c in confs],
                    "Status":       ["🔴 ATTACK" if p!="Normal" else "✅ Normal" for p in preds]
                })

                def color_rows(row):
                    c = "#FADADD" if "ATTACK" in str(row["Status"]) else "#E8F5E9"
                    return [f"background-color:{c}"]*len(row)

                st.dataframe(result_df.style.apply(color_rows, axis=1),
                             use_container_width=True, hide_index=True)
                st.bar_chart(pd.Series(preds).value_counts())
            else:
                st.error("Model or test data not available.")
    else:
        df = pd.read_csv(uploaded)
        st.markdown(f"""
        <div style="background:#E8F5E9;border-radius:12px;padding:12px 18px;
                    margin-bottom:1rem;border:1px solid #A5D6A7">
          ✅ &nbsp;<b>{uploaded.name}</b> loaded —
          {len(df):,} rows · {len(df.columns)} columns
        </div>""", unsafe_allow_html=True)

        with st.expander("👀 Preview data (first 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)

        numeric = df.select_dtypes(include=np.number)
        if len(numeric) < WINDOW:
            st.error(f"Need at least {WINDOW} rows. Got {len(numeric)}.")
        elif model is None:
            st.error("Model not loaded.")
        else:
            with st.spinner("🔍 Analyzing traffic..."):
                X = numeric.values.astype("float32")
                n = 46
                if X.shape[1] < n:
                    X = np.concatenate([X, np.zeros((X.shape[0],n-X.shape[1]),"float32")],1)
                elif X.shape[1] > n:
                    X = X[:, :n]
                preds, confs, rows = [], [], []
                for i in range(0, len(X)-WINDOW, WINDOW):
                    proba = model.predict(X[i:i+WINDOW][np.newaxis], verbose=0)[0]
                    c_val = float(np.max(proba))
                    if c_val >= conf_threshold:
                        preds.append(class_names[np.argmax(proba)])
                        confs.append(c_val); rows.append(i)

            attacks = [p for p in preds if p!="Normal"]
            c1,c2,c3,c4 = st.columns(4)
            with c1: card("Windows",str(len(preds)),"analyzed","#E8F5E9","📦")
            with c2: card("Attacks",str(len(attacks)),"detected","#FADADD","🚨")
            with c3: card("Normal",str(len(preds)-len(attacks)),"safe","#E3F2FD","✅")
            with c4: card("Avg Conf",f"{np.mean(confs):.1%}" if confs else "N/A","","#EDE7F6","🎯")

            st.markdown("<br>",unsafe_allow_html=True)
            if attacks:
                st.error(f"🚨 **{len(attacks)} attack windows detected** — Types: {', '.join(set(attacks))}")
            else:
                st.success("✅ All traffic appears normal")

            result_df = pd.DataFrame({
                "Window Start": rows,
                "Predicted":    preds,
                "Confidence":   [f"{c:.2%}" for c in confs],
                "Status":       ["🔴 ATTACK" if p!="Normal" else "✅ Normal" for p in preds]
            })
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            col1,col2 = st.columns(2)
            with col1:
                st.markdown("**Attack distribution**")
                st.bar_chart(pd.Series(preds).value_counts())
            with col2:
                st.markdown("**Confidence over time**")
                st.line_chart(pd.DataFrame({"Confidence": confs}))

            csv = result_df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV",
                               csv,"detection_results.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — EVALUATION RESULTS
# ══════════════════════════════════════════════════════════════════════
elif "Evaluation" in page:
    page_header("📊 Model Evaluation Results",
                "Complete performance analysis on the held-out test set")

    if results is None:
        st.error("evaluation/results.json not found — run evaluation/evaluate.py first.")
        st.stop()

    c1,c2,c3,c4 = st.columns(4)
    with c1: card("Accuracy", f"{results['accuracy']*100:.2f}%",
                  f"{results['total_samples']:,} test samples", "#E8F5E9", "🎯")
    with c2: card("Macro F1", f"{results['macro_f1']*100:.2f}%",
                  "avg across classes", "#EDE7F6", "📊")
    with c3: card("Precision", f"{results['precision']*100:.2f}%",
                  "macro avg", "#E3F2FD", "🔬")
    with c4: card("Recall", f"{results['recall']*100:.2f}%",
                  "macro avg", "#FFF8E1", "📡")

    st.markdown("<br>",unsafe_allow_html=True)
    tab1,tab2,tab3,tab4 = st.tabs(["📋 Per-class","🗺️ Confusion Matrix","📈 ROC Curves","📊 Charts"])

    with tab1:
        supports = np.array(results["cm"]).sum(axis=1)
        per_data = {
            "Class": results["classes"],
            "Precision": [f"{v*100:.1f}%" for v in results["per_p"]],
            "Recall":    [f"{v*100:.1f}%" for v in results["per_r"]],
            "F1 Score":  [f"{v*100:.1f}%" for v in results["per_f1"]],
            "Support":   [f"{int(s):,}" for s in supports],
        }
        st.dataframe(pd.DataFrame(per_data),use_container_width=True,hide_index=True)
        low_f1 = [c for c, f1v in zip(results["classes"], results["per_f1"]) if f1v < 0.5]
        if low_f1:
            info_box(f"💡 <b>{', '.join(low_f1)}</b> "
                     f"{'has' if len(low_f1)==1 else 'have'} low F1 — check support above. "
                     f"If it's small relative to other classes, increasing "
                     f"<code>ROWS_PER_CIC_FILE</code> and re-running the pipeline may help.")

    with tab2:
        col1,col2 = st.columns([1,1])
        with col1:
            cm = np.array(results["cm"])
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="PuBu",
                        xticklabels=results["classes"], yticklabels=results["classes"], ax=ax)
            ax.set_title("Confusion Matrix", pad=12)
            ax.set_ylabel("True"); ax.set_xlabel("Predicted")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        with col2:
            section("📖","Reading the Matrix")
            st.markdown("""
            <div style="font-size:13px;color:#555;line-height:1.9">
              <div style="background:#BBDEFB;padding:8px 12px;border-radius:8px;margin:4px 0">
                🔵 <b>Diagonal cells</b> — correct predictions</div>
              <div style="background:#FADADD;padding:8px 12px;border-radius:8px;margin:4px 0">
                🔴 <b>Off-diagonal</b> — misclassifications</div>
              <br>
              <b>Key observations:</b><br>
              · Normal traffic classified perfectly<br>
              · DDoS correctly identified most of the time<br>
              · DoS misclassified due to very few samples<br>
              · Other attacks well separated from Normal
            </div>""", unsafe_allow_html=True)

    with tab3:
        if X_test is not None and model is not None:
            with st.spinner("Computing ROC curves..."):
                sample_size = min(2000,len(X_test))
                idx    = np.random.choice(len(X_test),sample_size,replace=False)
                X_s,y_s = X_test[idx], y_test[idx]
                y_prob  = model.predict(X_s,verbose=0)
                y_bin   = label_binarize(y_s,classes=range(NUM_CLASSES))

            fig,axes = plt.subplots(1,NUM_CLASSES,figsize=(14,4))
            colors_roc = ["#64B5F6","#F48FB1","#81C784","#FFAB76"]
            for i,cls in enumerate(class_names):
                ax = axes[i]
                if y_bin[:,i].sum() > 0:
                    fpr,tpr,_ = roc_curve(y_bin[:,i],y_prob[:,i])
                    roc_auc_val = auc(fpr,tpr)
                    ax.plot(fpr,tpr,color=colors_roc[i],lw=2,
                            label=f"AUC={roc_auc_val:.2f}")
                    ax.fill_between(fpr,tpr,alpha=0.15,color=colors_roc[i])
                ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
                ax.set_title(cls,fontsize=12,fontweight="bold")
                ax.set_xlabel("FPR",fontsize=10)
                ax.set_ylabel("TPR",fontsize=10) if i==0 else None
                ax.legend(fontsize=9)
                ax.set_facecolor("#FAFAF7")
                ax.spines[["top","right"]].set_visible(False)
            fig.suptitle("ROC Curves (One-vs-Rest) per Class",
                         fontsize=13,y=1.02)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.info("Load model and test data to generate ROC curves")

    with tab4:
        col1,col2 = st.columns(2)
        with col1:
            section("📊","F1 Score by Class")
            f1_df = pd.DataFrame({"F1": results["per_f1"]}, index=results["classes"])
            st.bar_chart(f1_df, color="#B39DDB")
        with col2:
            section("📈","Precision vs Recall")
            pr_df = pd.DataFrame({
                "Precision": results["per_p"],
                "Recall":    results["per_r"]
            }, index=results["classes"])
            st.bar_chart(pr_df)

# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — TRAINING HISTORY
# ══════════════════════════════════════════════════════════════════════
elif "Training" in page:
    page_header("📈 Training History",
                "How the model learned over 8 epochs with early stopping")

    st.warning("⚠️ **ILLUSTRATIVE DATA — NOT FROM A REAL TRAINING RUN.** "
               "These numbers are placeholder values, not logged training history. "
               "Do not cite these in the paper.")

    epochs     = [1,2,3,4,5,6,7,8]
    train_acc  = [0.9106,0.9412,0.9439,0.9472,0.9477,0.9497,0.9501,0.9545]
    val_acc    = [0.9429,0.9355,0.9385,0.9445,0.9466,0.9464,0.9378,0.9462]
    train_loss = [0.2350,0.1542,0.1438,0.1369,0.1314,0.1253,0.1236,0.1136]
    val_loss   = [0.1496,0.1642,0.1477,0.1549,0.1342,0.1410,0.1712,0.1372]
    lr         = [0.001,0.001,0.001,0.001,0.001,0.001,0.0005,0.0005]

    c1,c2,c3,c4 = st.columns(4)
    with c1: card("Best Epoch","5","lowest val loss","#E8F5E9","🏆")
    with c2: card("Best Val Loss","0.1342","at epoch 5","#EDE7F6","📉")
    with c3: card("Best Val Acc","94.66%","at epoch 5","#E3F2FD","📈")
    with c4: card("Early Stop","Epoch 8","patience = 3","#FFF8E1","⏹️")

    st.markdown("<br>",unsafe_allow_html=True)
    col1,col2 = st.columns(2)

    with col1:
        section("📈","Accuracy over Epochs")
        acc_df = pd.DataFrame({
            "Train Accuracy": train_acc,
            "Val Accuracy":   val_acc
        }, index=epochs)
        st.line_chart(acc_df)
        info_box("📌 <b>Epoch 5</b> achieved the best validation accuracy of 94.66%. "
                 "EarlyStopping restored weights from this epoch after patience=3 was reached.")

    with col2:
        section("📉","Loss over Epochs")
        loss_df = pd.DataFrame({
            "Train Loss": train_loss,
            "Val Loss":   val_loss
        }, index=epochs)
        st.line_chart(loss_df)
        info_box("📌 <b>ReduceLROnPlateau</b> halved the learning rate at epoch 7 "
                 "(from 0.001 → 0.0005) when val_loss stopped improving.")

    st.markdown("<br>",unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        section("⚡","Learning Rate Schedule")
        lr_df = pd.DataFrame({"Learning Rate":lr},index=epochs)
        st.line_chart(lr_df)

    with col2:
        section("📋","Epoch-by-epoch Details")
        hist_df = pd.DataFrame({
            "Epoch":      epochs,
            "Train Acc":  [f"{v:.4f}" for v in train_acc],
            "Val Acc":    [f"{v:.4f}" for v in val_acc],
            "Train Loss": [f"{v:.4f}" for v in train_loss],
            "Val Loss":   [f"{v:.4f}" for v in val_loss],
            "LR":         [f"{v:.4f}" for v in lr],
            "Note":       ["","","","","🏆 Best","","LR↓","Early Stop"]
        })
        st.dataframe(hist_df,use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════
elif "Baseline" in page:
    page_header("🆚 Baseline Model Comparison",
                "Our CNN+BiLSTM+Attention vs simpler architectures")

    st.warning("⚠️ **ILLUSTRATIVE DATA — NOT FROM REAL TRAINING RUNS.** "
               "These are estimated placeholders. If your paper's Table IV cites "
               "these, verify against real CNN-only/LSTM-only/CNN+LSTM training "
               "runs before submission.")

    st.markdown("""
    <div style="background:#EDE7F6;border-radius:16px;padding:16px 20px;
                margin-bottom:1.5rem;border:1px solid #D1C4E9">
      <b style="color:#3D3D3D">📌 Why compare?</b>
      <p style="margin:6px 0 0;font-size:13px;color:#666">
        Each baseline adds one component at a time to show the contribution
        of CNN, BiLSTM, and the Attention mechanism to final performance.
        Results shown are estimated based on typical architecture comparisons
        on similar IoT intrusion datasets.</p>
    </div>""", unsafe_allow_html=True)

    baselines = {
        "Model":["CNN Only","LSTM Only","CNN + LSTM","CNN + BiLSTM",
                 "FL CNN Only","FL CNN + BiLSTM","✨ FL CNN+BiLSTM+Attention"],
        "Accuracy":[0.8821,0.8934,0.9102,0.9234,0.9015,0.9312,0.9465],
        "Macro F1":[0.5812,0.5934,0.6123,0.6345,0.6102,0.6534,0.6840],
        "Params":["~120K","~95K","~190K","~310K","~120K","~310K","~450K"],
        "Training":"Fast,Fast,Medium,Medium,Medium,Slow,Slow".split(","),
        "FL":"❌,❌,❌,❌,✅,✅,✅".split(","),
        "Attention":"❌,❌,❌,❌,❌,❌,✅".split(","),
    }
    df_base = pd.DataFrame(baselines)

    def highlight_best(row):
        if "✨" in str(row["Model"]):
            return ["background-color:#E8F5E9;font-weight:bold"]*len(row)
        return [""]*len(row)

    st.dataframe(df_base.style.apply(highlight_best,axis=1),
                 use_container_width=True, hide_index=True)

    st.markdown("<br>",unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        section("📊","Accuracy Comparison")
        acc_comp = pd.DataFrame({
            "Accuracy": baselines["Accuracy"]
        }, index=[m.replace("✨ ","") for m in baselines["Model"]])
        st.bar_chart(acc_comp,color="#81C784")

    with col2:
        section("📊","Macro F1 Comparison")
        f1_comp = pd.DataFrame({
            "Macro F1": baselines["Macro F1"]
        }, index=[m.replace("✨ ","") for m in baselines["Model"]])
        st.bar_chart(f1_comp,color="#B39DDB")

    st.markdown("<br>",unsafe_allow_html=True)
    section("🔍","Key Takeaways")
    takeaways = [
        ("🧠","Adding Attention",
         "Attention mechanism improved Macro F1 by +3.06% over FL CNN+BiLSTM",
         "#EDE7F6"),
        ("🔄","Federated vs Centralized",
         "FL CNN matches centralized CNN accuracy while preserving data privacy",
         "#E8F5E9"),
        ("⚡","BiLSTM over LSTM",
         "Bidirectional LSTM captures both past and future context, improving by ~1.1%",
         "#E3F2FD"),
        ("🏗️","CNN contribution",
         "CNN branch extracts local spatial patterns that LSTM alone misses",
         "#FFF8E1"),
    ]
    cols = st.columns(2)
    for i,(icon,title,desc,color) in enumerate(takeaways):
        with cols[i%2]:
            st.markdown(f"""
            <div style="background:{color};border-radius:14px;padding:16px 18px;
                        margin:6px 0;border:1px solid rgba(0,0,0,0.05)">
              <p style="margin:0;font-size:18px">{icon}
                <b style="font-size:14px;color:#3D3D3D;margin-left:6px">{title}</b></p>
              <p style="margin:6px 0 0;font-size:13px;color:#666">{desc}</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 7 — MODEL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════
elif "Explainability" in page:
    page_header("🧠 Model Explainability (XAI)",
                "Understanding what the model focuses on when making predictions")

    info_box("🔬 <b>What is XAI?</b> Explainable AI shows <i>why</i> the model made a "
             "decision — not just what it predicted. This makes the IDS trustworthy "
             "and auditable for real-world deployment.",
             "#E3F2FD","#64B5F6")

    tab1,tab2,tab3 = st.tabs(["🎯 Attention Weights","📊 Feature Importance","🔎 Sample Analysis"])

    with tab1:
        section("🎯","Attention Heatmap",
                "Which time steps in the window the model focuses on most")

        if model is not None and X_test is not None:
            sample_idx = st.slider("Select test sample",0,min(100,len(X_test)-1),0)
            seq = X_test[sample_idx:sample_idx+1]

            try:
                attn_layer = model.get_layer("attention_weights")
                attn_model = tf.keras.Model(inputs=model.input, outputs=[model.output, attn_layer.output])
                pred_out, attn_out = attn_model.predict(seq, verbose=0)
                attn_weights = attn_out[0,:,0]
            except ValueError:
                st.warning("⚠️ 'attention_weights' layer not found — is fl_model.h5 up to date with the current architecture.py?")
                st.stop()

            pred_class = class_names[np.argmax(pred_out[0])]
            pred_conf  = float(np.max(pred_out[0]))

            col1,col2 = st.columns([2,1])
            with col1:
                fig,ax = plt.subplots(figsize=(10,3))
                attn_norm = (attn_weights - attn_weights.min()) / \
                            (attn_weights.max() - attn_weights.min() + 1e-8)
                im = ax.imshow(attn_norm.reshape(1,-1),aspect="auto",
                               cmap="RdYlGn",vmin=0,vmax=1)
                ax.set_xlabel("Time Step in Window",fontsize=11)
                ax.set_title(f"Attention Weights — Sample {sample_idx}",fontsize=12)
                ax.set_yticks([])
                ax.set_xticks(range(WINDOW))
                ax.set_xticklabels([f"t{i}" for i in range(WINDOW)],fontsize=8)
                plt.colorbar(im,ax=ax,label="Attention weight")
                fig.patch.set_facecolor("#FAFAF7")
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            with col2:
                color = PASTEL.get(pred_class,"#EEE")
                st.markdown(f"""
                <div style="background:{color};border-radius:14px;padding:20px;
                            text-align:center;margin-top:10px">
                  <p style="margin:0;font-size:12px;color:#666">Prediction</p>
                  <p style="margin:6px 0;font-size:28px;font-weight:700;
                            color:#3D3D3D">{pred_class}</p>
                  <p style="margin:0;font-size:14px;color:#555">
                    Confidence: {pred_conf:.2%}</p>
                  <p style="margin:6px 0 0;font-size:20px">
                    {SEVERITY.get(pred_class,"🔵")}</p>
                </div>""", unsafe_allow_html=True)

            top_steps = np.argsort(attn_weights)[-5:][::-1]
            st.markdown(f"""
            <div style="background:#E8F5E9;border-radius:12px;padding:14px 18px;
                        margin-top:12px;border:1px solid #A5D6A7">
              🎯 <b>Model focused most on time steps:</b>
              {', '.join([f'<code>t{s}</code>' for s in top_steps])}
              — these rows in the traffic window triggered the prediction.
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Load model to view attention weights")

    with tab2:
        section("📊","Top Features by Importance",
                "Which network traffic features contribute most to detection")
        st.warning("⚠️ **ILLUSTRATIVE DATA — NOT COMPUTED FROM THE MODEL.** "
                   "These importance scores are hardcoded, not derived via a real "
                   "method (permutation importance, SHAP, gradient-based attribution).")
        features_cic = [
            "flow_duration","Rate","Srate","Drate","syn_flag_number",
            "ack_flag_number","psh_flag_number","TCP","UDP","ICMP",
            "fin_flag_number","rst_flag_number","IAT","Tot size","AVG",
            "Std","Variance","HTTP","HTTPS","DNS",
        ]
        importance = np.array([
            0.92,0.88,0.85,0.82,0.79,0.76,0.73,0.71,0.68,0.65,
            0.61,0.58,0.54,0.51,0.48,0.44,0.41,0.37,0.33,0.29
        ])
        sorted_idx  = np.argsort(importance)[::-1]
        feat_df = pd.DataFrame({
            "Feature":    [features_cic[i] for i in sorted_idx[:12]],
            "Importance": importance[sorted_idx[:12]]
        })
        col1,col2 = st.columns([2,1])
        with col1:
            fig,ax = plt.subplots(figsize=(9,5))
            colors_feat = plt.cm.Purples(np.linspace(0.4,0.85,12))
            bars = ax.barh(feat_df["Feature"][::-1],
                           feat_df["Importance"][::-1],color=colors_feat)
            ax.set_xlabel("Relative Importance",fontsize=11)
            ax.set_title("Top 12 Features",fontsize=12)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_facecolor("#FAFAF7")
            fig.patch.set_facecolor("#FAFAF7")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        with col2:
            section("💡","Insight")
            st.markdown("""
            <div style="font-size:13px;color:#555;line-height:1.8">
              <b>flow_duration</b> and <b>Rate</b> are the most
              discriminative features — DDoS attacks generate
              abnormally high packet rates over short durations.<br><br>
              <b>TCP/UDP flags</b> (SYN, ACK, PSH) reveal
              attack patterns like SYN floods and port scans.<br><br>
              <b>IAT</b> (Inter-Arrival Time) distinguishes
              automated bot traffic from human traffic.
            </div>""", unsafe_allow_html=True)

    with tab3:
        section("🔎","Sample-level Analysis",
                "Inspect individual predictions with full probability breakdown")
        if model is not None and X_test is not None:
            idx = st.slider("Choose test sample index",
                            0, min(200,len(X_test)-1), 42)
            seq   = X_test[idx:idx+1]
            proba = model.predict(seq,verbose=0)[0]
            pred  = class_names[np.argmax(proba)]
            true  = class_names[y_test[idx]] if y_test is not None else "?"

            cols = st.columns(NUM_CLASSES)
            for i,(cls,p) in enumerate(zip(class_names,proba)):
                with cols[i]:
                    color = PASTEL.get(cls,"#EEE")
                    border = "3px solid #81C784" if cls==pred else "1px solid rgba(0,0,0,0.05)"
                    st.markdown(f"""
                    <div style="background:{color};border-radius:14px;
                                padding:16px;text-align:center;border:{border}">
                      <p style="margin:0;font-weight:600;font-size:14px">{cls}</p>
                      <p style="margin:6px 0;font-size:28px;font-weight:700;
                                color:#3D3D3D">{p:.1%}</p>
                      {"<p style='margin:0;font-size:11px;color:#81C784;font-weight:600'>▲ PREDICTED</p>" if cls==pred else ""}
                    </div>""", unsafe_allow_html=True)

            correct = "✅ Correct" if pred==true else "❌ Wrong"
            st.markdown(f"""
            <div style="background:#F5F5F5;border-radius:12px;padding:14px 18px;
                        margin-top:12px;display:flex;gap:24px;font-size:14px">
              <span>🎯 <b>Predicted:</b> {pred}</span>
              <span>🏷️ <b>True label:</b> {true}</span>
              <span>{correct}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Load model and test data to analyze samples")

# ══════════════════════════════════════════════════════════════════════
# PAGE 8 — FEDERATION MAP
# ══════════════════════════════════════════════════════════════════════
elif "Map" in page:
    page_header("🗺️ Federation Map",
                "How 5 smart energy clients collaborate without sharing raw data")

    st.warning("⚠️ **ILLUSTRATIVE DATA** — per-client sample counts, per-client "
               "accuracy, and the convergence chart below are placeholder values, "
               "not logged from a real federated training run.")

    st.markdown("""
    <div style="background:white;border-radius:20px;padding:2rem;
                border:1px solid #EEE;margin-bottom:1.5rem;
                box-shadow:0 2px 16px rgba(0,0,0,0.05)">
      <p style="font-weight:600;font-size:16px;margin:0 0 1.5rem;color:#3D3D3D">
        🌐 Federated Learning Network Topology</p>

      <div style="display:grid;grid-template-columns:1fr auto 1fr;
                  gap:20px;align-items:center">

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
          <div style="background:#E8F5E9;border-radius:14px;padding:14px;text-align:center">
            <div style="font-size:28px">🏠</div>
            <p style="margin:6px 0 2px;font-weight:600;font-size:12px">Smart Home</p>
            <p style="margin:0;font-size:10px;color:#888">8,591 samples</p>
            <p style="margin:4px 0 0;font-size:10px;color:#81C784">Acc: 94.1%</p>
          </div>
          <div style="background:#E3F2FD;border-radius:14px;padding:14px;text-align:center">
            <div style="font-size:28px">🚗</div>
            <p style="margin:6px 0 2px;font-weight:600;font-size:12px">EV Charging</p>
            <p style="margin:0;font-size:10px;color:#888">8,591 samples</p>
            <p style="margin:4px 0 0;font-size:10px;color:#81C784">Acc: 93.8%</p>
          </div>
          <div style="background:#FFF8E1;border-radius:14px;padding:14px;text-align:center">
            <div style="font-size:28px">⚡</div>
            <p style="margin:6px 0 2px;font-weight:600;font-size:12px">Grid Sensor</p>
            <p style="margin:0;font-size:10px;color:#888">8,591 samples</p>
            <p style="margin:4px 0 0;font-size:10px;color:#81C784">Acc: 94.3%</p>
          </div>
          <div style="background:#EDE7F6;border-radius:14px;padding:14px;text-align:center">
            <div style="font-size:28px">☀️</div>
            <p style="margin:6px 0 2px;font-weight:600;font-size:12px">Solar/Wind</p>
            <p style="margin:0;font-size:10px;color:#888">8,591 samples</p>
            <p style="margin:4px 0 0;font-size:10px;color:#81C784">Acc: 94.0%</p>
          </div>
        </div>

        <div style="text-align:center;padding:0 20px">
          <div style="font-size:11px;color:#aaa;margin-bottom:6px">weights only ⟶</div>
          <div style="font-size:40px">⇄</div>
          <div style="font-size:11px;color:#aaa;margin-top:6px">⟵ global model</div>
        </div>

        <div style="background:linear-gradient(135deg,#EDE7F6,#E8F5E9);
                    border-radius:20px;padding:1.5rem;text-align:center;
                    border:2px solid #B39DDB">
          <div style="font-size:40px">🖥️</div>
          <p style="margin:8px 0 4px;font-weight:700;font-size:16px;color:#3D3D3D">
            FL Server</p>
          <p style="margin:0;font-size:12px;color:#888">FedAvg Aggregator</p>
          <div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:6px">
            <div style="background:white;border-radius:8px;padding:8px">
              <p style="margin:0;font-size:18px;font-weight:700;color:#3D3D3D">10</p>
              <p style="margin:0;font-size:10px;color:#888">FL rounds</p>
            </div>
            <div style="background:white;border-radius:8px;padding:8px">
              <p style="margin:0;font-size:18px;font-weight:700;color:#3D3D3D">5</p>
              <p style="margin:0;font-size:10px;color:#888">clients</p>
            </div>
          </div>
        </div>

      </div>

      <div style="background:#FADADD;border-radius:12px;padding:8px 14px;
                  margin-top:12px;text-align:center;font-size:12px">
        <b>🏭 Industrial Energy System</b> — Client 5 — 8,591 samples — Acc: 94.5%
      </div>
    </div>""", unsafe_allow_html=True)

    section("🔄","FedAvg Process — Round by Round")
    rounds = list(range(1,11))
    global_acc = [0.812,0.851,0.878,0.901,0.918,0.928,0.936,0.941,0.944,0.946]
    global_loss= [0.521,0.412,0.341,0.289,0.241,0.212,0.191,0.178,0.162,0.148]

    col1,col2 = st.columns(2)
    with col1:
        acc_df = pd.DataFrame({"Global Accuracy":global_acc},index=rounds)
        st.line_chart(acc_df)
    with col2:
        loss_df = pd.DataFrame({"Global Loss":global_loss},index=rounds)
        st.line_chart(loss_df)

    info_box("🔒 <b>Privacy guarantee:</b> Raw traffic data from Smart Homes, EV Stations, "
             "and Grid Sensors <b>never leaves the device</b>. Only gradient updates "
             "(model weights) are transmitted to the server — protecting user privacy "
             "while enabling collaborative learning.",
             "#E8F5E9","#81C784")

# ══════════════════════════════════════════════════════════════════════
# PAGE 9 — ATTACK ENCYCLOPEDIA
# ══════════════════════════════════════════════════════════════════════
elif "Encyclopedia" in page:
    page_header("🛡️ Attack Encyclopedia",
                "Visual reference guide to all attack types detected by the IDS")

    _f1_lookup = {}
    if results is not None:
        _f1_lookup = dict(zip(results["classes"],
                               [f"{v*100:.1f}%" for v in results["per_f1"]]))

    attacks_info = [
        {
            "name":"DDoS",
            "full":"Distributed Denial of Service",
            "icon":"🌊",
            "color":"#BBDEFB",
            "severity":"🔴 Critical",
            "f1": _f1_lookup.get("DDoS", "N/A"),
            "desc":"Overwhelms a target system with massive traffic from multiple sources, "
                   "making services unavailable to legitimate users.",
            "smart_energy":"Can disable smart grid control systems, EV charging networks, "
                           "and solar inverter communication — causing city-wide power disruption.",
            "example":"Mirai botnet DDoS against power utility DNS (2016)",
            "indicators":["Unusually high packet rate","Multiple source IPs","SYN/UDP/ICMP floods"],
            "variants":["DDoS-ICMP_Flood","DDoS-UDP_Flood","DDoS-TCP_SYN","DDoS-HTTP_Flood"],
        },
        {
            "name":"DoS",
            "full":"Denial of Service",
            "icon":"🚫",
            "color":"#FADADD",
            "severity":"🟠 High",
            "f1": _f1_lookup.get("DoS", "N/A"),
            "desc":"A single-source attack that exhausts a target's resources — "
                   "CPU, memory, or bandwidth — to deny legitimate access.",
            "smart_energy":"Can crash individual smart meters or SCADA systems "
                           "controlling industrial energy infrastructure.",
            "example":"DoS against substation RTU causing outage",
            "indicators":["Single source IP","Resource exhaustion","Connection timeout"],
            "variants":["DoS-UDP_Flood","DoS-TCP_SYN","DoS-HTTP_Flood"],
        },
        {
            "name":"Normal",
            "full":"Benign Traffic",
            "icon":"✅",
            "color":"#C8E6C9",
            "severity":"✅ Safe",
            "f1": _f1_lookup.get("Normal", "N/A"),
            "desc":"Legitimate network traffic from authorized devices — sensor readings, "
                   "control commands, firmware updates, and routine communication.",
            "smart_energy":"Includes smart meter data uploads, EV charging session "
                           "negotiation, and solar panel telemetry.",
            "example":"Normal MQTT telemetry from 1,000 smart meters",
            "indicators":["Regular intervals","Known source IPs","Expected payload sizes"],
            "variants":["BenignTraffic","Normal","Background"],
        },
        {
            "name":"Other",
            "full":"Other Attack Types",
            "icon":"⚠️",
            "color":"#FFE0B2",
            "severity":"🟡 Medium",
            "f1": _f1_lookup.get("Other", "N/A"),
            "desc":"A broad category covering Botnet, Brute Force, Reconnaissance, "
                   "Port Scan, Spoofing, Injection, and Malware attacks.",
            "smart_energy":"Botnets can weaponize IoT devices; credential attacks "
                           "target energy management portals; injection attacks "
                           "manipulate SCADA commands.",
            "example":"Mirai botnet compromising 100K smart home devices",
            "indicators":["Unusual protocols","Failed auth attempts","Scanning behavior"],
            "variants":["Botnet","Brute Force","Recon","Port Scan","Spoofing","Injection","Malware"],
        },
    ]

    for atk in attacks_info:
        with st.expander(
            f"{atk['icon']} {atk['name']} — {atk['full']} "
            f"| Severity: {atk['severity']} | F1: {atk['f1']}",
            expanded=(atk['name']=="DDoS")):
            col1,col2 = st.columns([2,1])
            with col1:
                st.markdown(f"""
                <div style="background:{atk['color']};border-radius:14px;
                            padding:18px;margin-bottom:10px">
                  <p style="margin:0;font-size:13px;color:#555;line-height:1.7">
                    {atk['desc']}</p>
                </div>
                <b>🏭 Impact on Smart Energy:</b>
                <p style="font-size:13px;color:#555;margin:6px 0 12px">
                  {atk['smart_energy']}</p>
                <b>📌 Real-world example:</b>
                <p style="font-size:13px;color:#555;margin:6px 0 12px">
                  {atk['example']}</p>
                <b>🔍 Detection indicators:</b>
                <ul style="font-size:13px;color:#555;margin:6px 0">
                  {"".join(f"<li>{i}</li>" for i in atk['indicators'])}
                </ul>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="background:white;border-radius:14px;padding:16px;
                            border:1px solid #EEE">
                  <p style="margin:0 0 10px;font-weight:600;font-size:13px">
                    Subtypes / Variants</p>
                  {"".join(f'<div style="background:{atk[chr(99)+chr(111)+chr(108)+chr(111)+chr(114)]};border-radius:8px;padding:6px 10px;margin:4px 0;font-size:12px">{v}</div>' for v in atk["variants"])}
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 10 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════
elif "Architecture" in page:
    page_header("🏗️ Model Architecture",
                "Detailed breakdown of the CNN + BiLSTM + Attention hybrid model")

    col1,col2 = st.columns(2)
    with col1:
        section("🧬","Layer Stack","Input → CNN/BiLSTM → Attention → Softmax")
        layers = [
            ("Input Layer",       "(20, 46)","#EDE7F6","Window × Features"),
            ("Conv1D (64)",       "(20, 64)", "#E3F2FD","Local pattern extraction"),
            ("BatchNorm + Pool",  "(10, 64)", "#E3F2FD","Normalize + downsample"),
            ("Dropout 0.3",       "(10, 64)", "#E3F2FD","Regularization"),
            ("Conv1D (128)",      "(10,128)", "#E3F2FD","Deeper feature maps"),
            ("GlobalAvgPool",     "(128,)",   "#E3F2FD","CNN branch output"),
            ("BiLSTM (64×2)",     "(20,128)", "#E8F5E9","Temporal dependencies"),
            ("BiLSTM (64×2)",     "(20,128)", "#E8F5E9","Deeper temporal"),
            ("Attention Dense",   "(20,  1)", "#FFF8E1","Compute attention scores"),
            ("Attention Softmax", "(20,  1)", "#FFF8E1","Normalize over time"),
            ("Weighted Sum",      "(128,)",   "#FFF8E1","BiLSTM branch output"),
            ("Concatenate",       "(256,)",   "#FADADD","Merge CNN + BiLSTM"),
            ("Dense (256)",       "(256,)",   "#FADADD","Fully connected"),
            ("Dense (128)",       "(128,)",   "#FADADD","Fully connected"),
            ("Softmax Output",    "(4,)",     "#EDE7F6","Class probabilities"),
        ]
        for name,shape,color,desc in layers:
            st.markdown(f"""
            <div style="background:{color};border-radius:9px;padding:9px 14px;
                        margin:3px 0;border:1px solid rgba(0,0,0,0.04);
                        display:flex;justify-content:space-between;align-items:center">
              <div>
                <span style="font-weight:500;font-size:13px;color:#3D3D3D">{name}</span>
                <span style="color:#999;font-size:11px;margin-left:8px">{desc}</span>
              </div>
              <code style="background:white;padding:2px 8px;border-radius:6px;
                           font-size:11px;color:#666">{shape}</code>
            </div>""", unsafe_allow_html=True)

    with col2:
        section("⚙️","Hyperparameter Table")
        params = [
            ("Window Size","20 rows","#E8F5E9"),
            ("Stride","5 rows","#E8F5E9"),
            ("CNN Filters","64 → 128","#E3F2FD"),
            ("CNN Kernel Size","3","#E3F2FD"),
            ("LSTM Units","64 per direction","#EDE7F6"),
            ("BiLSTM Total","128 per layer","#EDE7F6"),
            ("Dropout Rate","0.3","#FFF8E1"),
            ("Attention Type","Scaled dot-product","#FFF8E1"),
            ("Optimizer","Adam","#FADADD"),
            ("Learning Rate","0.001 → 0.0005","#FADADD"),
            ("Batch Size","64","#FADADD"),
            ("Loss Function","Categorical CrossEntropy","#E8F5E9"),
            ("Early Stopping","patience = 3","#E8F5E9"),
            ("LR Reduction","factor=0.5, patience=2","#E3F2FD"),
            ("Total Parameters","~450,000","#EDE7F6"),
        ]
        for name,val,color in params:
            st.markdown(f"""
            <div style="background:{color};border-radius:9px;padding:10px 14px;
                        margin:3px 0;border:1px solid rgba(0,0,0,0.04);
                        display:flex;justify-content:space-between">
              <span style="font-weight:500;font-size:13px;color:#3D3D3D">{name}</span>
              <span style="color:#555;font-size:13px">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        section("📐","Why This Architecture?")
        st.markdown("""
        <div style="font-size:13px;color:#555;line-height:1.9">
          <b>CNN</b> — captures local spatial patterns in traffic features<br>
          <b>BiLSTM</b> — models temporal dependencies in both directions<br>
          <b>Attention</b> — focuses on the most suspicious time steps<br>
          <b>Parallel branches</b> — avoids information bottleneck<br>
          <b>BatchNorm</b> — stabilizes training, reduces internal covariate shift
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 11 — ABOUT
# ══════════════════════════════════════════════════════════════════════
elif "About" in page:
    page_header("ℹ️ About This Project",
                "Final year major project — complete system overview")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:white;border-radius:20px;padding:1.5rem;
                    border:1px solid #EEE;margin-bottom:1rem">
          <p style="font-weight:600;font-size:16px;margin:0 0 1rem">🎯 Project Goal</p>
          <p style="color:#666;font-size:14px;line-height:1.7;margin:0">
            Build a <b>privacy-preserving IDS</b> for Smart Energy IoT environments
            using Federated Learning + CNN + BiLSTM + Attention. Raw data never
            leaves the local IoT device — only model weights are shared.
          </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#EDE7F6;border-radius:20px;padding:1.5rem;
                    border:1px solid #D1C4E9;margin-bottom:1rem">
          <p style="font-weight:600;font-size:16px;margin:0 0 1rem">📚 Datasets</p>
          <div style="background:white;border-radius:10px;padding:12px 14px;margin-bottom:8px">
            <p style="margin:0;font-weight:600;font-size:13px">CICIoT2023</p>
            <p style="margin:3px 0 0;font-size:12px;color:#888">
              University of New Brunswick · 1,352,000 rows · 33 CSV files</p>
          </div>
          <div style="background:white;border-radius:10px;padding:12px 14px">
            <p style="margin:0;font-weight:600;font-size:13px">EdgeIIoTset</p>
            <p style="margin:3px 0 0;font-size:12px;color:#888">
              IEEE DataPort · 69,387 rows · Selected ML dataset</p>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#E3F2FD;border-radius:20px;padding:1.5rem;
                    border:1px solid #BBDEFB">
          <p style="font-weight:600;font-size:16px;margin:0 0 1rem">
            🛠️ Tech Stack</p>
          <div style="font-size:13px;color:#555;line-height:2">
            🐍 Python 3.11<br>
            🧠 TensorFlow / Keras<br>
            🌸 Flower (flwr) — Federated Learning<br>
            🔢 NumPy · Pandas · Scikit-learn<br>
            📊 Matplotlib · Seaborn<br>
            🌐 Streamlit — Dashboard
          </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        if results is not None:
            f1s = dict(zip(results["classes"], results["per_f1"]))
            _acc_str   = f"{results['accuracy']*100:.2f}%"
            _norm_str  = f"{f1s.get('Normal',0)*100:.1f}%"
            _ddos_str  = f"{f1s.get('DDoS',0)*100:.1f}%"
            _other_str = f"{f1s.get('Other',0)*100:.1f}%"
        else:
            _acc_str = _norm_str = _ddos_str = _other_str = "N/A"

        st.markdown(f"""
        <div style="background:#E8F5E9;border-radius:20px;padding:1.5rem;
                    border:1px solid #C8E6C9;margin-bottom:1rem">
          <p style="font-weight:600;font-size:16px;margin:0 0 1rem">🏆 Results</p>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
            <div style="background:white;border-radius:10px;padding:14px;text-align:center">
              <p style="margin:0;font-size:26px;font-weight:700;color:#3D3D3D">{_acc_str}</p>
              <p style="margin:4px 0 0;font-size:11px;color:#888">Test Accuracy</p>
            </div>
            <div style="background:white;border-radius:10px;padding:14px;text-align:center">
              <p style="margin:0;font-size:26px;font-weight:700;color:#3D3D3D">{_norm_str}</p>
              <p style="margin:4px 0 0;font-size:11px;color:#888">Normal F1</p>
            </div>
            <div style="background:white;border-radius:10px;padding:14px;text-align:center">
              <p style="margin:0;font-size:26px;font-weight:700;color:#3D3D3D">{_ddos_str}</p>
              <p style="margin:4px 0 0;font-size:11px;color:#888">DDoS F1</p>
            </div>
            <div style="background:white;border-radius:10px;padding:14px;text-align:center">
              <p style="margin:0;font-size:26px;font-weight:700;color:#3D3D3D">{_other_str}</p>
              <p style="margin:4px 0 0;font-size:11px;color:#888">Other F1</p>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#FFF8E1;border-radius:20px;padding:1.5rem;
                    border:1px solid #FFE082;margin-bottom:1rem">
          <p style="font-weight:600;font-size:16px;margin:0 0 1rem">
            🔒 Privacy Features</p>
          <div style="font-size:13px;color:#666;line-height:2">
            ✅ No raw data leaves client devices<br>
            ✅ Only model weights transmitted<br>
            ✅ FedAvg secure aggregation<br>
            ✅ Non-IID data support<br>
            ✅ 5 heterogeneous client types
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#FADADD;border-radius:20px;padding:1.5rem;
                    border:1px solid #F8BBD0">
          <p style="font-weight:600;font-size:16px;margin:0 0 0.5rem">
            📝 Project Info</p>
          <div style="font-size:13px;color:#666;line-height:2">
            📅 Year: 2025–2026<br>
            🎓 Type: Final Year Major Project<br>
            🏫 Domain: Cybersecurity + AI/ML<br>
            📂 Datasets: CICIoT2023 + EdgeIIoTset<br>
            🌐 Framework: Federated Learning (Flower)
          </div>
        </div>""", unsafe_allow_html=True)