import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import glob

# ─── PATHS ───────────────────────────────────────────────────────────
CICIOT_PATH = "data/raw/CICIoT2023/wataiData/csv/CICIoT2023/"
EDGE_PATH   = "data/raw/EdgeIIoTset/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
SAVE_DIR    = "data/processed/"
os.makedirs(SAVE_DIR, exist_ok=True)

ROWS_PER_CIC_FILE = 20000   # rows sampled per CICIoT CSV file
EDGE_ROWS         = 80000  # rows sampled from EdgeIIoTset

# ─── CICIoT2023 has 46 numeric features ──────────────────────────────
CIC_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration",
    "Rate", "Srate", "Drate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "DNS",
    "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP",
    "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std",
    "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight"
]

# ─── EdgeIIoTset numeric features ────────────────────────────────────
EDGE_FEATURES = [
    "arp.opcode", "arp.hw.size", "icmp.checksum", "icmp.seq_le",
    "icmp.transmit_timestamp", "icmp.unused", "http.file_data",
    "http.content_length", "http.request.uri.query",
    "http.request.method", "http.referer", "http.request.full_uri",
    "http.request.version", "http.response", "http.tls_port",
    "tcp.ack", "tcp.ack_raw", "tcp.checksum", "tcp.connection.fin",
    "tcp.connection.rst", "tcp.connection.syn", "tcp.connection.synack",
    "tcp.dstport", "tcp.flags", "tcp.flags.ack", "tcp.len",
    "tcp.seq", "tcp.srcport", "udp.port", "udp.stream",
    "udp.time_delta", "dns.qry.name.len", "dns.qry.qu",
    "dns.qry.type", "dns.retransmission", "dns.retransmit_request",
    "dns.retransmit_request_in", "mqtt.conflag.cleansess",
    "mqtt.conflags", "mqtt.hdrflags", "mqtt.len",
    "mqtt.msg_decoded_as", "mqtt.msgtype", "mqtt.proto_len",
    "mqtt.topic_len", "mqtt.ver"
]

# ─── LABEL MAPS ──────────────────────────────────────────────────────
CIC_LABEL_MAP = {
    "DDoS-ICMP_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_SYN_Flood": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS", "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS", "DDoS-SlowLoris": "DDoS",
    "DoS-UDP_Flood": "DoS", "DoS-TCP_SYN_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS", "DoS-SYN_Flood": "DoS",
    "Mirai-udpplain": "Botnet", "Mirai-greip": "Botnet",
    "Mirai-greeth_flood": "Botnet", "Mirai-udpflood": "Botnet",
    "Mirai-ack": "Botnet", "Mirai-syn": "Botnet",
    "BruteForce-SSH": "Brute Force", "BruteForce-Web": "Brute Force",
    "BruteForce-XSS": "Brute Force",
    "Recon-HostDiscovery": "Reconnaissance",
    "Recon-OSScan": "Reconnaissance", "Recon-PortScan": "Port Scan",
    "Recon-VulScan": "Reconnaissance","Recon-PingSweep": "Reconnaissance",
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    "SqlInjection": "Injection", "CommandInjection": "Injection",
    "Backdoor_Malware": "Malware", "Uploading_Attack": "Malware",
    "BenignTraffic": "Normal",
}

EDGE_LABEL_MAP = {
    "Normal": "Normal",
    "DDoS_UDP": "DDoS", "DDoS_ICMP": "DDoS",
    "DDoS_HTTP": "DDoS", "DDoS_TCP": "DDoS",
    "DoS_UDP": "DoS", "DoS_ICMP": "DoS",
    "DoS_HTTP": "DoS", "DoS_TCP": "DoS",
    "Backdoor": "Malware", "SQL_injection": "Injection",
    "Port_Scanning": "Port Scan",
    "Vulnerability_scanner": "Reconnaissance",
    "Uploading": "Malware", "Ransomware": "Malware",
    "XSS": "Injection", "Password": "Brute Force",
    "Fingerprinting": "Reconnaissance", "MITM": "Spoofing",
}

# ─── LOAD CICIoT2023 ─────────────────────────────────────────────────
def load_ciciot():
    print("\n📂 Loading CICIoT2023...")
    files = glob.glob(os.path.join(CICIOT_PATH, "*.csv"))
    print(f"   Found {len(files)} CSV files")

    all_dfs = []
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f, low_memory=False)

            # Keep only needed feature columns + label
            available = [c for c in CIC_FEATURES if c in df.columns]
            if "label" not in df.columns:
                print(f"   Skipping {os.path.basename(f)} — no label column")
                continue

            df = df[available + ["label"]].copy()
            df = df.drop_duplicates()
            df = df.dropna()
            df["attack_type"] = df["label"].map(CIC_LABEL_MAP).fillna("Other")
            df = df.drop(columns=["label"])

            if len(df) > ROWS_PER_CIC_FILE:
                df = df.sample(n=ROWS_PER_CIC_FILE, random_state=42)

            all_dfs.append(df)
            print(f"   [{i+1}/{len(files)}] {os.path.basename(f)[:40]} → {len(df)} rows")
            del df

        except Exception as e:
            print(f"   Skipping: {e}")

    result = pd.concat(all_dfs, ignore_index=True)
    print(f"   ✅ CICIoT total: {result.shape}")
    return result

# ─── LOAD EdgeIIoTset ─────────────────────────────────────────────────
def load_edgeiiot():
    print("\n📂 Loading EdgeIIoTset...")
    df = pd.read_csv(EDGE_PATH, low_memory=False, nrows=EDGE_ROWS)

    available = [c for c in EDGE_FEATURES if c in df.columns]
    df = df[available + ["Attack_type"]].copy()
    df = df.drop_duplicates()
    df = df.dropna()
    df["attack_type"] = df["Attack_type"].map(EDGE_LABEL_MAP).fillna("Other")
    df = df.drop(columns=["Attack_type"])

    print(f"   ✅ EdgeIIoTset total: {df.shape}")
    return df

# ─── PROCESS AND SAVE ─────────────────────────────────────────────────
def process_and_save(df, features, name, le=None):
    """Scale, encode, and save one dataset."""
    print(f"\n⚙️  Processing {name}...")

    # Fill NaN
    df = df.fillna(0)

    # Keep only available features
    available = [c for c in features if c in df.columns]
    X = df[available].values.astype(np.float32)
    y_raw = df["attack_type"].values

    # Encode labels
    if le is None:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
    else:
        # Map unseen labels to "Other" index
        known = set(le.classes_)
        y_raw = np.array([l if l in known else "Other" for l in y_raw])
        y = le.transform(y_raw)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save
    np.save(f"{SAVE_DIR}X_{name}.npy", X)
    np.save(f"{SAVE_DIR}y_{name}.npy", y)
    with open(f"{SAVE_DIR}scaler_{name}.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{SAVE_DIR}features_{name}.pkl", "wb") as f:
        pickle.dump(available, f)

    print(f"   X shape   → {X.shape}")
    print(f"   y shape   → {y.shape}")
    print(f"   Features  → {len(available)}")

    return X, y, le, scaler

# ─── MAIN ─────────────────────────────────────────────────────────────
def preprocess():
    # Load both datasets
    cic  = load_ciciot()
    edge = load_edgeiiot()

    # Get all unique classes from BOTH datasets combined
    all_labels = list(cic["attack_type"].unique()) + \
                 list(edge["attack_type"].unique())
    all_labels = sorted(set(all_labels))
    print(f"\n🏷️  All unified classes ({len(all_labels)}): {all_labels}")

    # Fit ONE shared label encoder on all classes
    le = LabelEncoder()
    le.fit(all_labels)

    # Process and save each dataset separately
    X_cic,  y_cic,  le, _  = process_and_save(cic,  CIC_FEATURES,  "cic",  le)
    X_edge, y_edge, le, _  = process_and_save(edge, EDGE_FEATURES, "edge", le)

    # Save shared label encoder
    with open(f"{SAVE_DIR}label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"\n✅ ALL DONE!")
    print(f"   CICIoT    → X:{X_cic.shape}  y:{y_cic.shape}")
    print(f"   EdgeIIoT  → X:{X_edge.shape} y:{y_edge.shape}")
    print(f"   Classes   → {list(le.classes_)}")
    print(f"   Saved to  → {SAVE_DIR}")

if __name__ == "__main__":
    preprocess()