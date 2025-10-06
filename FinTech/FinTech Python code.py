
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
import shutil

warnings.filterwarnings("ignore")

DATA = {
    "banks": pd.DataFrame(),
    "entities": pd.DataFrame(),
    "accounts": pd.DataFrame(),
    "transactions": pd.DataFrame(),
    "ledger": pd.DataFrame(),
    "partner_statements": pd.DataFrame(),
    "recon_matches": pd.DataFrame(),
    "recon_exceptions": pd.DataFrame(),
    "patterns_raw": pd.DataFrame(),
    "patterns_parsed": pd.DataFrame(),
    "pattern_tags": pd.DataFrame(),
    "alerts": [],
}

VIEWS = {
    "txn_hourly_health": pd.DataFrame(),
    "failure_reasons": pd.DataFrame(),
    "recon_open": pd.DataFrame(),
    "recon_aging": pd.DataFrame(),
    "pattern_hourly_health": pd.DataFrame(),
}

def log(msg: str):
    """
    Print a timestamped message for simple runtime logging.
    """ 
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def _safe_accounts_columns(df: pd.DataFrame):
    """
    Standardize the accounts dataframe and return columns account_number, bank_id, entity_id.
    """
    cols = df.columns.str.strip()
    df.columns = cols
    colmap = {
        "Account Number": "account_number",
        "Bank ID": "bank_id",
        "Entity ID": "entity_id",
    }
    miss = [c for c in ["Account Number","Bank ID","Entity ID"] if c not in cols]
    if miss:
        raise ValueError(f"Missing required account columns: {miss}")
    out = df[["Account Number","Bank ID","Entity ID"]].rename(columns=colmap)
    for c in out.columns:
        out[c] = out[c].astype(str).str.strip()
    return out

def load_all_accounts(folder: Path):
    """
    Load account files from folder, and build banks, entities, and accounts tables.
    Expected file pattern is *_accounts.csv.
    """
    files = sorted(Path(folder).glob("*_accounts.csv"))
    if not files:
        log("No *_accounts.csv files found")
        return
    banks_all, ents_all, accts_all = [], [], []
    for f in files:
        raw = pd.read_csv(f)
        raw.columns = raw.columns.str.strip()
        if {"Bank ID","Bank Name"}.issubset(raw.columns):
            b = raw[["Bank ID","Bank Name"]].drop_duplicates()
            b.columns = ["bank_id","bank_name"]
            banks_all.append(b)
        if {"Entity ID","Entity Name"}.issubset(raw.columns):
            e = raw[["Entity ID","Entity Name"]].drop_duplicates()
            e.columns = ["entity_id","entity_name"]
            ents_all.append(e)
        accts_all.append(_safe_accounts_columns(raw))
    if banks_all:
        DATA["banks"] = pd.concat([DATA["banks"], *banks_all], ignore_index=True).drop_duplicates()
    if ents_all:
        DATA["entities"] = pd.concat([DATA["entities"], *ents_all], ignore_index=True).drop_duplicates()
    if accts_all:
        DATA["accounts"] = pd.concat([DATA["accounts"], *accts_all], ignore_index=True).drop_duplicates()
    log(f"Accounts loaded. Banks={len(DATA['banks'])}, Entities={len(DATA['entities'])}, Accounts={len(DATA['accounts'])}")

def _clean_transactions_file(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Clean and normalize a single transactions file, return a standardized dataframe.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    from_acc_col = "From Bank Account" if "From Bank Account" in df.columns else ("Account" if "Account" in df.columns else None)
    to_acc_col   = "To Bank Account"   if "To Bank Account" in df.columns   else ("Account.1" if "Account.1" in df.columns else None)
    if not from_acc_col or not to_acc_col:
        cols = list(df.columns)
        try:
            from_acc_col = cols[cols.index("From Bank") + 1]
            to_acc_col   = cols[cols.index("To Bank") + 1]
        except Exception:
            raise ValueError(f"Cannot infer account columns from headers: {list(df.columns)}")
    rename_map = {
        "Timestamp": "txn_ts",
        "From Bank": "from_bank_id",
        from_acc_col: "from_account",
        "To Bank": "to_bank_id",
        to_acc_col: "to_account",
        "Amount Received": "amount_received",
        "Receiving Currency": "recv_currency",
        "Amount Paid": "amount_paid",
        "Payment Currency": "pay_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering",
    }
    keep = [c for c in rename_map if c in df.columns]
    df = df[keep].rename(columns=rename_map)
    df["txn_ts"] = pd.to_datetime(df["txn_ts"], errors="coerce")
    for c in ["amount_received","amount_paid"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "is_laundering" in df.columns:
        df["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_laundering"] = 0
    for c in ["from_bank_id","from_account","to_bank_id","to_account","recv_currency","pay_currency","payment_format"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df = df[df["txn_ts"].notna() & ((df["amount_received"]>0) | (df["amount_paid"]>0))]
    df["source_file"] = source_name
    df["source_txn_key"] = (
        df["txn_ts"].astype(str) + "|" +
        df["from_bank_id"].fillna("") + "|" + df["from_account"].fillna("") + "|" +
        df["to_bank_id"].fillna("") + "|" + df["to_account"].fillna("") + "|" +
        df["amount_received"].round(2).astype(str) + "|" +
        df["amount_paid"].round(2).astype(str)
    )
    return df

def load_all_transactions(folder: Path):
    """
    Load and combine all transaction files from folder, removing duplicates by a stable key.
    Expected file pattern is *_Trans.csv.
    """
    files = sorted(Path(folder).glob("*_Trans.csv"))
    if not files:
        log("No *_Trans.csv files found")
        return
    frames = []
    for f in files:
        raw = pd.read_csv(f)
        frames.append(_clean_transactions_file(raw, f.name))
    tx = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["source_txn_key"])
    DATA["transactions"] = pd.concat([DATA["transactions"], tx], ignore_index=True)\
                             .drop_duplicates(subset=["source_txn_key"])
    log(f"Transactions loaded: {len(DATA['transactions'])}")

def load_all_patterns(folder: Path):
    """
    Read all pattern files, capture raw lines and parse transaction like rows inside pattern blocks.
    Expected file pattern is *_Patterns.txt.
    """
    files = sorted(Path(folder).glob("*_Patterns.txt"))
    if not files:
        log("No *_Patterns.txt files found")
        return
    raw_rows, parsed_rows = [], []
    for f in files:
        pattern_name = None
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw_rows.append({"source_file": f.name, "raw_line": line})
                u = line.upper()
                if u.startswith("BEGIN LAUNDERING ATTEMPT"):
                    pattern_name = line.split("-", 1)[1].strip() if "-" in line else "UNKNOWN"
                    continue
                if u.startswith("END LAUNDERING ATTEMPT"):
                    pattern_name = None
                    continue
                if pattern_name is None:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 11:
                    continue
                ts_str, fb, fa, tb, ta, ar, rccy, ap, pccy, ch, isl = parts[:11]
                ts = pd.to_datetime(ts_str, errors="coerce")
                if pd.isna(ts): 
                    continue
                def _num(x):
                    try:
                        return float(str(x).replace(",", ""))
                    except Exception:
                        return np.nan
                parsed_rows.append({
                    "pattern_name": pattern_name,
                    "txn_ts": ts,
                    "from_bank_id": fb,
                    "from_account": fa,
                    "to_bank_id": tb,
                    "to_account": ta,
                    "amount_received": _num(ar),
                    "recv_currency": rccy,
                    "amount_paid": _num(ap),
                    "pay_currency": pccy,
                    "payment_format": ch,
                    "is_laundering": int(str(isl).strip()=="1"),
                    "source_file": f.name
                })
    if raw_rows:
        DATA["patterns_raw"] = pd.concat([DATA["patterns_raw"], pd.DataFrame(raw_rows)], ignore_index=True)
    if parsed_rows:
        DATA["patterns_parsed"] = pd.concat([DATA["patterns_parsed"], pd.DataFrame(parsed_rows)], ignore_index=True)
    log(f"Patterns loaded: raw={len(DATA['patterns_raw'])}, parsed={len(DATA['patterns_parsed'])}")

def build_ledger():
    """
    Create a double entry style ledger with inflow and outflow legs from transactions.
    """
    if DATA["transactions"].empty:
        log("No transactions for ledger")
        return
    df = DATA["transactions"]
    outflows = df.copy()
    outflows["direction"] = "outflow"
    outflows["bank_id"] = outflows["from_bank_id"]
    outflows["account_number"] = outflows["from_account"]
    outflows["amount_gbp"] = outflows["amount_paid"]
    outflows["counterparty_bank_id"] = outflows["to_bank_id"]
    outflows["channel"] = outflows["payment_format"]
    inflows = df.copy()
    inflows["direction"] = "inflow"
    inflows["bank_id"] = inflows["to_bank_id"]
    inflows["account_number"] = inflows["to_account"]
    inflows["amount_gbp"] = inflows["amount_received"]
    inflows["counterparty_bank_id"] = inflows["from_bank_id"]
    inflows["channel"] = inflows["payment_format"]
    keep = ["txn_ts","direction","bank_id","account_number","amount_gbp","channel","counterparty_bank_id","source_txn_key"]
    DATA["ledger"] = pd.concat([outflows[keep], inflows[keep]], ignore_index=True)
    log(f"Ledger built: {len(DATA['ledger'])} legs")

def tag_transactions_with_patterns():
    """
    Tag transactions that match parsed patterns on time, endpoints, and near equal amounts.
    """
    if DATA["patterns_parsed"].empty or DATA["transactions"].empty:
        log("Skip pattern tagging (no patterns or transactions)")
        return
    t = DATA["transactions"].copy()
    p = DATA["patterns_parsed"].copy()
    t["ts_min"] = t["txn_ts"].dt.floor("min")
    p["ts_min"] = p["txn_ts"].dt.floor("min")
    j = t.merge(
        p,
        left_on=["ts_min","from_bank_id","from_account","to_bank_id","to_account"],
        right_on=["ts_min","from_bank_id","from_account","to_bank_id","to_account"],
        suffixes=("_t","_p")
    )
    j = j[
        (np.abs(j["amount_received_t"] - j["amount_received_p"]) <= 0.01) |
        (np.abs(j["amount_paid_t"]     - j["amount_paid_p"])     <= 0.01)
    ]
    DATA["pattern_tags"] = j[["source_txn_key","pattern_name"]].drop_duplicates()
    log(f"Pattern tags: {len(DATA['pattern_tags'])} rows")

def simulate_partner_statements(sample_ratio=0.97):
    """
    Create a simulated partner statement by sampling ledger rows, used for reconciliation testing.
    """
    if DATA["ledger"].empty:
        log("No ledger to simulate partner statements")
        return
    part = DATA["ledger"].sample(frac=sample_ratio, random_state=42).copy()
    part["stmt_ts"] = part["txn_ts"]
    part["partner_ref"] = "SIM-" + part["source_txn_key"].str[-8:]
    part["source_stmt_key"] = (
        part["stmt_ts"].astype(str) + "|" +
        part["bank_id"] + "|" +
        part["account_number"].fillna("") + "|" +
        part["amount_gbp"].round(2).astype(str) + "|" +
        part["direction"]
    )
    DATA["partner_statements"] = part
    log(f"Partner statements simulated: {len(part)}")

def reconcile_transactions():
    """
    Reconcile internal ledger rows to partner statements on time bucket, account, direction, and amount.
    """
    if DATA["ledger"].empty or DATA["partner_statements"].empty:
        log("Cannot reconcile: missing ledger or partner data")
        return
    l = DATA["ledger"][["txn_ts","bank_id","account_number","direction","amount_gbp","source_txn_key"]].copy()
    l["ts_min"] = l["txn_ts"].dt.floor("min")
    p = DATA["partner_statements"][["stmt_ts","bank_id","account_number","direction","amount_gbp","source_stmt_key"]].copy()
    p["ts_min"] = p["stmt_ts"].dt.floor("min")
    m = l.merge(
        p,
        on=["ts_min","bank_id","account_number","direction"],
        suffixes=("", "_p")
    )
    m = m[np.abs(m["amount_gbp"] - m["amount_gbp_p"]) < 0.01]
    DATA["recon_matches"] = m[["source_txn_key","source_stmt_key"]].drop_duplicates()
    unmatched_ledger = l[~l["source_txn_key"].isin(DATA["recon_matches"]["source_txn_key"])].copy()
    unmatched_ledger["side"] = "ledger"
    unmatched_ledger["key"] = unmatched_ledger["source_txn_key"]
    unmatched_ledger["status"] = "open"
    unmatched_ledger["created_at"] = datetime.now()
    unmatched_partner = p[~p["source_stmt_key"].isin(DATA["recon_matches"]["source_stmt_key"])].copy()
    unmatched_partner["side"] = "partner"
    unmatched_partner["key"] = unmatched_partner["source_stmt_key"]
    unmatched_partner["status"] = "open"
    unmatched_partner["created_at"] = datetime.now()
    keep_cols = ["side","key","bank_id","account_number","direction","amount_gbp","txn_ts","status","created_at"]
    DATA["recon_exceptions"] = pd.concat([
        unmatched_ledger.rename(columns={"txn_ts":"txn_ts"})[keep_cols],
        unmatched_partner.rename(columns={"stmt_ts":"txn_ts"})[keep_cols]
    ], ignore_index=True)
    log(f"Reconciliation: {len(DATA['recon_matches'])} matched, {len(DATA['recon_exceptions'])} exceptions")

def build_hourly_health_view():
    """
    Aggregate transactions by hour, channel, direction, outcome, and label counts and amounts.
    """
    if DATA["transactions"].empty:
        return
    df = DATA["transactions"].copy()
    df["hour_ts"] = df["txn_ts"].dt.floor("H")
    df["direction"] = np.where(df["amount_received"]>0, "inflow",
                        np.where(df["amount_paid"]>0, "outflow","unknown"))
    df["outcome"] = np.where((df["amount_received"]>0)|(df["amount_paid"]>0), "success", "failed")
    VIEWS["txn_hourly_health"] = df.groupby(
        ["hour_ts","payment_format","direction","outcome","is_laundering"]
    ).agg(
        txn_cnt=("source_txn_key","count"),
        recv_amount=("amount_received","sum"),
        paid_amount=("amount_paid","sum")
    ).reset_index()
    log("Built: txn_hourly_health")

def build_failure_reasons_view():
    """
    Flag simple failure reasons and count them by hour and channel.
    """
    if DATA["transactions"].empty:
        return
    df = DATA["transactions"].copy()
    df["hour_ts"] = df["txn_ts"].dt.floor("H")
    df["fail_reason"] = "other"
    df.loc[(df["amount_paid"]==0) & (df["amount_received"]>0), "fail_reason"] = "missing_outflow"
    df.loc[(df["amount_received"]==0) & (df["amount_paid"]>0), "fail_reason"] = "missing_inflow"
    df.loc[np.abs(df["amount_paid"]-df["amount_received"])>0.01, "fail_reason"] = "amount_mismatch"
    failures = df[df["fail_reason"]!="other"]
    VIEWS["failure_reasons"] = failures.groupby(
        ["hour_ts","payment_format","is_laundering","fail_reason"]
    ).size().reset_index(name="fail_cnt")
    log("Built: failure_reasons")

def build_recon_views():
    """
    Build open break summary and aging buckets for reconciliation exceptions.
    """
    if DATA["recon_exceptions"].empty:
        return
    ex = DATA["recon_exceptions"]
    VIEWS["recon_open"] = ex[ex["status"]=="open"].groupby(
        ["side","bank_id","direction"]
    ).agg(
        open_cnt=("key","count"),
        open_amount=("amount_gbp","sum"),
        oldest_ts=("txn_ts","min")
    ).reset_index()
    now = pd.Timestamp.now()
    df = ex[ex["status"]=="open"].copy()
    df["age_hours"] = (now - df["txn_ts"]).dt.total_seconds()/3600
    df["age_bucket"] = pd.cut(
        df["age_hours"], bins=[0,24,72,168,10**9], labels=["d1","d3","d7","d7p"]
    )
    VIEWS["recon_aging"] = df.groupby(["side","bank_id","direction","age_bucket"]).size()\
                             .unstack(fill_value=0).reset_index()
    log("Built: recon views")

def build_pattern_hourly_health():
    """
    Summarize hourly health split by pattern tagged versus untagged transactions.
    """
    if DATA["transactions"].empty:
        return
    t = DATA["transactions"].copy()
    tags = DATA["pattern_tags"].copy()
    t["hour_ts"] = t["txn_ts"].dt.floor("H")
    tagged = set(tags["source_txn_key"]) if not tags.empty else set()
    t["tag_class"] = np.where(t["source_txn_key"].isin(tagged), "pattern_tagged", "untagged")
    t["outcome"] = np.where((t["amount_received"]>0)|(t["amount_paid"]>0), "success", "failed")
    VIEWS["pattern_hourly_health"] = t.groupby(
        ["hour_ts","payment_format","tag_class","outcome"]
    ).size().reset_index(name="txn_cnt")
    log("Built: pattern_hourly_health")

def detect_3sigma_anomalies():
    """
    Detect three sigma anomalies on hourly success rate by channel, and register alerts.
    """
    if VIEWS["txn_hourly_health"].empty:
        return
    df = VIEWS["txn_hourly_health"].copy()
    total = df.groupby(["hour_ts","payment_format"])["txn_cnt"].sum().rename("t")
    succ  = df[df["outcome"]=="success"].groupby(["hour_ts","payment_format"])["txn_cnt"].sum().rename("s")
    summary = pd.merge(total, succ, left_index=True, right_index=True, how="left").fillna(0).reset_index()
    summary["success_rate"] = summary["s"]/summary["t"]
    alerts_before = len(DATA["alerts"])
    for ch, g in summary.groupby("payment_format"):
        g = g.sort_values("hour_ts").copy()
        g["ma"] = g["success_rate"].rolling(24, min_periods=6).mean()
        g["sd"] = g["success_rate"].rolling(24, min_periods=6).std()
        g["z"]  = (g["success_rate"] - g["ma"]) / g["sd"]
        anom = g[(g["sd"].notna()) & (np.abs(g["z"])>3)]
        for _, r in anom.iterrows():
            DATA["alerts"].append({
                "type": "3SIGMA_SUCCESS_RATE",
                "severity": "HIGH",
                "channel": ch,
                "timestamp": r["hour_ts"],
                "success_rate": float(r["success_rate"]),
                "z_score": float(r["z"]),
                "message": f"3σ anomaly for {ch}: success_rate={r['success_rate']:.2%}, z={r['z']:.2f}"
            })
    log(f"3-sigma anomalies: +{len(DATA['alerts'])-alerts_before}")

def check_break_aging(threshold_hours=24):
    """
    Raise alerts for open breaks that exceed the given age threshold in hours.
    """
    if DATA["recon_exceptions"].empty:
        return
    now = pd.Timestamp.now()
    df = DATA["recon_exceptions"][DATA["recon_exceptions"]["status"]=="open"].copy()
    df["hours_open"] = (now - df["txn_ts"]).dt.total_seconds()/3600
    crit = df[df["hours_open"]>threshold_hours]
    before = len(DATA["alerts"])
    for _, r in crit.iterrows():
        DATA["alerts"].append({
            "type": "AGING_BREAK",
            "severity": "CRITICAL",
            "timestamp": r["txn_ts"],
            "bank_id": r["bank_id"],
            "amount": float(r["amount_gbp"]),
            "message": f"Aging break >{threshold_hours}h on {r['side']} side"
        })
    log(f"Aging breaks: +{len(DATA['alerts'])-before}")

def analyze_failure_spikes(min_failures=10):
    """
    Identify hours and channels with failure counts above a given threshold, and raise alerts.
    """
    if VIEWS["failure_reasons"].empty:
        return
    summary = VIEWS["failure_reasons"].groupby(["hour_ts","payment_format"])["fail_cnt"].sum().reset_index()
    spikes = summary[summary["fail_cnt"]>min_failures]
    before = len(DATA["alerts"])
    for _, r in spikes.iterrows():
        DATA["alerts"].append({
            "type": "FAILURE_SPIKE",
            "severity": "HIGH",
            "timestamp": r["hour_ts"],
            "channel": r["payment_format"],
            "failure_count": int(r["fail_cnt"]),
            "message": f"Failure spike on {r['payment_format']}: {int(r['fail_cnt'])} failures"
        })
    log(f"Failure spikes: +{len(DATA['alerts'])-before}")

def aml_profile_summary() -> pd.DataFrame:
    """
    Return a small summary of rows and laundering flags for the loaded transactions.
    """
    if DATA["transactions"].empty:
        return pd.DataFrame()
    df = DATA["transactions"].copy()
    out = pd.DataFrame({
        "n_rows":[len(df)],
        "n_laundering":[int((df["is_laundering"]==1).sum())],
        "pct_laundering":[float((df["is_laundering"]==1).mean()*100.0)]
    })
    return out

def chronological_split(output_dir: Path):
    """
    Create a simple chronological split of transactions into train, valid, and test sets.
    """
    if DATA["transactions"].empty: 
        return
    df = DATA["transactions"][["source_txn_key","txn_ts","source_file"]].copy().sort_values("txn_ts")
    n = len(df); n60=int(n*0.6); n80=int(n*0.8)
    df["split"] = ["train"]*n60 + ["valid"]*(n80-n60) + ["test"]*(n-n80)
    (output_dir/"splits").mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir/"splits"/"splits_all.csv", index=False)
    log("Wrote splits/splits_all.csv")

def export_all_csvs(output_dir: Path):
    """
    Export all non empty analytical views and summary tables to CSV files in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "txn_hourly_health.csv": VIEWS["txn_hourly_health"],
        "failure_reasons.csv": VIEWS["failure_reasons"],
        "recon_open.csv": VIEWS["recon_open"],
        "recon_aging.csv": VIEWS["recon_aging"],
        "pattern_hourly_health.csv": VIEWS["pattern_hourly_health"],
        "alerts.csv": pd.DataFrame(DATA["alerts"]),
        "aml_profile_summary.csv": aml_profile_summary(),
    }
    for name, df in exports.items():
        if df is not None and not df.empty:
            df.to_csv(output_dir/name, index=False)
            log(f"Exported {name}")

def generate_oncall_csv(output_path: Path):
    """
    Write a compact on call summary CSV with transaction counts, breaks, and success rate.
    """
    now = datetime.now()
    open_breaks = 0
    crit_breaks = 0
    if not DATA["recon_exceptions"].empty:
        ex = DATA["recon_exceptions"]
        open_breaks = int((ex["status"]=="open").sum())
        crit_breaks = int(((pd.Timestamp.now() - ex["txn_ts"]).dt.total_seconds()/3600 > 24).sum())
    success_rate = None
    if not VIEWS["txn_hourly_health"].empty:
        tot = VIEWS["txn_hourly_health"]["txn_cnt"].sum()
        succ = VIEWS["txn_hourly_health"][VIEWS["txn_hourly_health"]["outcome"]=="success"]["txn_cnt"].sum()
        success_rate = float(succ / tot) if tot else 0.0
    row = {
        "report_time": now,
        "total_transactions": len(DATA["transactions"]),
        "open_breaks": open_breaks,
        "critical_breaks_24h": crit_breaks,
        "success_rate": success_rate,
        "active_alerts": len(DATA["alerts"]),
        "critical_alerts": sum(1 for a in DATA["alerts"] if a.get("severity")=="CRITICAL"),
    }
    pd.DataFrame([row]).to_csv(output_path, index=False)
    log(f"Exported {output_path.name}")

def print_summary():
    """
    Print a short executive summary of core counts and alert types.
    """
    print("\n" + "="*64)
    print("EXECUTIVE SUMMARY")
    print("="*64)
    print(f"Transactions:       {len(DATA['transactions']):,}")
    print(f"Ledger legs:        {len(DATA['ledger']):,}")
    print(f"Partner statements: {len(DATA['partner_statements']):,}")
    if not DATA["recon_matches"].empty or not DATA["recon_exceptions"].empty:
        match_rate = len(DATA["recon_matches"]) / max(len(DATA["ledger"]), 1) * 100.0
        print(f"Recon match rate:   {match_rate:.2f}%")
        print(f"Open breaks:        {int((DATA['recon_exceptions']['status']=='open').sum()):,}")
    if not VIEWS["txn_hourly_health"].empty:
        tot = VIEWS["txn_hourly_health"]["txn_cnt"].sum()
        succ = VIEWS["txn_hourly_health"][VIEWS["txn_hourly_health"]["outcome"]=="success"]["txn_cnt"].sum()
        print(f"Overall success:    {succ/tot*100:.2f}%")
    if DATA["alerts"]:
        by = pd.DataFrame(DATA["alerts"]).groupby("type").size().to_dict()
        print("Alerts by type:", by)
    if not DATA["pattern_tags"].empty:
        print(f"Pattern-tagged tx:  {len(DATA['pattern_tags']):,} (patterns: {DATA['pattern_tags']['pattern_name'].nunique()})")
    print("="*64 + "\n")

def run_full_pipeline(data_folder: str, output_dir: str = "./output", simulate_partner=True):
    """
    Run the full pipeline, from loading to exports and a final console summary.
    """
    data_folder = Path(data_folder)
    output_dir = Path(output_dir)
    log("=== Phase 1: Load ===")
    load_all_accounts(data_folder)
    load_all_transactions(data_folder)
    load_all_patterns(data_folder)
    if DATA["transactions"].empty:
        log("No transactions found. Stop.")
        return
    log("=== Phase 2: Transform ===")
    tag_transactions_with_patterns()
    build_ledger()
    log("=== Phase 3: Partner & Recon ===")
    if simulate_partner:
        simulate_partner_statements()
    reconcile_transactions()
    log("=== Phase 4: Views ===")
    build_hourly_health_view()
    build_failure_reasons_view()
    build_recon_views()
    build_pattern_hourly_health()
    log("=== Phase 5: Alerts ===")
    detect_3sigma_anomalies()
    check_break_aging(threshold_hours=24)
    analyze_failure_spikes(min_failures=10)
    log("=== Phase 6: Exports ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    export_all_csvs(output_dir)
    chronological_split(output_dir)
    generate_oncall_csv(output_dir/"oncall_report.csv")
    print_summary()
    log("Done.")

def push_outputs_to_drive(local_output_dir: str | Path,
                          drive_processed_dir: str | Path = r"G:\My Drive\FinTech\data\processed",
                          filenames: list[str] | None = None) -> None:
    """
    Copy selected CSVs from a local output folder to a Google Drive folder for downstream reporting.
    """
    local_output_dir = Path(local_output_dir)
    drive_processed_dir = Path(drive_processed_dir)
    drive_processed_dir.mkdir(parents=True, exist_ok=True)
    if filenames:
        sources = [local_output_dir / name for name in filenames]
    else:
        sources = sorted(local_output_dir.glob("*.csv"))
    copied = 0
    for src in sources:
        if not src.exists():
            print(f"[skip] Not found: {src}")
            continue
        dst = drive_processed_dir / src.name
        shutil.copy2(src, dst)
        copied += 1
        print(f"[copied] {src.name} -> {dst}")
    print(f"[done] Copied {copied} file(s) to {drive_processed_dir}")

if __name__ == "__main__":
    run_full_pipeline(data_folder="./data/raw/aml", output_dir="./output")
    csvs_to_publish = [
        "txn_hourly_health.csv",
        "failure_reasons.csv",
        "recon_open.csv",
        "recon_aging.csv",
        "pattern_hourly_health.csv",
        "aml_profile_summary.csv",
        "alerts.csv",
        "oncall_report.csv",
    ]
    LOCAL_OUTPUT = "./output"
    push_outputs_to_drive(
        local_output_dir=LOCAL_OUTPUT,
        drive_processed_dir=r"G:\My Drive\FinTech\data\processed",
        filenames=csvs_to_publish
    )
