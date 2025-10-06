
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_FOLDER = Path("./data/raw/aml")
DB_PATH = Path("./ledgerops.db")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    """
    Print a timestamped message for simple runtime logging.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_conn():
    """
    Open a SQLite connection with pragmatic performance settings.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn

def init_schema():
    """
    Create core tables for banks, entities, accounts, transactions, and patterns if they do not exist.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS banks(
      bank_id   TEXT PRIMARY KEY,
      bank_name TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entities(
      entity_id   TEXT PRIMARY KEY,
      entity_name TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts(
      account_number TEXT PRIMARY KEY,
      bank_id        TEXT,
      entity_id      TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions(
      txn_ts TEXT NOT NULL,
      hour_ts TEXT,
      from_bank_id TEXT,
      from_account TEXT,
      to_bank_id   TEXT,
      to_account   TEXT,
      amount_received REAL,
      recv_currency TEXT,
      amount_paid REAL,
      pay_currency TEXT,
      payment_format TEXT,
      is_laundering INTEGER,
      source_file TEXT,
      source_txn_key TEXT UNIQUE
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_hour ON transactions(hour_ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions(from_account)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_to   ON transactions(to_account)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_key  ON transactions(source_txn_key)")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patterns_raw(
      source_file TEXT,
      raw_line    TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patterns_parsed(
      pattern_name   TEXT,
      txn_ts         TEXT,
      from_bank_id   TEXT,
      from_account   TEXT,
      to_bank_id     TEXT,
      to_account     TEXT,
      amount_received REAL,
      recv_currency   TEXT,
      amount_paid     REAL,
      pay_currency    TEXT,
      payment_format  TEXT,
      is_laundering   INTEGER,
      source_file     TEXT
    )
    """)

    conn.commit()
    conn.close()
    log("SQLite schema ready.")

def load_accounts(folder: Path):
    """
    Load bank, entity, and account records from CSV files, then upsert into SQLite.
    Files should match the pattern *_accounts.csv.
    """
    files = sorted(folder.glob("*_accounts.csv"))
    if not files:
        log("No *_accounts.csv files found.")
        return

    banks_all, ents_all, accts_all = [], [], []

    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()

        if {"Bank ID","Bank Name"}.issubset(df.columns):
            b = df[["Bank ID","Bank Name"]].drop_duplicates()
            b.columns = ["bank_id","bank_name"]
            b["bank_id"] = b["bank_id"].astype(str).str.strip()
            b["bank_name"] = b["bank_name"].astype(str).str.strip()
            banks_all.append(b)

        if {"Entity ID","Entity Name"}.issubset(df.columns):
            e = df[["Entity ID","Entity Name"]].drop_duplicates()
            e.columns = ["entity_id","entity_name"]
            e["entity_id"] = e["entity_id"].astype(str).str.strip()
            e["entity_name"] = e["entity_name"].astype(str).str.strip()
            ents_all.append(e)

        need = {"Account Number","Bank ID","Entity ID"}
        if not need.issubset(df.columns):
            raise ValueError(f"Missing account columns: {need - set(df.columns)} in {f.name}")
        a = df[["Account Number","Bank ID","Entity ID"]].drop_duplicates()
        a.columns = ["account_number","bank_id","entity_id"]
        for c in a.columns:
            a[c] = a[c].astype(str).str.strip()
        accts_all.append(a)

    conn = get_conn()
    if banks_all:
        pd.concat(banks_all, ignore_index=True).drop_duplicates().to_sql("_banks_stage", conn, "append", index=False)
        conn.execute("INSERT OR IGNORE INTO banks(bank_id, bank_name) SELECT bank_id, bank_name FROM _banks_stage")
        conn.execute("DROP TABLE IF EXISTS _banks_stage")
    if ents_all:
        pd.concat(ents_all, ignore_index=True).drop_duplicates().to_sql("_ents_stage", conn, "append", index=False)
        conn.execute("INSERT OR IGNORE INTO entities(entity_id, entity_name) SELECT entity_id, entity_name FROM _ents_stage")
        conn.execute("DROP TABLE IF EXISTS _ents_stage")
    if accts_all:
        pd.concat(accts_all, ignore_index=True).drop_duplicates().to_sql("_accts_stage", conn, "append", index=False)
        conn.execute("INSERT OR IGNORE INTO accounts(account_number, bank_id, entity_id) SELECT account_number, bank_id, entity_id FROM _accts_stage")
        conn.execute("DROP TABLE IF EXISTS _accts_stage")

    conn.commit()
    conn.close()
    log("Accounts loaded, duplicates ignored.")

def _clean_tx_chunk(df: pd.DataFrame, src_name: str) -> pd.DataFrame:
    """
    Normalize a single chunk of transaction rows to a standard schema.
    Returns a dataframe with consistent column names and a stable row key.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    from_acc_col = "From Bank Account" if "From Bank Account" in df.columns else ("Account" if "Account" in df.columns else None)
    to_acc_col   = "To Bank Account"   if "To Bank Account"   in df.columns else ("Account.1" if "Account.1" in df.columns else None)
    if not from_acc_col or not to_acc_col:
        cols = list(df.columns)
        try:
            from_acc_col = cols[cols.index("From Bank") + 1]
            to_acc_col   = cols[cols.index("To Bank") + 1]
        except Exception:
            raise ValueError(f"Cannot infer account columns from headers: {list(df.columns)}")

    rename = {
        "Timestamp":"txn_ts",
        "From Bank":"from_bank_id",
        from_acc_col:"from_account",
        "To Bank":"to_bank_id",
        to_acc_col:"to_account",
        "Amount Received":"amount_received",
        "Receiving Currency":"recv_currency",
        "Amount Paid":"amount_paid",
        "Payment Currency":"pay_currency",
        "Payment Format":"payment_format",
        "Is Laundering":"is_laundering",
    }
    keep = [c for c in rename if c in df.columns]
    out = df[keep].rename(columns=rename)

    out["txn_ts"] = pd.to_datetime(out["txn_ts"], errors="coerce")
    for c in ["amount_received","amount_paid"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["is_laundering"] = pd.to_numeric(out.get("is_laundering", 0), errors="coerce").fillna(0).astype(int)

    for c in ["from_bank_id","from_account","to_bank_id","to_account","recv_currency","pay_currency","payment_format"]:
        if c in out:
            out[c] = out[c].astype(str).str.strip()

    out = out[out["txn_ts"].notna() & ((out["amount_received"]>0) | (out["amount_paid"]>0))]

    out["source_file"] = src_name
    out["hour_ts"] = out["txn_ts"].dt.floor("H")
    out["source_txn_key"] = (
        out["txn_ts"].astype(str) + "|" +
        out["from_bank_id"].fillna("") + "|" + out["from_account"].fillna("") + "|" +
        out["to_bank_id"].fillna("") + "|" + out["to_account"].fillna("") + "|" +
        out["amount_received"].round(2).astype(str) + "|" +
        out["amount_paid"].round(2).astype(str)
    )

    out["txn_ts"]  = out["txn_ts"].astype("datetime64[ns]")
    out["hour_ts"] = out["hour_ts"].astype("datetime64[ns]")
    return out

def load_transactions_chunked(folder: Path, chunksize: int = 200_000):
    """
    Stream large transaction files in chunks, clean them, and upsert into SQLite.
    Files should match the pattern *_Trans.csv.
    """
    files = sorted(folder.glob("*_Trans.csv"))
    if not files:
        log("No *_Trans.csv files found.")
        return

    conn = get_conn()
    for f in files:
        log(f"Loading {f.name} ...")
        reader = pd.read_csv(f, chunksize=chunksize)
        for i, chunk in enumerate(reader, start=1):
            clean = _clean_tx_chunk(chunk, f.name)
            clean.to_sql("_tx_stage", conn, if_exists="replace", index=False)
            conn.execute("""
                INSERT OR IGNORE INTO transactions
                (txn_ts, hour_ts, from_bank_id, from_account, to_bank_id, to_account,
                 amount_received, recv_currency, amount_paid, pay_currency, payment_format,
                 is_laundering, source_file, source_txn_key)
                SELECT txn_ts, hour_ts, from_bank_id, from_account, to_bank_id, to_account,
                       amount_received, recv_currency, amount_paid, pay_currency, payment_format,
                       is_laundering, source_file, source_txn_key
                FROM _tx_stage
            """)
            conn.execute("DROP TABLE IF EXISTS _tx_stage")
            if i % 10 == 0:
                conn.commit()
                log(f"{f.name}: chunk {i} committed")
        conn.commit()
        log(f"{f.name}: done")
    conn.close()
    log("Transactions loaded, duplicates ignored.")

def load_patterns(folder: Path):
    """
    Read pattern files, store raw lines, and parse transaction like rows within named blocks.
    Files should match the pattern *_Patterns.txt.
    """
    files = sorted(folder.glob("*_Patterns.txt"))
    if not files:
        log("No *_Patterns.txt files found.")
        return

    raw_rows, parsed_rows = [], []
    for f in files:
        patt = None
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                raw_rows.append({"source_file": f.name, "raw_line": s})

                u = s.upper()
                if u.startswith("BEGIN LAUNDERING ATTEMPT"):
                    patt = s.split("-", 1)[1].strip() if "-" in s else "UNKNOWN"
                    continue
                if u.startswith("END LAUNDERING ATTEMPT"):
                    patt = None
                    continue
                if patt is None:
                    continue

                parts = [p.strip() for p in s.split(",")]
                if len(parts) < 11:
                    continue

                ts_str, fb, fa, tb, ta, ar, rccy, ap, pccy, ch, isl = parts[:11]
                ts = pd.to_datetime(ts_str, errors="coerce")
                if pd.isna(ts):
                    continue

                def _num(x):
                    try:
                        return float(str(x).replace(",", ""))
                    except:
                        return np.nan

                parsed_rows.append({
                    "pattern_name": patt,
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
                    "is_laundering": int(str(isl).strip() == "1"),
                    "source_file": f.name
                })

    conn = get_conn()
    if raw_rows:
        pd.DataFrame(raw_rows).to_sql("patterns_raw", conn, if_exists="append", index=False)
    if parsed_rows:
        pd.DataFrame(parsed_rows).to_sql("patterns_parsed", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    log(f"Patterns loaded: raw={len(raw_rows)}, parsed={len(parsed_rows)}")

def quick_eda():
    """
    Print a few row counts and lightweight summaries from SQLite.
    """
    conn = get_conn()
    cur = conn.cursor()

    n_tx = cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    n_ac = cur.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
    n_pat = cur.execute("SELECT COUNT(*) FROM patterns_parsed").fetchone()[0]
    log(f"Rows -> transactions: {n_tx:,} | accounts: {n_ac:,} | patterns_parsed: {n_pat:,}")

    df_ch = pd.read_sql_query("""
        SELECT payment_format AS channel, COUNT(*) AS txn_cnt
        FROM transactions
        GROUP BY payment_format
        ORDER BY txn_cnt DESC
        LIMIT 10
    """, conn)
    if not df_ch.empty:
        print("\nTop channels:")
        print(df_ch.to_string(index=False))

    df_hour = pd.read_sql_query("""
        SELECT substr(hour_ts,1,13) AS hour, COUNT(*) AS txn_cnt
        FROM transactions
        GROUP BY substr(hour_ts,1,13)
        ORDER BY hour
        LIMIT 24
    """, conn)
    if not df_hour.empty:
        print("\nFirst 24 hourly totals:")
        print(df_hour.to_string(index=False))

    conn.close()

def exec1(conn, sql, params=None):
    """
    Execute a single SQL statement with optional parameters.
    """
    conn.execute(sql) if params is None else conn.execute(sql, params)

def ensure_analysis_schema():
    """
    Create analysis tables for ledger, partner statements, reconciliation, and hourly views.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ledger(
      txn_ts TEXT NOT NULL,
      direction TEXT,
      bank_id TEXT,
      account_number TEXT,
      amount_gbp REAL,
      channel TEXT,
      counterparty_bank_id TEXT,
      source_txn_key TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ledger_ts ON ledger(txn_ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ledger_min ON ledger(txn_ts, bank_id, account_number, direction)")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS partner_statements(
      stmt_ts TEXT NOT NULL,
      bank_id TEXT NOT NULL,
      account_number TEXT,
      amount_gbp REAL,
      direction TEXT,
      partner_ref TEXT,
      source_stmt_key TEXT UNIQUE
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_partner_ts ON partner_statements(stmt_ts, bank_id, account_number, direction)")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS recon_matches(
      source_txn_key  TEXT,
      source_stmt_key TEXT,
      matched_on      TEXT,
      matched_at      TEXT,
      PRIMARY KEY (source_txn_key, source_stmt_key)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS recon_exceptions(
      side TEXT,
      key  TEXT,
      bank_id TEXT,
      account_number TEXT,
      direction TEXT,
      amount_gbp REAL,
      txn_ts TEXT,
      status TEXT DEFAULT 'open',
      created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS mv_txn_hourly_health(
      hour_ts TEXT,
      channel TEXT,
      direction TEXT,
      outcome TEXT,
      is_laundering INTEGER,
      txn_cnt INTEGER,
      recv_amount REAL,
      paid_amount REAL,
      PRIMARY KEY (hour_ts, channel, direction, outcome, is_laundering)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS mv_failure_reasons(
      hour_ts TEXT,
      channel TEXT,
      is_laundering INTEGER,
      fail_reason TEXT,
      fail_cnt INTEGER,
      PRIMARY KEY (hour_ts, channel, is_laundering, fail_reason)
    )
    """)

    conn.commit()
    conn.close()
    log("Analysis schema ready.")

def build_ledger():
    """
    Populate a double entry ledger view with outflow and inflow legs from transactions.
    Uses a bulk insert approach for speed.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("BEGIN IMMEDIATE;")

    cur.execute("DROP INDEX IF EXISTS idx_ledger_ts;")
    cur.execute("DROP INDEX IF NOT EXISTS idx_ledger_min;")

    cur.execute("DELETE FROM ledger;")

    cur.execute("""
        INSERT INTO ledger
          (txn_ts, direction, bank_id, account_number, amount_gbp, channel, counterparty_bank_id, source_txn_key)
        SELECT
          txn_ts,
          'outflow',
          from_bank_id,
          from_account,
          COALESCE(amount_paid, 0.0),
          payment_format,
          to_bank_id,
          source_txn_key
        FROM transactions
        WHERE COALESCE(amount_paid, 0.0) > 0.0;
    """)

    cur.execute("""
        INSERT INTO ledger
          (txn_ts, direction, bank_id, account_number, amount_gbp, channel, counterparty_bank_id, source_txn_key)
        SELECT
          txn_ts,
          'inflow',
          to_bank_id,
          to_account,
          COALESCE(amount_received, 0.0),
          payment_format,
          from_bank_id,
          source_txn_key
        FROM transactions
        WHERE COALESCE(amount_received, 0.0) > 0.0;
    """)

    cur.execute("COMMIT;")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_ledger_ts  ON ledger(txn_ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ledger_min ON ledger(txn_ts, bank_id, account_number, direction);")

    cur.execute("PRAGMA synchronous=NORMAL;")

    conn.commit()
    conn.close()
    log("Ledger built.")

def simulate_partner_statements(sample_ratio: float = 0.97):
    """
    Generate partner statements by sampling the internal ledger, for reconciliation testing.
    """
    conn = get_conn()
    exec1(conn, "DELETE FROM partner_statements")
    exec1(conn, f"""
        INSERT INTO partner_statements
          (stmt_ts, bank_id, account_number, amount_gbp, direction, partner_ref, source_stmt_key)
        SELECT
          txn_ts,
          bank_id,
          account_number,
          amount_gbp,
          direction,
          'SIM-' || substr(source_txn_key, -8),
          CAST(txn_ts AS TEXT) || '|' ||
          IFNULL(bank_id,'') || '|' ||
          IFNULL(account_number,'') || '|' ||
          printf('%.2f', IFNULL(amount_gbp,0)) || '|' ||
          IFNULL(direction,'') || '|' ||
          substr(source_txn_key, -16)
        FROM ledger
        WHERE (ABS(random()) % 100) < {int(round(sample_ratio*100))}
    """)
    conn.commit()
    conn.close()
    log("Partner statements simulated.")

def reconcile():
    """
    Match ledger legs to partner statements on minute bucket, attributes, and amount, then record exceptions.
    """
    conn = get_conn()
    exec1(conn, "DELETE FROM recon_matches")
    exec1(conn, "DELETE FROM recon_exceptions")

    exec1(conn, """
        INSERT OR IGNORE INTO recon_matches(source_txn_key, source_stmt_key, matched_on, matched_at)
        SELECT
          l.source_txn_key,
          p.source_stmt_key,
          'minute_exact',
          datetime('now')
        FROM ledger l
        JOIN partner_statements p
          ON substr(l.txn_ts, 1, 16) = substr(p.stmt_ts, 1, 16)
         AND IFNULL(l.bank_id,'')        = IFNULL(p.bank_id,'')
         AND IFNULL(l.account_number,'') = IFNULL(p.account_number,'')
         AND IFNULL(l.direction,'')      = IFNULL(p.direction,'')
         AND ABS(IFNULL(l.amount_gbp,0.0) - IFNULL(p.amount_gbp,0.0)) < 0.01
    """)

    exec1(conn, """
        INSERT OR IGNORE INTO recon_exceptions
          (side, key, bank_id, account_number, direction, amount_gbp, txn_ts, status, created_at)
        SELECT
          'ledger',
          l.source_txn_key,
          l.bank_id,
          l.account_number,
          l.direction,
          l.amount_gbp,
          l.txn_ts,
          'open',
          datetime('now')
        FROM ledger l
        LEFT JOIN recon_matches m ON m.source_txn_key = l.source_txn_key
        WHERE m.source_txn_key IS NULL
    """)

    exec1(conn, """
        INSERT OR IGNORE INTO recon_exceptions
          (side, key, bank_id, account_number, direction, amount_gbp, txn_ts, status, created_at)
        SELECT
          'partner',
          p.source_stmt_key,
          p.bank_id,
          p.account_number,
          p.direction,
          p.amount_gbp,
          p.stmt_ts,
          'open',
          datetime('now')
        FROM partner_statements p
        LEFT JOIN recon_matches m ON m.source_stmt_key = p.source_stmt_key
        WHERE m.source_stmt_key IS NULL
    """)

    conn.commit()
    conn.close()
    log("Reconciliation complete.")

def rebuild_mv_txn_hourly_health():
    """
    Aggregate transactions by hour, channel, direction, outcome, and laundering flag into a view table.
    """
    conn = get_conn()
    exec1(conn, "DELETE FROM mv_txn_hourly_health")
    exec1(conn, """
    INSERT INTO mv_txn_hourly_health
      (hour_ts, channel, direction, outcome, is_laundering, txn_cnt, recv_amount, paid_amount)
    SELECT
      hour_ts,
      payment_format AS channel,
      CASE
        WHEN COALESCE(amount_received,0.0) > 0.0 THEN 'inflow'
        WHEN COALESCE(amount_paid,0.0)     > 0.0 THEN 'outflow'
        ELSE 'unknown'
      END AS direction,
      CASE WHEN (COALESCE(amount_received,0.0) > 0.0 OR COALESCE(amount_paid,0.0) > 0.0) THEN 'success' ELSE 'failed' END AS outcome,
      is_laundering,
      COUNT(*)                               AS txn_cnt,
      SUM(COALESCE(amount_received,0.0))     AS recv_amount,
      SUM(COALESCE(amount_paid,0.0))         AS paid_amount
    FROM transactions
    GROUP BY hour_ts, channel, direction, outcome, is_laundering
    """)
    conn.commit()
    conn.close()
    log("Built mv_txn_hourly_health.")

def rebuild_mv_failure_reasons():
    """
    Classify simple failure reasons and store hourly counts by channel.
    """
    conn = get_conn()
    exec1(conn, "DELETE FROM mv_failure_reasons")
    exec1(conn, """
    INSERT INTO mv_failure_reasons
      (hour_ts, channel, is_laundering, fail_reason, fail_cnt)
    SELECT
      hour_ts,
      payment_format AS channel,
      is_laundering,
      CASE
        WHEN COALESCE(amount_paid,0.0)=0.0  AND COALESCE(amount_received,0.0)>0.0 THEN 'missing_outflow'
        WHEN COALESCE(amount_received,0.0)=0.0 AND COALESCE(amount_paid,0.0)>0.0 THEN 'missing_inflow'
        WHEN ABS(COALESCE(amount_paid,0.0) - COALESCE(amount_received,0.0)) > 0.01 THEN 'amount_mismatch'
        ELSE 'other'
      END AS fail_reason,
      COUNT(*) AS fail_cnt
    FROM transactions
    WHERE (
      COALESCE(amount_paid,0.0)=0.0 OR
      COALESCE(amount_received,0.0)=0.0 OR
      ABS(COALESCE(amount_paid,0.0) - COALESCE(amount_received,0.0)) > 0.01
    )
    GROUP BY hour_ts, channel, is_laundering, fail_reason
    """)
    conn.commit()
    conn.close()
    log("Built mv_failure_reasons.")

def export_csv(sql: str, out_path: Path, parse_dates=None):
    """
    Run a query and export the result to CSV, writes nothing when the result is empty.
    """
    conn = get_conn()
    df = pd.read_sql_query(sql, conn, parse_dates=parse_dates)
    conn.close()
    if not df.empty:
        pd.DataFrame(df).to_csv(out_path, index=False)
        log(f"Exported {Path(out_path).name}")
    else:
        log(f"No rows for {Path(out_path).name}")

def export_recon_views():
    """
    Export open break summary and aging buckets to CSV files.
    """
    export_csv("""
        SELECT side, bank_id, direction,
               COUNT(*) AS open_cnt,
               SUM(amount_gbp) AS open_amount,
               MIN(txn_ts) AS oldest_ts
        FROM recon_exceptions
        WHERE status='open'
        GROUP BY side, bank_id, direction
        ORDER BY side, bank_id, direction
    """, OUTPUT_DIR/"recon_open.csv", parse_dates=["oldest_ts"])

    export_csv("""
        SELECT
          side, bank_id, direction,
          SUM(CASE WHEN (julianday('now') - julianday(txn_ts))*24 <= 24  THEN 1 ELSE 0 END) AS d1,
          SUM(CASE WHEN (julianday('now') - julianday(txn_ts))*24 > 24  AND (julianday('now') - julianday(txn_ts))*24 <= 72  THEN 1 ELSE 0 END) AS d3,
          SUM(CASE WHEN (julianday('now') - julianday(txn_ts))*24 > 72  AND (julianday('now') - julianday(txn_ts))*24 <= 168 THEN 1 ELSE 0 END) AS d7,
          SUM(CASE WHEN (julianday('now') - julianday(txn_ts))*24 > 168 THEN 1 ELSE 0 END) AS d7p
        FROM recon_exceptions
        WHERE status='open'
        GROUP BY side, bank_id, direction
        ORDER BY side, bank_id, direction
    """, OUTPUT_DIR/"recon_aging.csv")

def run_sql_analysis():
    """
    Orchestrate the SQL based pipeline end to end, then export CSV outputs.
    """
    log("=== Ensure analysis schema ===")
    ensure_analysis_schema()

    log("=== Build ledger ===")
    build_ledger()

    log("=== Partner simulate ===")
    simulate_partner_statements(sample_ratio=0.97)

    log("=== Reconcile ===")
    reconcile()

    log("=== Build MV tables ===")
    rebuild_mv_txn_hourly_health()
    rebuild_mv_failure_reasons()

    log("=== Export CSVs ===")
    export_csv("SELECT * FROM mv_txn_hourly_health ORDER BY hour_ts, channel, direction, outcome, is_laundering",
               OUTPUT_DIR/"txn_hourly_health.csv", parse_dates=["hour_ts"])
    export_csv("SELECT * FROM mv_failure_reasons ORDER BY hour_ts, channel, is_laundering, fail_reason",
               OUTPUT_DIR/"failure_reasons.csv", parse_dates=["hour_ts"])
    export_recon_views()

    log("Done.")

if __name__ == "__main__":
    init_schema()
    load_accounts(DATA_FOLDER)
    load_transactions_chunked(DATA_FOLDER, chunksize=200_000)
    load_patterns(DATA_FOLDER)
    quick_eda()
    run_sql_analysis()
