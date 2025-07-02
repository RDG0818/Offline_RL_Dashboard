import sqlite3
from utils import log
from backend.config import DashboardConfig

config = DashboardConfig()

def get_connection():
    return sqlite3.connect(config.db_path, check_same_thread=False)

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        algo TEXT NOT NULL,
        env TEXT NOT NULL,
        seed INTEGER,
        return REAL,
        log_path TEXT,
        config_path TEXT,
        timestamp TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        run_id INTEGER,
        step INTEGER,
        name TEXT,
        value REAL,
        FOREIGN KEY(run_id) REFERENCES runs(id)
    )
    """)

    conn.commit()
    conn.close()
    log.info("Initialized SQLite database at: {}", config.db_path)