import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "/mlflow/mlflow.db"
con = sqlite3.connect(db)
cur = con.execute("DELETE FROM experiments WHERE lifecycle_stage='deleted'")
con.commit()
print(f"purged {cur.rowcount} deleted experiment(s)")
