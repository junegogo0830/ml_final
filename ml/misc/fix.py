import sqlite3

conn = sqlite3.connect("db/bankruptcy_prediction.db")
cur = conn.cursor()

cur.execute("ALTER TABLE financial_raw ADD COLUMN sj_div TEXT")
cur.execute("ALTER TABLE financial_raw ADD COLUMN sj_nm TEXT")

conn.commit()
conn.close()
print("완료")