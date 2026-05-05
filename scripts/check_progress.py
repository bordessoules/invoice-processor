from db import _get_conn
conn = _get_conn()
n = conn.execute("SELECT COUNT(*) as c FROM llm_extractions WHERE pdf_extractor='mineru' AND llm_name='qwen3.6-35b'").fetchone()["c"]
print(f"MinerU+Qwen: {n}/96")
conn.close()
