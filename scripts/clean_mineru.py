from db import _get_conn
conn = _get_conn()
conn.execute("DELETE FROM llm_extractions WHERE pdf_extractor='mineru'")
conn.commit()
conn.close()
print("MinerU entries cleaned")
