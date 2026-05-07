"""Streamlit reviewer: PDF a gauche, recap editable a droite, save en DB."""
import json, sqlite3, hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF

from db import init_db
from utils.consensus import _match, FIELD_WEIGHTS

DB_PATH = "output/invoices.db"

# Champs affichés dans l'ordre, par section logique
FIELD_ORDER = [
    # identité
    "numero_facture", "date_facture", "type_document",
    # fournisseur
    "nom_fournisseur", "siret_fournisseur", "numero_tva",
    # montants
    "montant_ht", "montant_tva", "montant_ttc", "taux_tva", "devise",
    # contexte
    "categorie", "moyen_paiement", "numero_commande_client",
]

NUMERIC_FIELDS = {"montant_ht", "montant_tva", "montant_ttc", "taux_tva"}


# ── DB helpers (cached) ─────────────────────────────────────────────

@st.cache_data
def list_pdfs() -> list[str]:
    """All distinct pdf_paths in llm_extractions (excl. nemotron)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT DISTINCT pdf_path FROM llm_extractions
        WHERE llm_model NOT LIKE '%nemotron%'
        ORDER BY pdf_path
    """).fetchall()
    conn.close()
    return [r[0] for r in rows]


@st.cache_data
def load_extractions(pdf_path: str) -> list[dict]:
    """All extractions for one PDF (excl. nemotron)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT llm_name, llm_model, pdf_extractor, mode, status, result_json
        FROM llm_extractions
        WHERE pdf_path=? AND llm_model NOT LIKE '%nemotron%'
    """, (pdf_path,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        data: dict = {}
        try:
            if r[4] == "ok" and r[5]:
                parsed = json.loads(r[5])
                # Some LLMs return a list (one item per page) instead of a dict.
                # Take the first dict-like element.
                if isinstance(parsed, dict):
                    data = parsed
                elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    data = parsed[0]
        except Exception:
            pass
        out.append({
            "name": r[0], "model": r[1], "extractor": r[2], "mode": r[3], "status": r[4],
            "data": data,
            "route": f"{r[0]} / {r[2]}",
        })
    return out


def get_review(pdf_path: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT final_data_json, status, reviewed_at FROM manual_review WHERE pdf_path=?",
        (pdf_path,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"data": json.loads(row[0]), "status": row[1], "reviewed_at": row[2]}


def save_review(pdf_path: str, data: dict, status: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT OR REPLACE INTO manual_review
           (pdf_path, final_data_json, status, reviewed_at)
           VALUES (?, ?, ?, ?)""",
        (pdf_path, json.dumps(data, ensure_ascii=False, default=str),
         status, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


# ── Consensus computation ───────────────────────────────────────────

def field_consensus(extractions: list[dict], field: str) -> list[dict]:
    """Group routes by equivalent value, sorted by vote count desc."""
    routes = []
    for e in extractions:
        if e["status"] != "ok":
            continue
        val = e["data"].get(field) if field != "nb_lignes" else len(e["data"].get("lignes") or [])
        routes.append((e["route"], val))

    if not routes:
        return []

    groups: list[dict] = []
    assigned: set[int] = set()
    for i, (_, vi) in enumerate(routes):
        if i in assigned:
            continue
        group_idx = {i}
        for j in range(len(routes)):
            if j == i or j in assigned:
                continue
            if _match(vi, routes[j][1], field):
                group_idx.add(j)
        # Pick representative non-null value
        repr_val = next((routes[k][1] for k in group_idx if routes[k][1] is not None), None)
        groups.append({
            "value": repr_val,
            "votes": len(group_idx),
            "routes": [routes[k][0] for k in sorted(group_idx)],
        })
        assigned.update(group_idx)

    total = len(routes)
    for g in groups:
        g["pct"] = round(100 * g["votes"] / total, 1)
        g["total"] = total
    groups.sort(key=lambda g: g["votes"], reverse=True)
    return groups


@st.cache_data
def pdf_summary(pdf_path: str) -> dict:
    """Cached: global score (weighted avg of field consensus), n_ok routes, low-conf fields list."""
    exs = load_extractions(pdf_path)
    n_ok = sum(1 for e in exs if e["status"] == "ok")
    if n_ok == 0:
        return {"score": 0.0, "n_ok": 0, "n_total": len(exs), "issues": []}
    total_w = 0
    weighted = 0.0
    issues: list[str] = []
    for f, w in FIELD_WEIGHTS.items():
        groups = field_consensus(exs, f)
        if not groups:
            continue
        winner = groups[0]
        weighted += winner["pct"] * w
        total_w += w
        if winner["pct"] < 75:
            issues.append(f)
    return {
        "score": weighted / total_w if total_w else 0.0,
        "n_ok": n_ok,
        "n_total": len(exs),
        "issues": issues,
    }


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}


@st.cache_data
def render_pdf_png(pdf_path: str, dpi: int = 120) -> list[bytes]:
    """Render each page (PDF) or single raster image to PNG bytes.

    Reads file into memory first to avoid file-handle conflicts with concurrent
    run_llm.py vision processes.
    """
    suffix = Path(pdf_path).suffix.lower()
    with open(pdf_path, "rb") as f:
        data = f.read()

    if suffix in IMAGE_EXTS:
        # For raster images, just return the file bytes directly — st.image
        # handles JPG/PNG/etc natively, no need to re-encode through PyMuPDF.
        return [data]

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        out = [page.get_pixmap(dpi=dpi).tobytes("png") for page in doc]
    finally:
        doc.close()
    return out


# ── UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Invoice Reviewer", layout="wide", page_icon="📄")
init_db()  # ensure manual_review table exists

st.title("📄 Invoice Reviewer")

all_pdfs = list_pdfs()
if not all_pdfs:
    st.warning("Aucune extraction en DB. Lance un run d'abord (`python run_llm.py ...`).")
    st.stop()

# Sidebar: filter + sort + select
with st.sidebar:
    st.header("Sélection")

    if st.button("🔄 Recharger la DB", use_container_width=True):
        list_pdfs.clear()
        pdf_summary.clear()
        load_extractions.clear()
        st.rerun()

    # Build enriched list FIRST so we can show counts in the filter labels
    items_all = []
    for p in all_pdfs:
        s = pdf_summary(p)
        r = get_review(p)
        items_all.append({
            "pdf": p,
            "name": Path(p).name,
            "score": s["score"],
            "n_ok": s["n_ok"],
            "n_total": s["n_total"],
            "issues": s["issues"],
            "reviewed": r is not None,
            "review_status": r["status"] if r else None,
        })

    # Pre-compute counts per filter
    n_tout = len(items_all)
    n_non_revus = sum(1 for i in items_all if not i["reviewed"])
    n_revus = sum(1 for i in items_all if i["reviewed"])
    n_lt80 = sum(1 for i in items_all if i["score"] < 80)
    n_lt60 = sum(1 for i in items_all if i["score"] < 60)

    filter_keys = ["tout", "non revus", "revus", "score < 80%", "score < 60%"]
    # Counts shown as caption (the radio itself uses stable labels so its session
    # state isn't invalidated by changing labels at each rerun).
    st.caption(
        f"tout {n_tout}  ·  non revus {n_non_revus}  ·  revus {n_revus}  "
        f"·  <80% {n_lt80}  ·  <60% {n_lt60}"
    )
    if st.session_state.get("filter_radio") not in filter_keys:
        st.session_state["filter_radio"] = "tout"
    flt = st.radio("Filtre", options=filter_keys, key="filter_radio")

    sort_options = ["par dossier", "score croissant", "score décroissant", "alphabétique"]
    if st.session_state.get("sort_radio") not in sort_options:
        st.session_state["sort_radio"] = "par dossier"
    sort_by = st.radio("Trier par", sort_options, key="sort_radio")

    # Apply filter
    items = list(items_all)
    if flt == "non revus":
        items = [i for i in items if not i["reviewed"]]
    elif flt == "revus":
        items = [i for i in items if i["reviewed"]]
    elif flt == "score < 80%":
        items = [i for i in items if i["score"] < 80]
    elif flt == "score < 60%":
        items = [i for i in items if i["score"] < 60]

    # Sort
    if sort_by == "par dossier":
        items.sort(key=lambda i: i["pdf"])  # full path → folder order
    elif sort_by == "score croissant":
        items.sort(key=lambda i: (i["score"], i["name"]))
    elif sort_by == "score décroissant":
        items.sort(key=lambda i: (-i["score"], i["name"]))
    else:
        items.sort(key=lambda i: i["name"])

    st.caption(f"{len(items)} / {len(all_pdfs)} PDFs")

    if not items:
        st.warning("Aucun PDF dans ce filtre.")
        st.stop()

    # Persist selection across reruns
    if "selected_idx" not in st.session_state:
        st.session_state["selected_idx"] = 0
    if "selected_path_prev" not in st.session_state:
        st.session_state["selected_path_prev"] = None
    if "dirty" not in st.session_state:
        st.session_state["dirty"] = False

    labels = [
        f"{'✓ ' if i['reviewed'] else ''}{i['score']:5.1f}%  {Path(i['pdf']).parent.name}/{i['name'][:42]}"
        for i in items
    ]
    idx = st.selectbox(
        "PDF",
        options=list(range(len(items))),
        format_func=lambda k: labels[k],
        index=min(st.session_state["selected_idx"], len(items) - 1),
        key="pdf_select",
    )
    st.session_state["selected_idx"] = idx

    selected = items[idx]

    # Detect PDF switch with unsaved edits
    if (
        st.session_state["selected_path_prev"] is not None
        and st.session_state["selected_path_prev"] != selected["pdf"]
        and st.session_state["dirty"]
    ):
        st.error("⚠️ Modifications non sauvegardées sur le PDF précédent — elles ont été perdues.")
        st.session_state["dirty"] = False
    st.session_state["selected_path_prev"] = selected["pdf"]
    st.divider()
    st.metric("Routes OK", f"{selected['n_ok']}/{selected['n_total']}")
    st.metric("Score consensus", f"{selected['score']:.1f}%")
    if selected["issues"]:
        st.warning(f"Champs litigieux : {', '.join(selected['issues'])}")

# Main: 2 cols
left, right = st.columns([1.1, 1])

selected_path = selected["pdf"]
exs = load_extractions(selected_path)
review = get_review(selected_path)
review_data = review["data"] if review else {}

with left:
    st.subheader(Path(selected_path).name)
    st.caption(selected_path)
    # Independent scroll: fixed-height container so the PDF can scroll on its own
    pdf_box = st.container(height=900, border=False)
    with pdf_box:
        try:
            pages = render_pdf_png(selected_path)
            for i, png in enumerate(pages):
                st.image(png, caption=f"Page {i+1}/{len(pages)}", use_container_width=True)
        except Exception as e:
            st.error(f"Impossible de rendre le PDF : {e}")

with right:
    # Save+status block OUTSIDE the scrollable area → always visible at top of right column.
    # Filled at the end of the inner with-block once edited_recap / edited_lignes are known.
    save_block = st.container()
    st.divider()

    # Independent scroll: fixed-height container so the right column scrolls on its own
    with st.container(height=900, border=False):
        # PDF-specific suffix for widget keys → chaque PDF a son state isolé.
        path_key = hashlib.md5(selected_path.encode()).hexdigest()[:10]

        # Compute consensus winners for the recap fields
        def _winner(field):
            groups = field_consensus(exs, field)
            return (groups[0]["value"], groups[0]["pct"]) if groups else (None, 0.0)

        RECAP_FIELDS = [
            ("date_facture", "Date", "text"),
            ("categorie", "Poste comptable", "text"),
            ("nom_fournisseur", "Fournisseur", "text"),
            ("montant_ht", "Montant HT", "num"),
            ("montant_tva", "TVA", "num"),
            ("montant_ttc", "Montant TTC", "num"),
            ("taux_tva", "Taux TVA", "num"),
            ("type_document", "Type", "text"),
            ("moyen_paiement", "Paiement", "text"),
            ("numero_facture", "N° facture", "text"),
        ]

        rows = []
        for field, label, kind in RECAP_FIELDS:
            v, p = _winner(field)
            # Override with reviewed value if it exists
            if review_data and field in review_data:
                v = review_data[field]
            badge = "🟢" if p >= 90 else ("🟡" if p >= 60 else "🔴")
            rows.append({
                "champ": label,
                "confiance": f"{badge} {p:.0f}%",
                "valeur": "" if v is None else str(v),
                "_field": field,
                "_kind": kind,
            })
        recap_df = pd.DataFrame(rows)

        # ── Récap éditable ──────────────────────────────────────────
        st.subheader("📋 Récap pour déclaration TVA")
        st.caption(f"{selected['n_ok']}/{selected['n_total']} routes OK · score {selected['score']:.0f}%")

        edited_recap = st.data_editor(
            recap_df.drop(columns=["_field", "_kind"]),
            column_config={
                "champ": st.column_config.TextColumn("Champ", disabled=True, width="medium"),
                "confiance": st.column_config.TextColumn("Conf.", disabled=True, width="small"),
                "valeur": st.column_config.TextColumn("Valeur", width="large"),
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key=f"recap_{path_key}",
        )

        # Sanity check HT + TVA = TTC (read post-edit values)
        def _to_float(x):
            try:
                return float(str(x).replace(",", "."))
            except Exception:
                return None
        edited_map = {RECAP_FIELDS[i][0]: edited_recap.iloc[i]["valeur"] for i in range(len(RECAP_FIELDS))}
        ht_f = _to_float(edited_map.get("montant_ht"))
        tva_f = _to_float(edited_map.get("montant_tva"))
        ttc_f = _to_float(edited_map.get("montant_ttc"))
        if ht_f is not None and tva_f is not None and ttc_f is not None:
            diff = abs((ht_f + tva_f) - ttc_f)
            if diff > 0.05:
                st.warning(f"⚠️ HT + TVA = {ht_f + tva_f:.2f} ≠ TTC = {ttc_f:.2f}  (écart {diff:.2f})")
            else:
                st.caption(f"✓ HT + TVA = TTC ({ht_f:.2f} + {tva_f:.2f} = {ttc_f:.2f})")

        # ── Lignes ──────────────────────────────────────────────────
        candidates = [(e, len(e["data"].get("lignes") or []))
                      for e in exs if e["status"] == "ok" and e["data"].get("lignes")]
        candidates.sort(key=lambda c: c[1], reverse=True)
        line_routes = [c[0] for c in candidates]

        st.subheader("Lignes")
        if line_routes:
            labels_lr = [f"{e['name']} / {e['extractor']}  ({len(e['data'].get('lignes') or [])})" for e in line_routes]
            # Default to reviewed lines source if exists
            default_idx = 0
            if review_data and "_lignes_source" in review_data:
                try:
                    src = review_data["_lignes_source"]
                    default_idx = next((i for i, e in enumerate(line_routes)
                                       if f"{e['name']}/{e['extractor']}" == src), 0)
                except Exception:
                    pass

            sel_lr = st.selectbox(
                "Source des lignes",
                options=range(len(line_routes)),
                format_func=lambda i: labels_lr[i],
                index=default_idx,
                key=f"line_src_{path_key}",
            )
            chosen = line_routes[sel_lr]
            # If reviewed, use saved lignes by default; else use chosen route's lignes
            if review_data and "lignes" in review_data and review_data.get("_lignes_source") == f"{chosen['name']}/{chosen['extractor']}":
                base_lignes = review_data["lignes"]
            else:
                base_lignes = chosen["data"].get("lignes") or []
            lignes_df = pd.DataFrame([{
                "description": l.get("description") if isinstance(l, dict) else None,
                "qté": l.get("quantite") if isinstance(l, dict) else None,
                "prix unit. HT": l.get("prix_unitaire_ht") if isinstance(l, dict) else None,
                "montant HT": l.get("montant_ht") if isinstance(l, dict) else None,
                "taux TVA": l.get("taux_tva") if isinstance(l, dict) else None,
            } for l in base_lignes])

            edited_lignes = st.data_editor(
                lignes_df if not lignes_df.empty else pd.DataFrame(columns=["description", "qté", "prix unit. HT", "montant HT", "taux TVA"]),
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key=f"lignes_{path_key}",
            )

            # Sum check
            try:
                sum_ht = edited_lignes["montant HT"].fillna(0).astype(float).sum()
                if ht_f is not None and abs(sum_ht - ht_f) > 0.05:
                    st.caption(f"ℹ️ Σ lignes HT = {sum_ht:.2f}  vs  Total HT recap = {ht_f}  (écart {abs(sum_ht - ht_f):.2f})")
            except Exception:
                pass
        else:
            edited_lignes = pd.DataFrame()
            chosen = None
            st.caption("(aucune route n'a extrait de lignes)")

        # ── Statut + Save (rendu dans le placeholder en haut) ──────
        with save_block:
            col_status, col_btn = st.columns([2, 1])
            with col_status:
                statuses = ["pending", "validated", "needs_more_info"]
                cur_status = review["status"] if review else "pending"
                new_status = st.radio(
                    "Statut",
                    statuses,
                    index=statuses.index(cur_status) if cur_status in statuses else 0,
                    horizontal=True,
                    key=f"status_{path_key}",
                )
            with col_btn:
                st.write("")  # spacing
                submitted = st.button("💾 Sauvegarder", type="primary", use_container_width=True, key=f"save_{path_key}")
            if review:
                st.caption(f"Dernière révision : {review['reviewed_at']}  ·  statut `{review['status']}`")

        if submitted:
            # Build final dict from edited recap
            final = {}
            numeric_fields = {f for f, _, k in RECAP_FIELDS if k == "num"}
            for i, (field, _, _) in enumerate(RECAP_FIELDS):
                raw = edited_recap.iloc[i]["valeur"]
                raw = ("" if raw is None else str(raw)).strip()
                if not raw:
                    final[field] = None
                elif field in numeric_fields:
                    try:
                        final[field] = float(raw.replace(",", "."))
                    except ValueError:
                        final[field] = raw
                else:
                    final[field] = raw

            # Lignes from edited dataframe
            if not edited_lignes.empty:
                final["lignes"] = [
                    {
                        "description": (None if pd.isna(r["description"]) else str(r["description"])),
                        "quantite": (None if pd.isna(r["qté"]) else r["qté"]),
                        "prix_unitaire_ht": (None if pd.isna(r["prix unit. HT"]) else r["prix unit. HT"]),
                        "montant_ht": (None if pd.isna(r["montant HT"]) else r["montant HT"]),
                        "taux_tva": (None if pd.isna(r["taux TVA"]) else r["taux TVA"]),
                    }
                    for _, r in edited_lignes.iterrows()
                ]
                if chosen:
                    final["_lignes_source"] = f"{chosen['name']}/{chosen['extractor']}"

            save_review(selected_path, final, new_status)
            st.success(f"Sauvegardé : {Path(selected_path).name}")
            list_pdfs.clear()
            pdf_summary.clear()
            st.rerun()
