"""Migration des chemins de la DB après réorg factures/<vendor>/ -> factures/2025/<vendor>/.

Insère "2025" comme sous-dossier entre "factures" et le dossier vendeur.
Préserve toutes les manual_review existantes.
"""
import sqlite3
from pathlib import Path

DB = "output/invoices.db"


def migrate(old: str) -> str:
    """Insert '2025' folder right after 'factures' in a path string.
    Works for both Windows absolute (C:\...\factures\vendor\file.pdf)
    and relative (factures\vendor\file.pdf) paths.
    """
    parts = list(Path(old).parts)
    try:
        idx = parts.index("factures")
    except ValueError:
        return old  # nothing to migrate
    # Skip if already has '2025' inserted
    if idx + 1 < len(parts) and parts[idx + 1] == "2025":
        return old
    parts.insert(idx + 1, "2025")
    return str(Path(*parts))


def main(commit: bool = False):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # Build filename → disk path index (for fallback when migrate() doesn't find file)
    disk_by_name = {}
    for p in Path("factures").rglob("*.pdf"):
        disk_by_name.setdefault(p.name, []).append(str(p.resolve()))

    def resolve_one(old: str) -> str | None:
        """Return new path if findable on disk, else None."""
        candidate = migrate(old)
        if Path(candidate).exists():
            return candidate
        # Fallback: filename match
        name = Path(old).name
        candidates = disk_by_name.get(name, [])
        if len(candidates) == 1:
            return candidates[0]
        # Multiple candidates: prefer the one matching expected vendor folder
        if len(candidates) > 1:
            old_parent = Path(old).parent.name  # vendor folder
            for c in candidates:
                if Path(c).parent.name == old_parent:
                    return c
        return None

    # Build mapping for ALL distinct pdf_paths in any of the 3 tables
    paths_all = set()
    for r in conn.execute("SELECT DISTINCT pdf_path FROM llm_extractions").fetchall():
        paths_all.add(r[0])
    for r in conn.execute("SELECT DISTINCT pdf_path FROM pdf_texts").fetchall():
        paths_all.add(r[0])
    for r in conn.execute("SELECT pdf_path FROM manual_review").fetchall():
        paths_all.add(r[0])

    mapping = {}
    orphans = []
    for old in paths_all:
        new = resolve_one(old)
        if new is None:
            orphans.append(old)
        elif new != old:
            mapping[old] = new

    print(f"Total distinct paths in DB    : {len(paths_all)}")
    print(f"Will be remapped              : {len(mapping)}")
    print(f"Orphans (no disk match)       : {len(orphans)}")
    if orphans:
        print("\nOrphaned PDFs (will be deleted from DB):")
        for o in orphans[:10]:
            print(f"  {o}")
        if len(orphans) > 10:
            print(f"  ... +{len(orphans)-10} more")

    if not commit:
        print("\n[DRY RUN] Pass --commit to apply.")
        return

    # Apply updates inside a transaction
    cur = conn.cursor()
    n_ll = n_pt = n_mr = 0
    for old, new in mapping.items():
        n_ll += cur.execute("UPDATE llm_extractions SET pdf_path=? WHERE pdf_path=?", (new, old)).rowcount
        n_pt += cur.execute("UPDATE pdf_texts        SET pdf_path=? WHERE pdf_path=?", (new, old)).rowcount
        n_mr += cur.execute("UPDATE manual_review    SET pdf_path=? WHERE pdf_path=?", (new, old)).rowcount

    # Delete orphans (PDFs no longer on disk)
    o_ll = o_pt = o_mr = 0
    for old in orphans:
        o_ll += cur.execute("DELETE FROM llm_extractions WHERE pdf_path=?", (old,)).rowcount
        o_pt += cur.execute("DELETE FROM pdf_texts        WHERE pdf_path=?", (old,)).rowcount
        o_mr += cur.execute("DELETE FROM manual_review    WHERE pdf_path=?", (old,)).rowcount

    conn.commit()
    print(f"\n[OK] Updated rows:")
    print(f"  llm_extractions : {n_ll}")
    print(f"  pdf_texts       : {n_pt}")
    print(f"  manual_review   : {n_mr}")
    if orphans:
        print(f"\n[OK] Deleted orphan rows:")
        print(f"  llm_extractions : {o_ll}")
        print(f"  pdf_texts       : {o_pt}")
        print(f"  manual_review   : {o_mr}")
    conn.close()


if __name__ == "__main__":
    import sys
    commit = "--commit" in sys.argv
    main(commit=commit)
