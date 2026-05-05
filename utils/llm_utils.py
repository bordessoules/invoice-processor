import json
import re

# Ponctuations chinoises à pleine largeur → ASCII (Qwen peut les générer)
_CJK_PUNCT = str.maketrans(
    "\uff0c\uff1a\uff1b\uff01\uff1f\uff08\uff09\uff3b\uff3d\uff5b\uff5d\uff02\uff07\u3001\u3002",
    ",:;!?()[]{}\"'. "
)


def _clean_json_text(text: str) -> str:
    """Nettoie les caractères CJK et corrige les ponctuations malformées."""
    text = text.translate(_CJK_PUNCT)
    # Supprime les espaces entre guillemets et noms de clés
    text = re.sub(r'"\s+([a-z_]+)\s+"', r'"\1"', text)
    return text


def _try_yaml(text: str) -> dict | None:
    """Tente de parser un format YAML markdown type '- key: value'."""
    lines = text.strip().splitlines()
    obj = {}
    in_list = False
    current_list = []
    current_key = None

    for line in lines:
        stripped = line.lstrip("- ").strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val.lower() in ("null", "none", ""):
                val = None
            # Détection liste (lignes)
            if key == "description" and current_key == "lignes":
                current_list.append({"description": val})
            elif key in ("quantite", "prix_unitaire_ht", "montant_ht", "taux_tva") and current_list:
                try:
                    current_list[-1][key] = float(val) if "." in val else int(val)
                except (ValueError, TypeError):
                    current_list[-1][key] = val
            elif key == "lignes":
                current_key = "lignes"
                in_list = True
            else:
                in_list = False
                obj[key] = val
        else:
            in_list = False
    if current_list:
        obj["lignes"] = current_list
    return obj if obj else None


def extract_json(text: str) -> dict:
    """Extrait le JSON d'une réponse LLM, même si entourée de markdown, texte ou YAML."""
    original = text.strip()
    candidates = [original]

    # 1. Essai direct
    for raw in candidates:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # 2. Cherche entre accolades
    start = original.find("{")
    end = original.rfind("}")
    if start != -1 and end != -1 and end > start:
        block = original[start : end + 1]
        candidates.append(block)
        candidates.append(_clean_json_text(block))

    # 3. Cherche entre balises code
    if "```json" in original:
        block = original.split("```json", 1)[1].split("```", 1)[0].strip()
        candidates.append(block)
        candidates.append(_clean_json_text(block))

    for raw in candidates:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # 4. Fallback YAML markdown
    if original.startswith("-"):
        obj = _try_yaml(original)
        if obj:
            return obj

    raise ValueError(f"JSON introuvable dans la réponse LLM: {original[:200]}...")
