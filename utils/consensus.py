"""Module de consensus par vote majoritaire sur 4 routes d'extraction."""
from typing import Optional, Any
from collections import defaultdict
from decimal import Decimal

# Ordre de priorite en cas d'egalite
ROUTE_PRIORITY = ["qwen_text", "qwen_vision", "gemma_text", "gemma_vision"]

# Poids des champs pour le score global
FIELD_WEIGHTS = {
    "numero_facture": 2,
    "date_facture": 2,
    "nom_fournisseur": 2,
    "montant_ht": 2,
    "montant_ttc": 2,
    "montant_tva": 1,
    "siret_fournisseur": 1,
    "devise": 1,
    "taux_tva": 1,
    "type_document": 1,
    "categorie": 1,
    "moyen_paiement": 1,
    "numero_tva": 1,
    "numero_commande_client": 1,
    "nb_lignes": 1,
}


def _normalize(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, Decimal):
        return round(float(val), 2)
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    s = str(val).strip()
    if s.lower() in ("", "null", "none", "n/a", "nan"):
        return None
    return s.lower()


def _match(v1: Any, v2: Any, field: str) -> bool:
    n1, n2 = _normalize(v1), _normalize(v2)
    if n1 is None and n2 is None:
        return True
    if n1 is None or n2 is None:
        return False
    if field in ("montant_ht", "montant_tva", "montant_ttc"):
        try:
            return abs(float(n1) - float(n2)) < 0.05
        except Exception:
            return n1 == n2
    if field == "taux_tva":
        try:
            return abs(float(n1) - float(n2)) < 0.5
        except Exception:
            return n1 == n2
    return n1 == n2


def get_field_value(invoice, field: str):
    if invoice is None:
        return None
    if field == "nb_lignes":
        lignes = getattr(invoice, "lignes", None)
        return len(lignes) if lignes else 0
    return getattr(invoice, field, None)


def compute_consensus(results: dict[str, Any], field: str) -> dict:
    routes = {}
    for route in ROUTE_PRIORITY:
        inv = results.get(route)
        routes[route] = get_field_value(inv, field)

    # Groupe les valeurs equivalentes
    groups = []
    assigned = set()
    for route in ROUTE_PRIORITY:
        if route in assigned:
            continue
        val = routes[route]
        group_routes = {route}
        for other_route in ROUTE_PRIORITY:
            if other_route == route or other_route in assigned:
                continue
            other_val = routes[other_route]
            if _match(val, other_val, field):
                group_routes.add(other_route)
        repr_val = None
        for r in ROUTE_PRIORITY:
            if r in group_routes and routes[r] is not None:
                repr_val = routes[r]
                break
        groups.append((repr_val, group_routes, val))
        assigned.update(group_routes)

    groups.sort(key=lambda g: len(g[1]), reverse=True)
    winner_val, winner_routes, _ = groups[0]
    votes = len(winner_routes)
    score = (votes / 4) * 100

    winner_route = None
    for r in ROUTE_PRIORITY:
        if r in winner_routes:
            winner_route = r
            break

    return {
        "value": winner_val,
        "votes": votes,
        "score": round(score, 1),
        "routes": routes,
        "winner_route": winner_route,
    }


def compute_invoice_consensus(results: dict[str, Any]) -> dict:
    fields = list(FIELD_WEIGHTS.keys())
    consensus_fields = {}
    field_scores = {}
    total_weight = 0
    weighted_sum = 0.0

    for field in fields:
        c = compute_consensus(results, field)
        consensus_fields[field] = c["value"]
        field_scores[field] = c["score"]
        weight = FIELD_WEIGHTS.get(field, 1)
        total_weight += weight
        weighted_sum += c["score"] * weight

    global_score = round(weighted_sum / total_weight, 1) if total_weight else 0.0

    if global_score >= 90:
        confidence = "high"
    elif global_score >= 60:
        confidence = "medium"
    else:
        confidence = "low"

    route_wins = defaultdict(int)
    for field in fields:
        c = compute_consensus(results, field)
        if c["winner_route"]:
            route_wins[c["winner_route"]] += 1
    winner_route = None
    if route_wins:
        winner_route = max(route_wins, key=lambda r: (route_wins[r], -ROUTE_PRIORITY.index(r)))

    return {
        "fields": consensus_fields,
        "field_scores": field_scores,
        "global_score": global_score,
        "confidence": confidence,
        "raw_results": results,
        "winner_route": winner_route,
    }
