import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sanctions-api")

# --- Config ---
TRADEGOV_API_KEY = os.getenv("TRADEGOV_API_KEY")
BASE_URL = "https://api.trade.gov/consolidated_screening_list/search"
HTTP_TIMEOUT = 10.0
CACHE_TTL = 3600  # 1h — sanctions data is time-sensitive; CSL updates hourly
HIGH_SCORE_THRESHOLD = 95.0  # tunable; see note in classify_risk

# Required compliance disclaimer (Trade.gov frames CSL as a screening aid, not a determination)
LEGAL_DISCLAIMER = (
    "This result is derived from the U.S. Government Consolidated Screening List (CSL), an aggregation "
    "of restricted-party lists from the Departments of Commerce, State, and Treasury. It is a screening "
    "aid for review workflows, not a legal determination or clearance. A match is not conclusive; if any "
    "party appears to match, you MUST independently verify against the official Federal Register and the "
    "administering agency's list (OFAC/BIS/DDTC) and conduct further due diligence before taking any "
    "transaction, blocking, or compliance action."
)

# Map CSL source abbreviations -> human-readable name + administering agency.
# We map locally rather than trusting an upstream agency field.
SOURCES_METADATA: Dict[str, Dict[str, str]] = {
    "SDN": {"name": "Specially Designated Nationals List", "agency": "Treasury (OFAC)"},
    "EL": {"name": "Entity List", "agency": "Commerce (BIS)"},
    "DPL": {"name": "Denied Persons List", "agency": "Commerce (BIS)"},
    "UVL": {"name": "Unverified List", "agency": "Commerce (BIS)"},
    "MEU": {"name": "Military End User List", "agency": "Commerce (BIS)"},
    "FSE": {"name": "Foreign Sanctions Evaders List", "agency": "Treasury (OFAC)"},
    "SSI": {"name": "Sectoral Sanctions Identifications List", "agency": "Treasury (OFAC)"},
    "CMIC": {"name": "Non-SDN Chinese Military-Industrial Complex List", "agency": "Treasury (OFAC)"},
    "NS-MBS": {"name": "Non-SDN Menu-Based Sanctions List", "agency": "Treasury (OFAC)"},
    "CAP": {"name": "Correspondent/Payable-Through Account Sanctions", "agency": "Treasury (OFAC)"},
    "NS-PLC": {"name": "Nonproliferation Sanctions", "agency": "State"},
    "DTC": {"name": "ITAR Debarred List", "agency": "State (DDTC)"},
    "ISN": {"name": "Nonproliferation Sanctions (ISN)", "agency": "State"},
}

# --- Globals ---
client: Optional[httpx.AsyncClient] = None
cache: Dict[str, Dict[str, Any]] = {}  # key -> {"expiry": ts, "data": dict}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    if not TRADEGOV_API_KEY:
        # Fail loud: the product is useless without the upstream key.
        raise RuntimeError("TRADEGOV_API_KEY environment variable is required (free key at api.data.gov).")
    client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    logger.info("Sanctions API started; Trade.gov key configured.")
    yield
    await client.aclose()


app = FastAPI(
    title="Sanctions & Restricted-Party Screening API",
    version="1.0.0",
    description="Screen names against U.S. government restricted-party lists via the Trade.gov Consolidated Screening List.",
    lifespan=lifespan,
)


# --- Models ---
class Match(BaseModel):
    name: str
    source: str
    source_name: str
    source_agency: str
    programs: List[str] = []
    addresses: List[str] = []
    ids: List[str] = []
    score: Optional[float] = None
    source_list_url: Optional[str] = None


class ScreenResponse(BaseModel):
    query: str
    risk_level: str  # HIGH | REVIEW | NO_MATCH_FOUND
    review_required: bool
    match_count: int
    matches: List[Match]
    disclaimer: str


class BatchRequest(BaseModel):
    names: List[str] = Field(..., min_length=1, max_length=10)
    sources: Optional[str] = None
    fuzzy: bool = True


# --- Helpers ---
def _flatten_addresses(raw: Any) -> List[str]:
    out: List[str] = []
    for a in (raw or []):
        if isinstance(a, dict):
            parts = [a.get("address"), a.get("city"), a.get("state"), a.get("postal_code"), a.get("country")]
            joined = ", ".join(p for p in parts if p)
            if joined:
                out.append(joined)
        elif isinstance(a, str) and a.strip():
            out.append(a.strip())
    return out


def _flatten_ids(raw: Any) -> List[str]:
    out: List[str] = []
    for d in (raw or []):
        if isinstance(d, dict):
            s = f"{d.get('type', 'ID')}: {d.get('number', '')}".strip()
            if s and s != "ID:":
                out.append(s)
        elif isinstance(d, str) and d.strip():
            out.append(d.strip())
    return out


def _to_match(item: Dict[str, Any]) -> Match:
    code = item.get("source") or "UNKNOWN"
    meta = SOURCES_METADATA.get(code, {"name": code, "agency": "U.S. Federal Agency"})
    score = item.get("score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    return Match(
        name=item.get("name") or "",
        source=code,
        source_name=meta["name"],
        source_agency=meta["agency"],
        programs=item.get("programs") or [],
        addresses=_flatten_addresses(item.get("addresses")),
        ids=_flatten_ids(item.get("ids")),
        score=score,
        source_list_url=item.get("source_list_url") or item.get("source_information_url"),
    )


def classify_risk(query: str, matches: List[Match]) -> tuple[str, bool]:
    """Deterministic + conservative. Does not blindly trust the upstream score scale.
    - No matches            -> NO_MATCH_FOUND (NOT 'cleared'); review_required False
    - Exact name match, or a score at/above HIGH_SCORE_THRESHOLD -> HIGH
    - Any other match       -> REVIEW
    Any presence of matches always sets review_required True.
    """
    if not matches:
        return "NO_MATCH_FOUND", False
    q = query.strip().casefold()
    for m in matches:
        if m.name.strip().casefold() == q:
            return "HIGH", True
        if m.score is not None and m.score >= HIGH_SCORE_THRESHOLD:
            return "HIGH", True
    return "REVIEW", True


async def _call_csl(name: str, sources: Optional[str], fuzzy: bool) -> Dict[str, Any]:
    key = f"{name.casefold()}|{sources or ''}|{fuzzy}"
    now = time.time()
    hit = cache.get(key)
    if hit and hit["expiry"] > now:
        return hit["data"]
    params = {"api_key": TRADEGOV_API_KEY, "q": name, "fuzzy_name": str(fuzzy).lower(), "size": 50}
    if sources:
        params["sources"] = sources
    try:
        resp = await client.get(BASE_URL, params=params)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Trade.gov request timed out.")
    if resp.status_code in (401, 403):
        raise HTTPException(status_code=500, detail="Trade.gov API key invalid or unauthorized.")
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Trade.gov upstream error: {e}")
    data = resp.json()
    cache[key] = {"expiry": now + CACHE_TTL, "data": data}
    return data


async def _screen(name: str, sources: Optional[str], fuzzy: bool) -> ScreenResponse:
    clean = name.strip()
    if not clean:
        raise HTTPException(status_code=400, detail="name cannot be blank.")
    data = await _call_csl(clean, sources, fuzzy)
    matches = [_to_match(r) for r in (data.get("results") or [])]
    risk_level, review_required = classify_risk(clean, matches)
    return ScreenResponse(
        query=clean,
        risk_level=risk_level,
        review_required=review_required,
        match_count=len(matches),
        matches=matches,
        disclaimer=LEGAL_DISCLAIMER,
    )


# --- Endpoints ---
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "api_key_configured": bool(TRADEGOV_API_KEY),
        "cache_entries": len(cache),
    }


@app.get("/sources")
async def sources():
    return {
        "description": "Restricted-party lists aggregated by the Trade.gov Consolidated Screening List. "
                       "Live composition may vary; this is a reference map.",
        "sources": [{"code": c, **info} for c, info in SOURCES_METADATA.items()],
        "count": len(SOURCES_METADATA),
        "updated": "hourly via Trade.gov CSL",
    }


@app.get("/screen", response_model=ScreenResponse)
async def screen(
    name: str = Query(..., min_length=2, description="Entity or person name to screen, e.g. Rosoboronexport"),
    sources: Optional[str] = Query(None, description="Comma-separated source codes to filter, e.g. SDN,EL"),
    fuzzy: bool = Query(True, description="Enable fuzzy name matching"),
):
    return await _screen(name, sources, fuzzy)


@app.post("/screen/batch")
async def screen_batch(req: BatchRequest):
    results = []
    for n in req.names:
        try:
            results.append((await _screen(n, req.sources, req.fuzzy)).model_dump())
        except HTTPException as e:
            results.append({
                "query": n, "risk_level": "ERROR", "review_required": True,
                "match_count": 0, "matches": [], "error": e.detail, "disclaimer": LEGAL_DISCLAIMER,
            })
    return {"results": results, "total_screened": len(req.names), "disclaimer": LEGAL_DISCLAIMER}
