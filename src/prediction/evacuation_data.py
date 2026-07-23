"""
CloudBurst — Location-Aware Evacuation Database
══════════════════════════════════════════════════
Pre-mapped evacuation routes, assembly points,
blocked zones, and SMS/PA/press scripts per location.
"""

# ═══════════════════════════════════════════════════════════
# EVACUATION DATA
# ═══════════════════════════════════════════════════════════

LOCATION_EVAC_DATA: dict = {
    "kedarnath": {
        "routes": [
            {"id": "R1", "name": "Kedarnath \u2192 Gaurikund helicopter corridor", "status": "OPEN"},
            {"id": "R2", "name": "Gaurikund \u2192 Sonprayag motor road (uphill only)", "status": "CAUTION"},
            {"id": "R3", "name": "Triyuginarayan alternate foot trail", "status": "OPEN"},
        ],
        "assembly": [
            "Sonprayag Bus Terminal (1829m)",
            "Agastmuni Community Hall",
            "Rudraprayag District HQ",
        ],
        "blocked": [
            "Mandakini riverbank path (High Flood Risk)",
            "NH-107 river-level stretch (Landslide Hazard)",
            "Kedarnath valley floor trail (Debris flow risk)",
        ],
        "sms_evac": "EVACUATE via R1, R3",
        "sms_avoid": "Avoid: Mandakini riverbank, NH-107 stretch",
        "sms_assembly": "Sonprayag Bus Terminal",
        "pa_routes": "R1 (helicopter corridor) or R3 (foot trail)",
        "pa_assembly": "Sonprayag Bus Terminal",
        "pa_avoid": "riverbank and landslide zones",
        "press_dest": "Sonprayag Bus Terminal",
    },
    "haridwar": {
        "routes": [
            {"id": "R1", "name": "Haridwar \u2192 Roorkee via NH-334 (elevated stretch)", "status": "OPEN"},
            {"id": "R2", "name": "Har ki Pauri \u2192 Upper Town via Shyampur Road", "status": "OPEN"},
            {"id": "R3", "name": "Bypass Road \u2192 BHEL Township (flood-safe)", "status": "OPEN"},
            {"id": "R4", "name": "Ganga Canal road (low-lying)", "status": "BLOCKED"},
        ],
        "assembly": [
            "Roorkee Sports Stadium",
            "BHEL Community Centre",
            "Haridwar Govt Degree College Grounds",
        ],
        "blocked": [
            "Ganga canal road (flood zone)",
            "Har ki Pauri ghats (submersion risk)",
            "Railway underpass NH-334 (waterlogging)",
        ],
        "sms_evac": "EVACUATE via R1, R2, R3",
        "sms_avoid": "Avoid: Har ki Pauri ghats, canal road",
        "sms_assembly": "Roorkee Sports Stadium",
        "pa_routes": "R1 (NH-334 elevated) or R2 (Shyampur Road)",
        "pa_assembly": "Roorkee Sports Stadium or BHEL Community Centre",
        "pa_avoid": "Ganga ghats, canal roads and railway underpasses",
        "press_dest": "Roorkee Sports Stadium",
    },
    "dehradun": {
        "routes": [
            {"id": "R1", "name": "Dehradun \u2192 Sahastradhara Road (elevated ridge)", "status": "OPEN"},
            {"id": "R2", "name": "Rajpur Road \u2192 Mussoorie bypass (uphill)", "status": "OPEN"},
            {"id": "R3", "name": "Haridwar bypass NH-72A (plains-bound)", "status": "CAUTION"},
        ],
        "assembly": [
            "Parade Ground, Dehradun",
            "FRI Grounds (elevated)",
            "Jolly Grant Airport Zone",
        ],
        "blocked": [
            "Rispana river banks (flash flood corridor)",
            "Song river flood plain stretches",
            "Bindal nala crossings (debris risk)",
        ],
        "sms_evac": "EVACUATE via R1, R2",
        "sms_avoid": "Avoid: Rispana & Bindal river banks",
        "sms_assembly": "Parade Ground Dehradun",
        "pa_routes": "R1 (Sahastradhara ridge) or R2 (Rajpur Road uphill)",
        "pa_assembly": "Parade Ground or FRI Grounds",
        "pa_avoid": "Rispana and Bindal river banks and all low-lying nalas",
        "press_dest": "Parade Ground, Dehradun",
    },
    "rishikesh": {
        "routes": [
            {"id": "R1", "name": "Rishikesh \u2192 Haridwar via NH-58 (plains-bound)", "status": "OPEN"},
            {"id": "R2", "name": "Muni ki Reti \u2192 Shivpuri elevated road", "status": "CAUTION"},
            {"id": "R3", "name": "Lakshman Jhula \u2192 Narendra Nagar ridge trail", "status": "OPEN"},
        ],
        "assembly": [
            "Rishikesh Bus Stand (Ghanta Ghar)",
            "Triveni Ghat elevated zone",
            "AIIMS Rishikesh Campus",
        ],
        "blocked": [
            "Ram Jhula / Lakshman Jhula suspension bridges (flood risk)",
            "Ganga ghats (submersion + flash flood)",
            "Choliyari nala crossings",
        ],
        "sms_evac": "EVACUATE via R1, R3",
        "sms_avoid": "Avoid: Ram Jhula, Lakshman Jhula, all ghats",
        "sms_assembly": "Rishikesh Bus Stand",
        "pa_routes": "R1 (NH-58 plains) or R3 (Narendra Nagar ridge trail)",
        "pa_assembly": "Rishikesh Bus Stand (Ghanta Ghar) or AIIMS campus",
        "pa_avoid": "Ganga ghats, suspension bridges and nala crossings",
        "press_dest": "Rishikesh Bus Stand",
    },
    "haldwani": {
        "routes": [
            {"id": "R1", "name": "Haldwani \u2192 Bareilly via NH-74 (plains-bound)", "status": "OPEN"},
            {"id": "R2", "name": "Kathgodam \u2192 Ramnagar road (elevated foothills)", "status": "OPEN"},
            {"id": "R3", "name": "Haldwani bypass \u2192 Rudrapur industrial zone", "status": "CAUTION"},
        ],
        "assembly": [
            "Haldwani Railway Station Grounds",
            "Gaula River Relief Camp (elevated bank)",
            "Kumaon University Sports Ground",
        ],
        "blocked": [
            "Gaula riverbed tracks (flash flood corridor)",
            "Low-lying Banbhoolpura settlements (inundation risk)",
            "Nainital road near Kathgodam nala (debris risk)",
        ],
        "sms_evac": "EVACUATE via R1, R2",
        "sms_avoid": "Avoid: Gaula riverbed, Banbhoolpura, Kathgodam nala",
        "sms_assembly": "Haldwani Railway Station Grounds",
        "pa_routes": "R1 (NH-74 plains-bound) or R2 (Kathgodam foothills road)",
        "pa_assembly": "Haldwani Railway Station Grounds or Kumaon University Grounds",
        "pa_avoid": "Gaula riverbed, low-lying Banbhoolpura area and all nala crossings",
        "press_dest": "Haldwani Railway Station Grounds",
    },
    "kullu": {
        "routes": [
            {"id": "R1", "name": "Kullu \u2192 Bhuntar via NH-3 (valley floor)", "status": "CAUTION"},
            {"id": "R2", "name": "Bijli Mahadev ridge road \u2192 Bajaura elevated stretch", "status": "OPEN"},
            {"id": "R3", "name": "Raison \u2192 Paddhar alternate hill road", "status": "OPEN"},
        ],
        "assembly": [
            "Bhuntar Airport Relief Zone",
            "Kullu Dussehra Ground (elevated)",
            "Bajaura HRTC Bus Depot",
        ],
        "blocked": [
            "Beas river embankment road (extreme flood risk)",
            "NH-3 Aut tunnel approach (landslide zone)",
            "Parvati valley junction (debris flow)",
        ],
        "sms_evac": "EVACUATE via R2, R3",
        "sms_avoid": "Avoid: Beas river embankment, Aut tunnel, Parvati valley",
        "sms_assembly": "Bhuntar Airport Relief Zone",
        "pa_routes": "R2 (Bijli Mahadev ridge road) or R3 (Raison alternate hill road)",
        "pa_assembly": "Bhuntar Airport Relief Zone or Kullu Dussehra Ground",
        "pa_avoid": "Beas river embankment, Aut tunnel approach and Parvati valley roads",
        "press_dest": "Bhuntar Airport Relief Zone",
    },
    "delhi": {
        "routes": [
            {"id": "R1", "name": "Yamuna flood plain \u2192 Ring Road elevated expressway", "status": "OPEN"},
            {"id": "R2", "name": "Low-lying areas \u2192 Delhi Metro elevated corridors", "status": "OPEN"},
            {"id": "R3", "name": "Outer Ring Road bypass \u2192 Noida / Gurgaon exit", "status": "OPEN"},
        ],
        "assembly": [
            "Indira Gandhi Indoor Stadium",
            "Ramlila Maidan (elevated zone)",
            "DDA Sports Complex, Saket",
        ],
        "blocked": [
            "Yamuna floodplain roads (ITO\u2013Majnu ka Tila stretch)",
            "Nizamuddin railway underpass (waterlogging)",
            "Low-lying Civil Lines pocket streets",
        ],
        "sms_evac": "EVACUATE via R1, R3",
        "sms_avoid": "Avoid: Yamuna floodplain, Nizamuddin underpass",
        "sms_assembly": "Indira Gandhi Indoor Stadium",
        "pa_routes": "R1 (Ring Road elevated) or R3 (Outer Ring Road)",
        "pa_assembly": "Indira Gandhi Indoor Stadium or Ramlila Maidan",
        "pa_avoid": "Yamuna floodplain roads and low-lying underpasses",
        "press_dest": "Indira Gandhi Indoor Stadium",
    },
}

# ── Generic fallback ─────────────────────────────────────
_EVAC_GENERIC: dict = {
    "routes": [
        {"id": "R1", "name": "Primary highway to nearest safe town", "status": "OPEN"},
        {"id": "R2", "name": "Elevated alternate road (avoid valley floor)", "status": "CAUTION"},
        {"id": "R3", "name": "Secondary district road to relief camp", "status": "OPEN"},
    ],
    "assembly": [
        "District HQ Relief Camp",
        "Nearest govt school / college grounds",
        "Local police station compound",
    ],
    "blocked": [
        "All riverbank paths (flood corridor)",
        "Low-lying underpasses and nalas",
        "Valley floor roads in heavy rain",
    ],
    "sms_evac": "EVACUATE via R1, R3",
    "sms_avoid": "Avoid: all river banks and low-lying roads",
    "sms_assembly": "District HQ Relief Camp",
    "pa_routes": "R1 (primary highway) or R3 (district road)",
    "pa_assembly": "District HQ Relief Camp or nearest school grounds",
    "pa_avoid": "all river bank paths and low-lying nala crossings",
    "press_dest": "District HQ Relief Camp",
}


def get_evac_data(loc_name: str) -> dict:
    """Return the best-matching evacuation data dict for the given location name."""
    loc_lower = loc_name.lower()
    for key in LOCATION_EVAC_DATA:
        if key in loc_lower:
            return LOCATION_EVAC_DATA[key]
    return _EVAC_GENERIC

