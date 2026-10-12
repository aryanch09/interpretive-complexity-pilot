"""
Anonymization pipeline for the familiarity alternative explanation test.

Hypothesis: the LLM may recognize well-known firms from its training data and
converge on a consensus view regardless of filing content, producing artificially
low disagreement for familiar firms (large-caps).

Test design:
  1. Run agents on the original filing → get disagreement scores (raw)
  2. Scrub the filing to replace company-identifying strings with placeholders
  3. Re-run agents on the anonymized filing → get disagreement scores (anon)
  4. If raw_disagreement >> anon_disagreement for large-caps only, familiarity
     is a plausible explanation. If disagreement is similar, it is not.

Placeholders:
  [COMPANY]    - company name and common abbreviations
  [TICKER]     - stock ticker symbol
  [EXECUTIVE]  - named executives (CEO, CFO, etc.)
  [PRODUCT]    - product names, brand names, sub-brands
  [SUBSIDIARY] - subsidiary or division names
  [LOCATION]   - headquarters city/state
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ── Firm metadata for anonymization ──────────────────────────────────────────
# Keys: lists of strings to replace. Order matters — longer/more specific strings
# should appear before shorter ones to avoid partial replacements.

FIRM_METADATA: dict[str, dict] = {
    "AAPL": {
        "company": ["Apple Inc.", "Apple Inc", "Apple"],
        "executives": ["Tim Cook", "Luca Maestri", "Jeff Williams"],
        "products": ["iPhone", "iPad", "MacBook", "Mac Pro", "Mac mini", "Apple Watch",
                     "AirPods", "Apple TV", "App Store", "iCloud", "Apple Pay",
                     "Apple Silicon", "M1", "M2", "M3", "iOS", "macOS", "watchOS"],
        "subsidiaries": ["Apple Services", "Apple Music", "Apple Arcade"],
        "locations": ["Cupertino", "Silicon Valley"],
    },
    "MSFT": {
        "company": ["Microsoft Corporation", "Microsoft Corp.", "Microsoft Corp", "Microsoft"],
        "executives": ["Satya Nadella", "Amy Hood", "Brad Smith"],
        "products": ["Windows", "Azure", "Office 365", "Microsoft 365", "Teams",
                     "Xbox", "LinkedIn", "GitHub", "Bing", "Surface", "Dynamics",
                     "Power BI", "SharePoint", "OneDrive", "Copilot"],
        "subsidiaries": ["LinkedIn", "GitHub", "Nuance", "Activision Blizzard"],
        "locations": ["Redmond", "Puget Sound"],
    },
    "JPM": {
        "company": ["JPMorgan Chase & Co.", "JPMorgan Chase & Co", "JPMorgan Chase",
                    "JPMorgan", "J.P. Morgan"],
        "executives": ["Jamie Dimon", "Jeremy Barnum", "Mary Callahan Erdoes"],
        "products": ["Chase", "JPMorgan Asset Management", "CIB"],
        "subsidiaries": ["Chase Bank", "J.P. Morgan Securities", "WePay"],
        "locations": ["New York City", "New York, New York", "Manhattan"],
    },
    "JNJ": {
        "company": ["Johnson & Johnson", "J&J"],
        "executives": ["Joaquin Duato", "Joseph Wolk"],
        "products": ["Tylenol", "Band-Aid", "Neutrogena", "Listerine", "STELARA",
                     "DARZALEX", "Janssen", "Concerta"],
        "subsidiaries": ["Janssen Pharmaceuticals", "Kenvue", "DePuy Synthes"],
        "locations": ["New Brunswick", "New Jersey"],
    },
    "WMT": {
        "company": ["Walmart Inc.", "Walmart Inc", "Walmart"],
        "executives": ["Doug McMillon", "John David Rainey", "C. Douglas McMillon"],
        "products": ["Walmart+", "Sam's Club", "Walmart.com", "Great Value"],
        "subsidiaries": ["Sam's Club", "Flipkart", "PhonePe"],
        "locations": ["Bentonville", "Arkansas"],
    },
    "XOM": {
        "company": ["ExxonMobil Corporation", "ExxonMobil Corp.", "ExxonMobil"],
        "executives": ["Darren Woods", "Kathryn Mikells"],
        "products": ["Mobil", "Esso", "Synvisc", "Mobil 1"],
        "subsidiaries": ["XTO Energy", "ExxonMobil Chemical", "Imperial Oil"],
        "locations": ["Spring, Texas", "Spring, TX", "The Woodlands"],
    },
    "PG": {
        "company": ["Procter & Gamble Company", "Procter & Gamble Co.", "Procter & Gamble", "P&G"],
        "executives": ["Jon Moeller", "Jonathan R. Moeller", "Andre Schulten"],
        "products": ["Tide", "Pampers", "Gillette", "Crest", "Oral-B", "Olay",
                     "Pantene", "Head & Shoulders", "Ariel", "Febreze",
                     "Charmin", "Bounty", "Downy", "Old Spice", "Always"],
        "subsidiaries": [],
        "locations": ["Cincinnati", "Ohio"],
    },
    "HD": {
        "company": ["Home Depot, Inc.", "Home Depot Inc.", "Home Depot"],
        "executives": ["Ted Decker", "Richard McPhail", "Ann-Marie Campbell"],
        "products": ["THD", "Homer Fund", "Pro Xtra"],
        "subsidiaries": ["The Home Depot", "Interline Brands", "HD Supply"],
        "locations": ["Atlanta", "Georgia", "Marietta"],
    },
    "BAC": {
        "company": ["Bank of America Corporation", "Bank of America Corp.", "Bank of America", "BofA"],
        "executives": ["Brian Moynihan", "Alastair Borthwick"],
        "products": ["Merrill Lynch", "Merrill Edge", "BofA Securities", "Erica"],
        "subsidiaries": ["Merrill Lynch", "U.S. Trust"],
        "locations": ["Charlotte", "North Carolina"],
    },
    "HAS": {
        "company": ["Hasbro, Inc.", "Hasbro Inc.", "Hasbro"],
        "executives": ["Chris Cocks", "Gina Goetter"],
        "products": ["My Little Pony", "Transformers", "Dungeons & Dragons", "Monopoly",
                     "Magic: The Gathering", "Play-Doh", "G.I. Joe", "Peppa Pig",
                     "Power Rangers", "Nerf"],
        "subsidiaries": ["Wizards of the Coast", "eOne"],
        "locations": ["Pawtucket", "Rhode Island"],
    },
    "DRI": {
        "company": ["Darden Restaurants, Inc.", "Darden Restaurants Inc.", "Darden Restaurants", "Darden"],
        "executives": ["Rick Cardenas", "Raj Vennam"],
        "products": ["Olive Garden", "LongHorn Steakhouse", "Cheddar's Scratch Kitchen",
                     "Yard House", "The Capital Grille", "Seasons 52", "Eddie V's",
                     "Bahama Breeze"],
        "subsidiaries": [],
        "locations": ["Orlando", "Florida"],
    },
    "WYNN": {
        "company": ["Wynn Resorts, Limited", "Wynn Resorts, Ltd.", "Wynn Resorts", "Wynn"],
        "executives": ["Craig Billings", "Julie Cameron-Doe"],
        "products": ["Wynn Las Vegas", "Encore", "Wynn Macau", "Wynn Palace"],
        "subsidiaries": ["Wynn Las Vegas LLC", "Wynn Macau Limited"],
        "locations": ["Las Vegas", "Nevada", "Macau"],
    },
    "THS": {
        "company": ["TreeHouse Foods, Inc.", "TreeHouse Foods Inc.", "TreeHouse Foods", "TreeHouse"],
        "executives": ["Steve Oakland", "Pat O'Donnell"],
        "products": ["Private Label", "store brand"],
        "subsidiaries": [],
        "locations": ["Oak Brook", "Illinois"],
    },
    "IBP": {
        "company": ["Installed Building Products, Inc.", "Installed Building Products Inc.", "Installed Building Products", "IBP"],
        "executives": ["Jeff Edwards", "Michael Miller"],
        "products": ["TruTeam"],
        "subsidiaries": ["TruTeam"],
        "locations": ["Columbus", "Ohio"],
    },
    "WTS": {
        "company": ["Watts Water Technologies, Inc.", "Watts Water Technologies Inc.", "Watts Water Technologies", "Watts Water"],
        "executives": ["Robert J. Houghton", "Shashank Patel"],
        "products": ["Watts", "Mueller", "AERCO", "Tekmar", "Dormont"],
        "subsidiaries": ["Mueller Industries", "AERCO International"],
        "locations": ["North Andover", "Massachusetts"],
    },
    "CATY": {
        "company": ["Cathay General Bancorp", "Cathay General Bancorp Inc.", "Cathay"],
        "executives": ["Chang Liu", "Heng W. Chen"],
        "products": ["Cathay Bank"],
        "subsidiaries": ["Cathay Bank"],
        "locations": ["Los Angeles", "California"],
    },
    "TOL": {
        "company": ["Toll Brothers, Inc.", "Toll Brothers Inc.", "Toll Brothers"],
        "executives": ["Douglas C. Yearley Jr.", "Douglas Yearley", "Martin Connor"],
        "products": ["Toll Brothers"],
        "subsidiaries": [],
        "locations": ["Fort Washington", "Pennsylvania"],
    },
    "FNB": {
        "company": ["F.N.B. Corporation", "FNB Corp.", "FNB", "F.N.B."],
        "executives": ["Vincent J. Delie Jr.", "Vincent Delie", "Gary Guerrieri"],
        "products": ["First National Bank", "FNB Direct"],
        "subsidiaries": ["First National Bank of Pennsylvania"],
        "locations": ["Pittsburgh", "Pennsylvania"],
    },
    "NGVC": {
        "company": ["Natural Grocers by Vitamin Cottage, Inc.", "Natural Grocers by Vitamin Cottage",
                    "Natural Grocers", "Vitamin Cottage"],
        "executives": ["Kemper Isely", "Sandra Buffa"],
        "products": ["Natural Grocers", "Vitamin Cottage"],
        "subsidiaries": [],
        "locations": ["Lakewood", "Colorado"],
    },
    "OFLX": {
        "company": ["Omega Flex, Inc.", "Omega Flex Inc.", "Omega Flex"],
        "executives": ["Kevin R. Murphy", "David S. Mueller"],
        "products": ["TracPipe", "CounterStrike", "AutoSnap"],
        "subsidiaries": [],
        "locations": ["Exton", "Pennsylvania"],
    },
    "UTMD": {
        "company": ["Utah Medical Products, Inc.", "Utah Medical Products Inc.", "Utah Medical Products", "UTMD"],
        "executives": ["Kevin L. Cornwell", "Kevin Cornwell", "Craig Hannum"],
        "products": ["Disposa-Hood", "UtaheScope"],
        "subsidiaries": [],
        "locations": ["Midvale", "Utah"],
    },
    "LWAY": {
        "company": ["Lifeway Foods, Inc.", "Lifeway Foods Inc.", "Lifeway Foods", "Lifeway"],
        "executives": ["Julie Smolyansky", "Eric Hanson"],
        "products": ["Lifeway Kefir", "Plantiful", "BioKefir"],
        "subsidiaries": [],
        "locations": ["Morton Grove", "Illinois"],
    },
    "SPOK": {
        "company": ["Spok Holdings, Inc.", "Spok Holdings Inc.", "Spok Holdings", "Spok"],
        "executives": ["Vincent D. Kelly", "Vincent Kelly", "Michael Wallace"],
        "products": ["Spok Care Connect", "Spok Go", "Spok Mobile"],
        "subsidiaries": ["Spok, Inc."],
        "locations": ["Alexandria", "Virginia"],
    },
    "IIIN": {
        "company": ["Insteel Industries, Inc.", "Insteel Industries Inc.", "Insteel Industries", "Insteel"],
        "executives": ["H.O. Woltz III", "Howard Woltz", "Mark Carano"],
        "products": ["PC Strand", "Welded Wire Reinforcement", "WWR"],
        "subsidiaries": [],
        "locations": ["Mount Airy", "North Carolina"],
    },
    "RELL": {
        "company": ["Richardson Electronics, Ltd.", "Richardson Electronics Ltd.", "Richardson Electronics"],
        "executives": ["Edward J. Richardson", "Edward Richardson", "Robert Ben"],
        "products": ["Richardson RFPD", "Power and Microwave Technologies", "Display Systems Group"],
        "subsidiaries": ["Richardson RFPD", "PMT"],
        "locations": ["LaFox", "Illinois", "Carol Stream"],
    },
    "HAFC": {
        "company": ["Hanmi Financial Corporation", "Hanmi Financial Corp.", "Hanmi Financial", "Hanmi"],
        "executives": ["Bonnie Lee", "Anthony Kim"],
        "products": ["Hanmi Bank"],
        "subsidiaries": ["Hanmi Bank"],
        "locations": ["Los Angeles", "California"],
    },
    "CLMT": {
        "company": ["Calumet Specialty Products Partner", "Calumet Specialty Products", "Calumet"],
        "executives": ["Todd Borgmann", "Bruce Fleming"],
        "products": ["Montana Renewables", "Royal Purple", "Bel-Ray"],
        "subsidiaries": ["Montana Renewables LLC"],
        "locations": ["Indianapolis", "Indiana"],
    },
    # ── Full-pilot expansion firms ────────────────────────────────────────────
    "AMZN": {
        "company": ["Amazon.com, Inc.", "Amazon.com Inc.", "Amazon.com", "Amazon"],
        "executives": ["Andy Jassy", "Brian Olsavsky", "Doug Herrington"],
        "products": ["AWS", "Amazon Web Services", "Prime", "Alexa", "Kindle",
                     "Ring", "Whole Foods", "Twitch", "IMDb", "Audible"],
        "subsidiaries": ["Amazon Web Services", "Whole Foods Market", "Twitch", "Ring", "Zappos"],
        "locations": ["Seattle", "Washington", "Bellevue"],
    },
    "GOOGL": {
        "company": ["Alphabet Inc.", "Alphabet", "Google LLC", "Google"],
        "executives": ["Sundar Pichai", "Ruth Porat", "Philipp Schindler"],
        "products": ["Google Search", "YouTube", "Google Cloud", "Android", "Chrome",
                     "Gmail", "Google Maps", "Google Ads", "Google Play", "Waymo",
                     "DeepMind", "Pixel", "Bard", "Gemini"],
        "subsidiaries": ["Google LLC", "YouTube", "Waymo", "DeepMind", "Verily"],
        "locations": ["Mountain View", "California", "Sunnyvale"],
    },
    "KFRC": {
        "company": ["Kforce Inc.", "Kforce"],
        "executives": ["Joseph Liberatore", "David Kelly", "Jeffrey Hackman"],
        "products": ["Kforce Technology", "Kforce Finance and Accounting"],
        "subsidiaries": [],
        "locations": ["Tampa", "Florida"],
    },
    "PLAB": {
        "company": ["Photronics, Inc.", "Photronics Inc.", "Photronics"],
        "executives": ["Frank Lee", "Eric Tang", "Christopher Progler"],
        "products": ["photomasks", "FPD photomasks"],
        "subsidiaries": [],
        "locations": ["Brookfield", "Connecticut"],
    },
    "CLFD": {
        "company": ["Clearfield, Inc.", "Clearfield Inc.", "Clearfield"],
        "executives": ["Cheri Beranek", "Daniel Herzog"],
        "products": ["FieldSmart", "Clearview", "WaveSmart"],
        "subsidiaries": [],
        "locations": ["Brooklyn Park", "Minnesota"],
    },
    "HAYN": {
        "company": ["Haynes International, Inc.", "Haynes International Inc.", "Haynes International", "Haynes"],
        "executives": ["Michael Shor", "Daniel Maudlin"],
        "products": ["HAYNES alloys", "HASTELLOY", "WASPALOY"],
        "subsidiaries": [],
        "locations": ["Kokomo", "Indiana"],
    },
    "MLAB": {
        "company": ["Mesa Laboratories, Inc.", "Mesa Laboratories Inc.", "Mesa Labs", "Mesa Laboratories"],
        "executives": ["Gary Owens", "John Sullivan"],
        "products": ["DataTrace", "Bios", "Surg-O-Clave", "Nusonics"],
        "subsidiaries": ["Butler Scientific", "Bios International"],
        "locations": ["Lakewood", "Colorado"],
    },
    "UFPT": {
        "company": ["UFP Technologies, Inc.", "UFP Technologies Inc.", "UFP Technologies"],
        "executives": ["R. Jeffrey Bailly", "Jeffrey Bailly", "Ronald Lataille"],
        "products": ["Simulated Tissue", "Moulded Fibre"],
        "subsidiaries": [],
        "locations": ["Newburyport", "Massachusetts"],
    },
    "CLIR": {
        "company": ["ClearSign Technologies Corporation", "ClearSign Technologies Corp.", "ClearSign Technologies", "ClearSign"],
        "executives": ["Brian Fike", "Jim Deller"],
        "products": ["ClearSign Core", "Duplex technology"],
        "subsidiaries": [],
        "locations": ["Dallas", "Texas"],
    },
}


def _safe_sub(pattern: str, replacement: str, text: str, flags: int = re.IGNORECASE) -> str:
    """Word-boundary-aware substitution; silently skips bad patterns."""
    try:
        return re.sub(pattern, replacement, text, flags=flags)
    except re.error:
        return text


def anonymize_text(text: str, ticker: str) -> tuple[str, list[str]]:
    """
    Scrub company-identifying strings from filing text.

    Returns (anonymized_text, list_of_replacements_made).
    The replacement log is useful for debugging — check it to ensure nothing
    important slipped through.
    """
    meta = FIRM_METADATA.get(ticker.upper())
    if meta is None:
        logger.warning("No anonymization metadata for %s — returning original text", ticker)
        return text, []

    replacements_made: list[str] = []
    result = text

    # Ticker symbol (whole-word match to avoid replacing inside other words)
    result = _safe_sub(r"\b" + re.escape(ticker.upper()) + r"\b", "[TICKER]", result)
    if result != text:
        replacements_made.append(f"ticker:{ticker}")

    # Company names (longest first to prevent partial matches)
    for name in sorted(meta.get("company", []), key=len, reverse=True):
        new = _safe_sub(re.escape(name), "[COMPANY]", result)
        if new != result:
            replacements_made.append(f"company:{name}")
            result = new

    # Executives
    for exec_name in meta.get("executives", []):
        # Replace full name first, then last name alone (if ≥6 chars to avoid false positives)
        new = _safe_sub(re.escape(exec_name), "[EXECUTIVE]", result)
        if new != result:
            replacements_made.append(f"executive:{exec_name}")
            result = new
        last_name = exec_name.split()[-1]
        if len(last_name) >= 6:
            new = _safe_sub(r"\b" + re.escape(last_name) + r"\b", "[EXECUTIVE]", result)
            if new != result:
                replacements_made.append(f"executive_last:{last_name}")
                result = new

    # Products (longer strings first)
    for product in sorted(meta.get("products", []), key=len, reverse=True):
        new = _safe_sub(re.escape(product), "[PRODUCT]", result)
        if new != result:
            replacements_made.append(f"product:{product}")
            result = new

    # Subsidiaries
    for sub in sorted(meta.get("subsidiaries", []), key=len, reverse=True):
        if not sub:
            continue
        new = _safe_sub(re.escape(sub), "[SUBSIDIARY]", result)
        if new != result:
            replacements_made.append(f"subsidiary:{sub}")
            result = new

    # Locations
    for loc in meta.get("locations", []):
        new = _safe_sub(r"\b" + re.escape(loc) + r"\b", "[LOCATION]", result)
        if new != result:
            replacements_made.append(f"location:{loc}")
            result = new

    return result, replacements_made
