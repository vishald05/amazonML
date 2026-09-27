"""Text normalization functions for business names and addresses."""

import re
import unicodedata
from unidecode import unidecode
from typing import Set


# Legal suffix patterns to normalize
LEGAL_SUFFIXES = {
    'incorporated', 'inc', 'corporation', 'corp', 'company', 'co',
    'limited', 'ltd', 'llc', 'llp', 'lp', 'plc', 'pllc',
    'private', 'pvt', 'public', 'pub',
    'proprietorship', 'proprietor', 'prop',
    'partnership', 'partners',
    'enterprises', 'enterprise', 'ent',
    'group', 'grp', 'holdings', 'holding',
    'services', 'service', 'svc',
    'solutions', 'solution', 'soln',
    'technologies', 'technology', 'tech',
    'international', 'intl', 'global',
    'प्राइवेट', 'लिमिटेड', 'प्रा', 'लि',
}

# Common abbreviations
ABBREVIATION_MAP = {
    'street': 'st',
    'road': 'rd',
    'avenue': 'ave',
    'boulevard': 'blvd',
    'drive': 'dr',
    'lane': 'ln',
    'court': 'ct',
    'circle': 'cir',
    'place': 'pl',
    'square': 'sq',
    'apartment': 'apt',
    'suite': 'ste',
    'building': 'bldg',
    'floor': 'fl',
    'unit': 'unit',
    'number': 'no',
    'north': 'n',
    'south': 's',
    'east': 'e',
    'west': 'w',
    'post office': 'po',
    'private limited': 'pvt ltd',
}


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text."""
    if not text:
        return ""
    # NFC normalization
    text = unicodedata.normalize('NFC', text)
    return text


def normalize_basic(text: str) -> str:
    """
    Basic normalization: lowercase, whitespace, punctuation.
    Preserves numbers and most structure.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Normalize unicode
    text = normalize_unicode(text)

    # Replace common punctuation with spaces
    text = re.sub(r'[,;:|]', ' ', text)

    # Normalize ampersand and 'and'
    text = re.sub(r'\s+&\s+', ' and ', text)
    text = re.sub(r'\s+\+\s+', ' and ', text)

    # Remove other punctuation but keep apostrophes in words
    text = re.sub(r'[^\w\s\'-]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def normalize_name(business_name: str) -> str:
    """
    Normalize business name.

    Strategy:
    - Basic normalization (lowercase, punctuation, whitespace)
    - Transliteration for non-Latin scripts
    - Legal suffix handling (soft normalization)
    """
    if not business_name:
        return ""

    # Basic normalization
    name = normalize_basic(business_name)

    # Remove leading/trailing special markers
    name = re.sub(r'^[-<>»«]+\s*', '', name)
    name = re.sub(r'\s*[-<>»«]+$', '', name)

    # Handle common patterns like "xyz.com" -> "xyz"
    name = re.sub(r'\.com$', '', name)
    name = re.sub(r'\.in$', '', name)

    # Normalize multiple dots/dashes
    name = re.sub(r'\.{2,}', ' ', name)
    name = re.sub(r'-{2,}', ' ', name)

    # Final whitespace cleanup
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def normalize_name_aggressive(business_name: str) -> str:
    """
    More aggressive name normalization for matching.
    Removes common legal suffixes.
    """
    name = normalize_name(business_name)

    if not name:
        return ""

    # Tokenize
    tokens = name.split()

    # Remove legal suffixes from end
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return ' '.join(tokens)


def normalize_address(address: str) -> str:
    """
    Normalize address.

    Strategy:
    - Basic normalization
    - Preserve numbers (house numbers, PIN codes, unit numbers)
    - Standard abbreviations
    """
    if not address:
        return ""

    # Basic normalization
    addr = normalize_basic(address)

    # Standardize common abbreviations
    for full, abbr in ABBREVIATION_MAP.items():
        # Word boundary matching
        addr = re.sub(r'\b' + full + r'\b', abbr, addr)

    # Normalize "unit X" / "apt X" patterns
    addr = re.sub(r'\bunit\s+', 'unit ', addr)
    addr = re.sub(r'\bapt\s+', 'apt ', addr)
    addr = re.sub(r'\bste\s+', 'ste ', addr)

    # Clean up
    addr = re.sub(r'\s+', ' ', addr).strip()

    return addr


def normalize_country(country: str) -> str:
    """Normalize country string."""
    if not country:
        return ""
    return country.strip().lower()


def get_tokens(text: str) -> Set[str]:
    """Get token set from normalized text."""
    if not text:
        return set()
    return set(text.split())


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Get character n-grams from text."""
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def extract_numbers(text: str) -> Set[str]:
    """Extract all numeric tokens from text."""
    if not text:
        return set()
    return set(re.findall(r'\d+', text))


def normalize_transliterate(text: str) -> str:
    """
    Transliterate non-Latin text to ASCII.
    Use for fuzzy matching across scripts.
    """
    if not text:
        return ""
    try:
        return unidecode(text).lower()
    except:
        return text.lower()
