from urllib.parse import urlparse
import ipaddress
import math
import string


def calculate_entropy(text):
    if not text:
        return 0

    frequency = {}

    for char in text:
        frequency[char] = frequency.get(char, 0) + 1

    entropy = 0

    for count in frequency.values():
        probability = count / len(text)
        entropy -= probability * math.log2(probability)

    return entropy


def extract_url_features(url):

    url = str(url).strip()

    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # ==============================
    # IP ADDRESS
    # ==============================

    try:
        ipaddress.ip_address(hostname)
        is_ip = 1
    except ValueError:
        is_ip = 0

    # ==============================
    # HOSTNAME
    # ==============================

    hostname_parts = hostname.split(".")

    subdomain_count = max(len(hostname_parts) - 2, 0)

    tld = hostname_parts[-1] if len(hostname_parts) >= 2 else ""

    hostname_hyphen_count = hostname.count("-")

    hostname_special_count = sum(
        not c.isalnum() and c != "." and c != "-"
        for c in hostname
    )

    hostname_entropy = calculate_entropy(hostname)

    # ==============================
    # HOSTNAME CHARACTER FEATURES
    # ==============================

    hostname_lower = hostname.lower()

    hostname_char_features = {}

    for char in string.ascii_lowercase:
        hostname_char_features[
            f"hostname_{char}_count"
        ] = hostname_lower.count(char)

    # ==============================
    # SUSPICIOUS KEYWORDS
    # ==============================

    suspicious_keywords = [
        "login",
        "signin",
        "verify",
        "verification",
        "account",
        "secure",
        "update",
        "confirm",
        "password",
        "bank",
        "payment",
        "paypal",
        "wallet",
        "invoice",
        "free",
        "bonus",
        "gift",
        "security"
    ]

    url_lower = url.lower()

    suspicious_keyword_count = sum(
        keyword in url_lower
        for keyword in suspicious_keywords
    )

    # ==============================
    # RATIOS
    # ==============================

    total_length = len(url)

    digit_count = sum(c.isdigit() for c in url)

    letter_count = sum(c.isalpha() for c in url)

    digit_ratio = (
        digit_count / total_length
        if total_length > 0
        else 0
    )

    letter_ratio = (
        letter_count / total_length
        if total_length > 0
        else 0
    )

    path_digit_count = sum(
        c.isdigit() for c in path
    )

    path_digit_ratio = (
        path_digit_count / len(path)
        if len(path) > 0
        else 0
    )

    # ==============================
    # SUSPICIOUS TLD
    # ==============================

    suspicious_tlds = {
        "tk",
        "ml",
        "ga",
        "cf",
        "gq",
        "top",
        "xyz",
        "click",
        "link",
        "work"
    }

    suspicious_tld_flag = (
        1 if tld.lower() in suspicious_tlds else 0
    )

    # ==============================
    # URL FEATURES
    # ==============================

    features = {

        # Basic
        "url_length": len(url),
        "hostname_length": len(hostname),
        "is_ip": is_ip,

        # URL structure
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_symbol_count": url.count("@"),
        "https_flag": 1 if parsed.scheme == "https" else 0,

        # Character counts
        "digit_count": digit_count,
        "letter_count": letter_count,
        "special_char_count": sum(
            not c.isalnum() for c in url
        ),

        # URL symbols
        "slash_count": url.count("/"),
        "question_mark_count": url.count("?"),
        "equals_count": url.count("="),
        "ampersand_count": url.count("&"),
        "percent_count": url.count("%"),

        # Path/query
        "path_length": len(path),
        "query_length": len(query),
        "path_segment_count": len(
            [x for x in path.split("/") if x]
        ),

        # Domain
        "subdomain_count": subdomain_count,
        "tld_length": len(tld),
        "hostname_digit_count": sum(
            c.isdigit() for c in hostname
        ),
        "hostname_letter_count": sum(
            c.isalpha() for c in hostname
        ),
        "hostname_hyphen_count": hostname_hyphen_count,
        "hostname_special_count": hostname_special_count,

        # Entropy
        "url_entropy": calculate_entropy(url),
        "hostname_entropy": hostname_entropy,

        # Keywords
        "suspicious_keyword_count": suspicious_keyword_count,

        # Flags
        "www_flag": 1 if hostname.startswith("www.") else 0,
        "long_url_flag": 1 if len(url) > 75 else 0,
        "suspicious_tld_flag": suspicious_tld_flag,

        # Ratios
        "digit_ratio": digit_ratio,
        "letter_ratio": letter_ratio,
        "path_digit_ratio": path_digit_ratio
    }

    # Add hostname character features
    features.update(hostname_char_features)

    return features


if __name__ == "__main__":

    test_url = "https://example.com/login?id=123"

    result = extract_url_features(test_url)

    print("\nURL:", test_url)

    print("\nExtracted Features:")

    for name, value in result.items():
        print(f"{name}: {value}")

    print("\nTotal features:", len(result))