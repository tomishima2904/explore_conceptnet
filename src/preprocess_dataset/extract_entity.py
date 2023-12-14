import re


def extract_entity(uri: str) -> tuple:
    pattern = r"/c/([^/]+)/([^/]+)"
    match = re.match(pattern, uri)
    if match:
        lang, entity_str = match.groups()
    else:
        lang, entity_str = None, None
    return lang, entity_str


if __name__ == "__main__":
    strings = [
        "/c/ja/hoge/guoo/hoge",
        "/c/ja/hoge",
        "/c/ja/fuga/hei",
        "/c/ja/fuga"
    ]
    for string in strings:
        lang, entity = extract_entity(string)
        print(f"URI: {string}, Lang: {lang}, Entity: {entity}")
