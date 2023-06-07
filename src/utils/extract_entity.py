import re


def extract_entity(lang: str, uri: str) -> str:
    pattern = r"/c/" + re.escape(lang) + r"/([^/]+)"
    match = re.search(pattern, uri)
    if match:
        entity_str = match.group(1)
    else:
        entity_str = None
    return entity_str


if __name__ == "__main__":
    strings = [
    "/c/ja/hoge/guoo/hoge",
    "/c/ja/hoge",
    "/c/ja/fuga/hei",
    "/c/ja/fuga"
    ]
    for string in strings:
        print(extract_entity("ja", string))
