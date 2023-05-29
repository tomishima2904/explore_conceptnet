# encoding: utf8
from __future__ import unicode_literals
import re
import unicodedata


def _unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s


def _remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def _remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = _remove_space_between(blocks, blocks, s)
    s = _remove_space_between(blocks, basic_latin, s)
    s = _remove_space_between(basic_latin, blocks, s)
    return s


def normalize_neologd(s):
    s = s.strip()
    s = _unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = _remove_extra_spaces(s)
    s = _unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s


def subtract_hashtag(s):
    """ ##を削除するための関数
    wikipediaのダンプデータは entity (リンク) が `##xx##` のように # で囲まれているので、それを除きます。
    例. ##大鰐テレビ中継局##(おおわにてれびちゅうけいきょく) → 大鰐テレビ中継局(おおわにてれびちゅうけいきょく)
    """
    pattern = r"##(.*?)##"
    s = re.sub(pattern, r"\1", s)
    return s


if __name__ == "__main__":
    raw_text = "##大鰐テレビ中継局## （ おおわ に てれ びちゅう けいきょく ） は 、 ##青森県## ##南津軽郡## ##大鰐町## に 設置 さ れ て いる テレビ ##中継局## で ある 。"
    normalized_text = normalize_neologd(raw_text)
    print(subtract_hashtag(normalized_text))
