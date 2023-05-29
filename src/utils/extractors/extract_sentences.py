def extract_sentences(head: str, tail: str, corpus: list, num_extract=2) -> list:
    sentences = []
    for sentence in corpus:
        if head in sentence:
            if tail in sentence:
                sentences.append(sentence)
                if len(sentences) == num_extract:
                    break
    return sentences
