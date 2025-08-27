import re

def tokenize_gujarati_sentences(content):
    
    pattern_date = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
    pattern_url = r'https?://[^\s]+(\.\w+)'  # Avoid matching URLs ending with a dot
    pattern_email = r'[\w\.-]+@(\w+\.)+(com|in|org)'

    short_forms = [
        r'Dr\.', r'Mr\.', r'Mrs\.', r'Ms\.', r'Prof\.', r'Sr\.', r'Jr\.', r'St\.', r'vs\.', r'etc\.', r'e\.g\.', r'i\.e\.', r'a\.m\.', r'p\.m\.',
        r'એલ\.સી\.બી\.', r'પી\.એસ\.આઇ\.', r'શ્રી\.', r'શ્રીમતી\.', r'કું\.', r'શ્રીમ\.', r'ડૉ\.', r'પ્રો\.', r'સ્વ\.'
    ]
    pattern_abbr = r'(' + '|'.join(short_forms) + r')'

    num_with_dot = r'(?:\d+|[\u0AE6-\u0AEF]+)\.'

   
    protected_pattern = f"({pattern_url}|{pattern_email}|{pattern_date}|{pattern_abbr}|{num_with_dot}|\.\.\.)"


    split_pattern = r'([\.!?।\u0964])\s+'

    parts = re.split(split_pattern, content)

    sentence_list = []
    for idx in range(0, len(parts) - 1, 2):
        segment = parts[idx] + parts[idx + 1] if idx + 1 < len(parts) else parts[idx]
        segment = segment.strip()
        if segment:
            sentence_list.append(segment)

    # Handle leftover text
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentence_list.append(parts[-1].strip())

    # Merge very short trailing segments into the previous sentence
    merged_sentences = []
    for sent in sentence_list:
        if merged_sentences and len(sent.split()) < 3:
            merged_sentences[-1] = merged_sentences[-1].rstrip() + ' ' + sent
        else:
            merged_sentences.append(sent)

    return merged_sentences


if __name__ == "__main__":
    with open(r"Lab1\gu.txt", encoding="utf-8") as file:
        raw_text = file.read()

    results = tokenize_gujarati_sentences(raw_text)

    with open("gu_sentences.txt", "w", encoding="utf-8") as output_file:
        for line in results:
            output_file.write(line + "\n")

    total_sents = len(results)
    total_words = sum(len(s.split()) for s in results)
    total_chars = sum(len(s) for s in results)
    words_per_sent = total_words / total_sents if total_sents else 0
    avg_char_word = total_chars / total_words if total_words else 0
    all_tokens = [word for s in results for word in s.split()]
    ttr_value = len(set(all_tokens)) / total_words if total_words else 0

    with open("gu_sentences_metrics.txt", "w", encoding="utf-8") as metrics_file:
        metrics_file.write(f"Total sentences: {total_sents}\n")
        metrics_file.write(f"Total words: {total_words}\n")
        metrics_file.write(f"Total characters: {total_chars}\n")
        metrics_file.write(f"Words per sentence: {words_per_sent:.2f}\n")
        metrics_file.write(f"Average characters per word: {avg_char_word:.2f}\n")
        metrics_file.write(f"Type-Token Ratio (TTR): {ttr_value:.4f}\n")
