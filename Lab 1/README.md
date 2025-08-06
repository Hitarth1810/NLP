Gujarati Text Tokenization – Regex Based

Introduction-
This project provides a custom-built tokenizer for Gujarati text that can accurately split text into words and sentences, while also calculating performance metrics for each stage.
It is built entirely in Python using regular expressions tailored for the Gujarati script, ensuring correct handling of matras, diacritics, and special symbols.

The goal is to process real-world Gujarati text without breaking important elements like:

URLs & Email addresses
Dates & Numbers with dots
Abbreviations
Ellipses (...)

Project Files-

word_tokenizer_Regex.py – Tokenizes Gujarati text into individual words.

sentence_tokenizer_Regex.py – Segments text into sentences with proper punctuation handling.

gu_words.txt – Output file containing tokenized words (one per line).

gu_sentences.txt – Output file containing segmented sentences (one per line).

gu_words_metrics.txt – Evaluation metrics for the word tokenizer.

gu_sentences_metrics.txt – Evaluation metrics for the sentence tokenizer.

How It Works:

1. Word Tokenizer (word_tokenizer_Regex.py)-

Detects and keeps Gujarati Unicode characters along with their associated matras.
Temporarily replaces protected elements (URLs, emails, numbers with dots) with placeholders.
Splits the remaining text into words.
Restores protected elements back to their original form.
Saves the output in gu_words.txt.

2. Sentence Tokenizer (sentence_tokenizer_Regex.py)-

Identifies sentence boundaries based on punctuation marks like ., !, ?, ।, etc.
Uses placeholder protection to prevent incorrect splitting in:
Abbreviations (e.g., Dr., એલ.સી.બી.)
Ellipses (...)
URLs, emails, dotted numbers
Merges very short sentences (less than 3 words) with the next sentence to avoid unnatural breaks.
Saves the output in gu_sentences.txt.

3. Metrics Calculation-

After tokenization, the scripts generate:
        gu_words_metrics.txt – Word-level statistics such as total tokens, unique tokens, and frequency distribution.

        gu_sentences_metrics.txt – Sentence-level statistics including sentence count, average length, and shortest/longest sentence.


Features & Highlights:

Gujarati Script Awareness – Correctly handles matras and diacritic marks.
Edge Case Protection – Prevents accidental splitting of important text elements.
Custom Regex Patterns – Designed specifically for Gujarati text structure.
Post-processing Merging – Eliminates unnatural short sentence fragments.
Metrics Reporting – Helps evaluate tokenizer accuracy and performance.

Usage-

1. Place your Gujarati text file in the working directory.
2. Run:
python word_tokenizer_Regex.py
python sentence_tokenizer_Regex.py
3. Check:
gu_words.txt for word tokens
gu_sentences.txt for sentence tokens
Metric files for analysis

Summary:
This tokenizer was built through extensive regex refinement and testing on real Gujarati text.
It aims to offer a reliable and accurate tool for natural language processing tasks involving Gujarati.

