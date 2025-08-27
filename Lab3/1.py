from collections import defaultdict
import math

BRANCH_THRESHOLD = 15  # same hyperparameter

class Node:
    def __init__(self):
        self.ch = {}   # children dictionary
        self.cnt = 0   # number of words passing through

def insert_word(root, w: str):
    n = root
    n.cnt += 1
    for c in w:
        if c not in n.ch:
            n.ch[c] = Node()
        n = n.ch[c]
        n.cnt += 1

def best_split(root, w: str):
    """Return tuple (best_index, best_score, best_support)."""
    n = root
    best_i = -1
    best_score = 0.0
    best_support = 0
    for i, c in enumerate(w):
        if c not in n.ch:
            break
        n = n.ch[c]
        branching = len(n.ch)
        if branching < BRANCH_THRESHOLD:
            continue
        max_child = max((child.cnt for child in n.ch.values()), default=0)
        if n.cnt <= 0:
            continue
        frac = 1.0 - float(max_child) / float(n.cnt)
        score = frac * branching
        # prefer later split if scores are close
        if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and i > best_i):
            best_i = i
            best_score = score
            best_support = n.cnt
    return best_i, best_score, best_support

def main():
    path = r"brown_nouns.txt"

    with open(path, "r") as fin:
        words = [line.strip().lower() for line in fin if line.strip()]

    pref = Node()
    suf = Node()

    # insert words
    for w in words:
        insert_word(pref, w)
        rw = w[::-1]
        insert_word(suf, rw)

    pref_ofs = open("prefix_out.txt", "w")
    suf_ofs = open("suffix_out.txt", "w")

    pref_count = suf_count = 0
    pref_score_sum = suf_score_sum = 0.0

    for w in words:
        # prefix trie split
        ip, sp, support_p = best_split(pref, w)
        stem_p, sfx_p = "", ""
        score_p, support_pref = sp, support_p
        if ip != -1:
            stem_p = w[:ip + 1]
            sfx_p = w[ip + 1:]
            if len(stem_p) < 2 or not sfx_p:
                stem_p, sfx_p, score_p, support_pref = w, "", 0, 0
        else:
            stem_p, sfx_p = w, ""

        if sfx_p:
            pref_ofs.write(f"{w}={stem_p}+{sfx_p}  # score={score_p} support={support_pref}\n")
            pref_count += 1
            pref_score_sum += score_p
        else:
            pref_ofs.write(f"{w}={w}+  # nosplit\n")

        # suffix trie split
        rw = w[::-1]
        ir, sr, support_s = best_split(suf, rw)
        stem_s, sfx_s = "", ""
        score_s, support_suf = sr, support_s
        if ir != -1:
            rev = rw[:ir + 1]
            sfx_s = rev[::-1]
            stem_s = w[:len(w) - len(sfx_s)]
            if len(stem_s) < 2 or not sfx_s:
                stem_s, sfx_s, score_s, support_suf = w, "", 0, 0
        else:
            stem_s, sfx_s = w, ""

        if sfx_s:
            suf_ofs.write(f"{w}={stem_s}+{sfx_s}  # score={score_s} support={support_suf}\n")
            suf_count += 1
            suf_score_sum += score_s
        else:
            suf_ofs.write(f"{w}={w}+  # nosplit\n")

    pref_ofs.close()
    suf_ofs.close()

    # decide winner
    winner = "prefix"
    if suf_count > pref_count or (suf_count == pref_count and suf_score_sum > pref_score_sum):
        winner = "suffix"

    chosen_fname = "prefix_out.txt" if winner == "prefix" else "suffix_out.txt"
    with open(chosen_fname, "r") as chosen_in, open("trie_q1_output.txt", "w") as final_ofs:
        for l in chosen_in:
            final_ofs.write(l)

    print(f"written prefix_out={pref_count} suffix_out={suf_count} winner={winner}")

if __name__ == "__main__":
    main()
