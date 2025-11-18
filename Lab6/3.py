# ---------- GENERATION ONLY (SHORT GENERIC VERSION) ----------

import heapq, math

# probability_fn should be a function: prob(hist3, w) → float

def greedy_generate(prob_fn):
    hist = ("<s>","<s>","<s>")
    out=[]
    for _ in range(30):
        w = max(V, key=lambda x: prob_fn(hist,x))
        if w=="</s>": break
        out.append(w)
        hist=(hist[1],hist[2],w)
    return " ".join(out)

def beam_generate(prob_fn, beam=20):
    beams=[(0.0,("<s>","<s>","<s>"),[])]
    for _ in range(30):
        new=[]
        for logp,hist,sent in beams:
            for w in V:
                p = prob_fn(hist,w)
                if p>0:
                    new.append((logp+math.log(p),(hist[1],hist[2],w),sent+[w]))
        beams=heapq.nlargest(beam,new)
    return " ".join(beams[0][2])

# Example usage:
# greedy_generate(lambda h,w: katz_prob(h,w))
# greedy_generate(lambda h,w: p_kn(h,w))
# beam_generate(lambda h,w: katz_prob(h,w), beam=20)
