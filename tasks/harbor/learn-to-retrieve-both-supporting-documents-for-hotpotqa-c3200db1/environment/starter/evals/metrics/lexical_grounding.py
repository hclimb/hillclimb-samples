"""Lexical grounding metric — fraction of an answer's content words that are literally
copied from (present in) the gold document. Python port of `groundAnswer` in
scripts/analysis/qa_compare_viewer.html.

A token counts as grounded if it is either (a) part of a >=2-token span copied verbatim
from the doc, or (b) a single non-stopword that appears anywhere in the doc. Stopwords are
excluded from both numerator and denominator. score = grounded content words / total
content words (None if the answer has no content words).

This is a surface/faithfulness proxy, NOT correctness: a hallucinated number won't be in
the doc -> ungrounded -> lowers the score; a correct-but-paraphrased answer also scores low.
Report alongside llm_judge_accuracy (correctness).
"""
import re

STOPWORDS = set((
    "a,an,the,is,are,was,were,be,been,being,of,to,in,on,for,and,or,"
    "as,at,by,with,from,that,this,these,those,it,its,he,she,they,them,his,her,their,you,"
    "your,i,we,us,our,not,no,do,does,did,have,has,had,will,would,can,could,should,may,"
    "might,must,shall,than,then,so,if,but,about,into,over,after,before,between,under,"
    "also,such,which,who,whom,what,when,where,why,how,all,any,both,each,few,more,most,"
    "other,some,only,own,same,too,very,just,there,here,up,down,out,off,again,once,s,t"
).split(","))

_TOK = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def _tokens(text):
    return [m.group(0).lower().replace("'", "") for m in _TOK.finditer(text or "")]


def ground_score(answer, doc):
    """Return the lexical grounding score (grounded content words / content words), or None."""
    a = _tokens(answer)
    d = _tokens(doc)
    if not a or not d:
        return None
    pos = {}
    for i, w in enumerate(d):
        pos.setdefault(w, []).append(i)
    matched = [False] * len(a)
    i = 0
    while i < len(a):
        best = 0
        for p in pos.get(a[i], ()):  # greedy longest verbatim span starting at answer[i]
            ln = 0
            while i + ln < len(a) and p + ln < len(d) and a[i + ln] == d[p + ln]:
                ln += 1
            if ln > best:
                best = ln
        if best >= 2 or (best == 1 and a[i] not in STOPWORDS):
            for k in range(best):
                matched[i + k] = True
            i += best
        else:
            i += 1
    num = den = 0
    for w, m in zip(a, matched):
        if w not in STOPWORDS:
            den += 1
            if m:
                num += 1
    return (num / den) if den else None


def _answer_and_doc(r):
    ans = r.get("generated_answer") or r.get("generated") or ""
    doc = r.get("doc") or r.get("document") or r.get("pos_doc") or ""
    return ans, doc


def lexical_grounding(results, **kwargs):
    """Metric entry point: per-sample lexical grounding scores (None where undefined)."""
    return [ground_score(*_answer_and_doc(r)) for r in results]
