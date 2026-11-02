"""One streaming pass over the 7.7 GB news export.

Writes
  article_index.csv  every article: id, date, newspaper, length, climate-term count, rule label
  candidates.csv     articles with >= 1 climate term, with the 500-character focus window
  noncandidate_sample.csv  random articles with no climate term (filter-recall check)
"""
import csv
import random
from collections import Counter
from pathlib import Path
from .common import utc_now, write_json
from .lexicon import CLIMATE_RE, focus_window, rule_label
from .text import NEWSPAPERS, iter_news, light_clean, parse_article

INDEX_FIELDS = ["article_id", "date", "newspaper", "chars", "climate_hits", "rule_label"]
CANDIDATE_FIELDS = ["article_id", "date", "newspaper", "title", "chars", "climate_hits", "top_terms", "rule_label", "focus_text"]


def build_corpus(path, out, start="2000-01-01", end="2022-12-31", limit=None,
                 noncandidate_sample=400, seed=666):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    reservoir, seen_non = [], 0
    stats = Counter()
    with (out / "article_index.csv").open("w", encoding="utf-8-sig", newline="") as fi, \
         (out / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as fc:
        wi = csv.DictWriter(fi, INDEX_FIELDS)
        wc = csv.DictWriter(fc, CANDIDATE_FIELDS)
        wi.writeheader()
        wc.writeheader()
        for i, row in iter_news(path, limit):
            a = parse_article(i, row, include_text=False, keep_parts=True)
            if not start <= a["date"] <= end or a["newspaper"] not in NEWSPAPERS:
                stats["out_of_scope"] += 1
                continue
            title, body = light_clean(a["title"]), light_clean(a["body"])
            terms = CLIMATE_RE.findall(title + body)
            label = rule_label(title + body) if terms else 0
            wi.writerow({"article_id": a["article_id"], "date": a["date"], "newspaper": a["newspaper"],
                         "chars": len(body), "climate_hits": len(terms), "rule_label": label})
            stats["articles"] += 1
            if terms:
                stats["candidates"] += 1
                top = ";".join(f"{t}:{n}" for t, n in Counter(terms).most_common(5))
                wc.writerow({"article_id": a["article_id"], "date": a["date"], "newspaper": a["newspaper"],
                             "title": title, "chars": len(body), "climate_hits": len(terms), "top_terms": top,
                             "rule_label": label, "focus_text": focus_window(title, body)})
            else:
                seen_non += 1
                j = len(reservoir) if len(reservoir) < noncandidate_sample else rng.randrange(seen_non)
                if j < noncandidate_sample:
                    item = {"article_id": a["article_id"], "date": a["date"], "newspaper": a["newspaper"],
                            "title": title, "chars": len(body), "climate_hits": 0, "top_terms": "",
                            "rule_label": 0, "focus_text": focus_window(title, body, lead=500)}
                    if j == len(reservoir):
                        reservoir.append(item)
                    else:
                        reservoir[j] = item
            if stats["articles"] % 100_000 == 0:
                print(f"scanned {stats['articles']:,} articles, {stats['candidates']:,} candidates", flush=True)
    with (out / "noncandidate_sample.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, CANDIDATE_FIELDS)
        w.writeheader()
        w.writerows(reservoir)
    summary = {"generated_at_utc": utc_now(), "source": Path(path).name, "start": start, "end": end,
               "limit": limit, **stats, "candidate_share": stats["candidates"] / max(stats["articles"], 1),
               "noncandidate_sample": len(reservoir), "seed": seed}
    write_json(out / "corpus_summary.json", summary)
    return summary
