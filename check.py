"""Step 5: the project's check. Recompute with different code, in plain Python straight from the downloaded listings file (the csv, gzip, json and re modules, math and loops: no pandas, numpy or scipy):
every count in the audit, the sample, every feature column (rebuilt by hand), the least-squares fits behind every score (normal equations, solved by elimination, on sparse rows), the cross-validated R squared and errors of every model in
the ladder, of each group alone and with each group removed, the coefficients and their robust (HC1) errors, the effects, the location premium and the error tables. The one step that is not rewritten is the gradient-boosted model's fit:
check.py hands the matrix it built by hand to scikit-learn (and to numpy for that call only) and recomputes the scores from the predictions; without scikit-learn installed that part is skipped and said so.

    ./venv/bin/python check.py
"""
import csv
import gzip
import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path

csv.field_size_limit(10 ** 9)
HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
PRICE_TEXT = re.compile(r"\$\d{1,3}(,\d{3})*\.\d{2}")
AMENITIES = [("pool", r"\bpool\b(?! table| view)"), ("hot_tub", r"hot tub"), ("waterfront", r"waterfront"), ("lake_access", r"lake access"), ("ev_charger", r"ev charger"), ("gym", r"\bgym\b"), ("air_conditioning", r"air conditioning"),
             ("workspace", r"dedicated workspace"), ("free_parking", r"free parking on premises"), ("bbq_grill", r"bbq grill"), ("elevator", r"elevator"), ("washer", r"washer")]
GROUPS = ["quote", "size", "type", "place", "host and reviews", "amenities"]
MIN_PLACE = 100
n_ok = 0


def same(label, got, want, rel=1e-9):
    global n_ok
    n_ok += 1
    want = float(want) if want not in ("", "nan") else float("nan")
    if math.isnan(want):
        assert math.isnan(got), f"{label}: recount {got} != results nan"
    else:
        assert abs(got - want) <= rel * max(1.0, abs(want)), f"{label}: recount {got} != results {want}"


def eq(label, got, want):
    global n_ok
    n_ok += 1
    assert got == want, f"{label}: recount {got!r} != results {want!r}"


def table(name):
    return list(csv.DictReader(open(R / f"{name}.csv")))


def quantile(v, q):
    v = sorted(v); pos = (len(v) - 1) * q; lo = math.floor(pos); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def num(t):
    return float(t) if t != "" else None


def solve(A, b):
    """Gaussian elimination with partial pivoting"""
    m = len(b); M = [A[i][:] + [b[i]] for i in range(m)]
    for c in range(m):
        p = max(range(c, m), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]; piv = M[c][c]
        for r in range(c + 1, m):
            f = M[r][c] / piv
            if f:
                Mr, Mc = M[r], M[c]
                for j in range(c, m + 1):
                    Mr[j] -= f * Mc[j]
    x = [0.0] * m
    for i in range(m - 1, -1, -1):
        x[i] = (M[i][m] - sum(M[i][j] * x[j] for j in range(i + 1, m))) / M[i][i]
    return x


def invert(A):
    m = len(A); M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    for c in range(m):
        p = max(range(c, m), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]; piv = M[c][c]; M[c] = [v / piv for v in M[c]]
        for r in range(m):
            if r != c and M[r][c]:
                f = M[r][c]; Mr, Mc = M[r], M[c]
                for j in range(2 * m):
                    Mr[j] -= f * Mc[j]
    return [row[m:] for row in M]


def main():
    rows = list(csv.DictReader(gzip.open(D / "listings.csv.gz", "rt", encoding="utf-8", newline=""))); nb = list(csv.DictReader(open(D / "neighbourhoods.csv", encoding="utf-8")))
    aud = {r["check"]: float(r["value"]) for r in table("audit")}
    def A(name, got):
        same(name, got, aud[name])
    def ymd(t):
        return int(t.replace("-", ""))
    A("rows: listings", len(rows)); A("columns: listings", len(rows[0])); A("rows: neighbourhoods", len(nb)); A("duplicate ids", len(rows) - len({r["id"] for r in rows}))
    sc = {r["last_scraped"] for r in rows}; A("scrape dates", len(sc)); A("first scrape date", ymd(min(sc))); A("last scrape date", ymd(max(sc)))
    text = [r["price"] for r in rows if r["price"] != ""]; A("prices written as text", len(text)); A("prices not in the form $1,234.56", sum(not PRICE_TEXT.fullmatch(t) for t in text)); A("prices blank", sum(r["price"] == "" for r in rows)); A("prices that fail to convert", 0)
    for r in rows:
        r["p"] = float(r["price"].replace("$", "").replace(",", "")) if r["price"] != "" else None
        ci, co = r["price_quote_checkin_date"], r["price_quote_checkout_date"]
        r["nights"] = (date.fromisoformat(co) - date.fromisoformat(ci)).days if ci != "" and co != "" else None
        r["lead"] = (date.fromisoformat(ci) - date.fromisoformat(r["last_scraped"])).days if ci != "" else None; r["weekday"] = date.fromisoformat(ci).weekday() if ci != "" else None
        mn = num(r["minimum_nights"]); r["short"] = mn is not None and mn < 30
    A("listings with a price", sum(r["p"] is not None for r in rows)); A("priced, minimum stay under 30 nights", sum(r["p"] is not None and r["short"] for r in rows))
    S = [r for r in rows if r["p"] is not None and r["short"] and r["nights"] is not None and 1 <= r["nights"] <= 7]; n = len(S)
    A("priced, short minimum stay, quote of 1 to 7 nights (the sample)", n); A("priced, short minimum stay, quote of 8 nights or more", sum(r["p"] is not None and r["short"] and r["nights"] is not None and r["nights"] >= 8 for r in rows))
    A("sample: quote shorter than 1 day or negative lead", sum(r["nights"] < 1 or r["lead"] < 0 for r in S)); A("sample: median days from scrape to check-in", quantile([r["lead"] for r in S], 0.5))
    hc = Counter(r["host_id"] for r in S); A("sample: hosts", len(hc)); A("sample: listings of hosts with more than one listing", sum(v for v in hc.values() if v > 1)); A("sample: most listings of one host", max(hc.values()))
    def bath_text(t):
        if t == "":
            return None
        m = re.search(r"[0-9.]+", t); return float(m.group()) if m else 0.5
    for r in S:
        b = num(r["bathrooms"]); r["bath_all"] = b if b is not None else bath_text(r["bathrooms_text"])
    A("sample: bathrooms blank", sum(r["bathrooms"] == "" for r in S)); A("sample: of which read from the bathrooms text", sum(r["bathrooms"] == "" and r["bath_all"] is not None for r in S)); A("sample: bathrooms still missing", sum(r["bath_all"] is None for r in S))
    A("sample: bedrooms missing", sum(r["bedrooms"] == "" for r in S)); A("sample: beds missing", sum(r["beds"] == "" for r in S)); A("sample: review score missing", sum(r["review_scores_rating"] == "" for r in S)); A("sample: host years missing", sum(r["hosts_time_as_host_years"] == "" for r in S))
    A("sample: superhost field missing", sum(r["host_is_superhost"] == "" for r in S)); A("sample: kinds of property listed", len({r["property_type"] for r in S})); A("sample: zip codes", len({r["neighbourhood_cleansed"] for r in S}))
    A("sample: price quote equals the price", sum(r["price_quote_price_per_night"] != "" and abs(round(float(r["price_quote_price_per_night"]), 2) - r["p"]) <= 0.011 for r in S))

    # the features, rebuilt by hand
    def med(v):
        v = [x for x in v if x is not None]; return quantile(v, 0.5)
    m_bed, m_beds, m_bath = med([num(r["bedrooms"]) for r in S]), med([num(r["beds"]) for r in S]), med([r["bath_all"] for r in S]); m_years, m_score = med([num(r["hosts_time_as_host_years"]) for r in S]), med([num(r["review_scores_rating"]) for r in S])
    kind = [re.sub(r"^(Entire |Private room in |Shared room in |Room in )", "", r["property_type"]).lower() for r in S]; kc = Counter(kind); big = sorted(k for k in kc if kc[k] >= MIN_PLACE and k != "home")
    zc = Counter(r["neighbourhood_cleansed"] for r in S); ref = max(zc, key=lambda z: (zc[z], z)); zips = sorted(z for z in zc if z != ref)
    names = ["nights", "days_ahead"] + [f"checkin_day_{d}" for d in range(1, 7)] + ["log_guests", "bedrooms", "beds", "bathrooms_all", "bedrooms_missing", "beds_missing", "bathrooms_all_missing", "shared_bathroom", "room_private", "room_hotel", "room_shared"] \
        + [f"place_{k.replace(' ', '_')}" for k in big] + ["place_other"] + [f"zip_{z}" for z in zips] + ["superhost", "host_listings", "host_years", "host_verified", "log_reviews", "review_score", "review_score_missing", "log_amenities"] + [a for a, _ in AMENITIES]
    group_of = {}
    for i, c in enumerate(names):
        group_of[c] = ("quote" if i < 8 else "size" if i < 16 else "type" if i < 16 + 3 + len(big) + 1 else "place" if c.startswith("zip_") else "host and reviews" if c in ("superhost", "host_listings", "host_years", "host_verified", "log_reviews", "review_score", "review_score_missing") else "amenities")
    X = []
    for r, kd in zip(S, kind):
        am = [a.lower() for a in json.loads(r["amenities"])]; bd, bs = num(r["bedrooms"]), num(r["beds"]); sc_ = num(r["review_scores_rating"])
        v = [float(r["nights"]), math.log1p(r["lead"])] + [float(r["weekday"] == d) for d in range(1, 7)]
        v += [math.log(float(r["accommodates"])), bd if bd is not None else m_bed, bs if bs is not None else m_beds, r["bath_all"] if r["bath_all"] is not None else m_bath, float(bd is None), float(bs is None), float(r["bath_all"] is None), float("shared" in r["bathrooms_text"].lower())]
        v += [float(r["room_type"] == t) for t in ("Private room", "Hotel room", "Shared room")] + [float(kd == k) for k in big] + [float(kd not in big and kd != "home")] + [float(r["neighbourhood_cleansed"] == z) for z in zips]
        yrs = num(r["hosts_time_as_host_years"])
        v += [float(r["host_is_superhost"] == "t"), math.log1p(float(r["calculated_host_listings_count"])), yrs if yrs is not None else m_years, float(r["host_identity_verified"] == "t"), math.log1p(float(r["number_of_reviews"])), sc_ if sc_ is not None else m_score, float(sc_ is None)]
        v += [math.log(max(len(am), 1))] + [float(any(re.search(rx, x) for x in am)) for _, rx in AMENITIES]
        X.append(v)
    assert len(names) == len(X[0]) == 89
    A("sample: zip code used as the reference", float(ref)); A("sample: kinds of place with their own column", len(big)); A("sample: columns of the model (without the intercept)", len(names)); A("sample: amenity entries read", sum(len(json.loads(r["amenities"])) for r in S))
    gidx = {g: [1 + i for i, c in enumerate(names) if group_of[c] == g] for g in GROUPS}
    for g in GROUPS:
        A(f"columns: {g}", len(gidx[g]))
    price = [r["p"] for r in S]; y = [math.log(p) for p in price]
    pr = table("prices")
    for r in pr:
        sel = [p for p, s in zip(price, S) if r["group"] == "All listings" or s["room_type"] == r["group"]]
        same("prices listings", len(sel), r["listings"]); same("prices median", quantile(sel, 0.5), r["median"]); same("prices q1", quantile(sel, 0.25), r["q1"]); same("prices q3", quantile(sel, 0.75), r["q3"]); same("prices mean", math.fsum(sel) / len(sel), r["mean"])
        same("prices p99", quantile(sel, 0.99), r["p99"]); same("prices max", max(sel), r["max"])

    # least squares: sparse rows, normal equations per fold
    k = 1 + len(names); rowsp = [[(0, 1.0)] + [(1 + i, x) for i, x in enumerate(v) if x != 0.0] for v in X]
    def moments(fold):
        Af = [[[0.0] * k for _ in range(k)] for _ in range(5)]; bf = [[0.0] * k for _ in range(5)]
        for r, yi, f in zip(rowsp, y, fold):
            M = Af[f]; bb = bf[f]
            for a, (i, vi) in enumerate(r):
                bb[i] += vi * yi; Mi = M[i]
                for j, vj in r[a:]:
                    Mi[j] += vi * vj
        return Af, bf
    def fit_oof(cols, fold, Af, bf, TA, Tb):
        pred = [0.0] * n; cols = sorted(cols)
        for f in range(5):
            use = [c for c in cols if TA[c][c] - Af[f][c][c] != 0.0]                     # a column with no training rows has coefficient 0
            Mx = [[(TA[a][b] - Af[f][a][b]) for b in use[p:]] for p, a in enumerate(use)]; m = len(use)
            Mfull = [[0.0] * m for _ in range(m)]
            for p in range(m):
                for q in range(p, m):
                    Mfull[p][q] = Mfull[q][p] = Mx[p][q - p]
            beta = dict(zip(use, solve(Mfull, [Tb[c] - bf[f][c] for c in use])))
            for idx in range(n):
                if fold[idx] == f:
                    pred[idx] = sum(beta.get(j, 0.0) * v for j, v in rowsp[idx])
        return pred
    ybar = math.fsum(y) / n; sst = math.fsum((v - ybar) ** 2 for v in y)
    def score(pred):
        ape = [abs(math.exp(p_) - pr_) / pr_ for p_, pr_ in zip(pred, price)]
        return {"r2": 1 - math.fsum((a - b) ** 2 for a, b in zip(y, pred)) / sst, "mdape": quantile(ape, 0.5), "within10": sum(a <= 0.10 for a in ape) / n, "within25": sum(a <= 0.25 for a in ape) / n, "within50": sum(a <= 0.50 for a in ape) / n}
    fold_h = [int(r["host_id"]) % 5 for r in S]; fold_l = [int(r["id"]) % 5 for r in S]
    out = {}
    for nm, fold in (("by host", fold_h), ("by listing", fold_l)):
        Af, bf = moments(fold); TA = [[sum(Af[f][i][j] for f in range(5)) for j in range(k)] for i in range(k)]; Tb = [sum(bf[f][i] for f in range(5)) for i in range(k)]; out[nm] = (fold, Af, bf, TA, Tb)
    def ols_pred(cols, scheme="by host"):
        fold, Af, bf, TA, Tb = out[scheme]; return fit_oof([0] + cols, fold, Af, bf, TA, Tb)
    sc = {}
    pred_mean = ols_pred([]); sc["no features (the mean)"] = score(pred_mean); cum, cumpred = [], {}
    for g in GROUPS:
        cum = cum + gidx[g]; cumpred[g] = ols_pred(cum); sc[g if g == "quote" else f"+ {g}"] = score(cumpred[g])
    sc["least squares, all columns|by listing"] = score(ols_pred(cum, "by listing"))
    for r in table("scores"):
        if r["model"] == "gradient boosting":
            continue
        key = r["model"] if r["folds"] == "by host" else f"{r['model']}|{r['folds']}"; got = sc[key]; same("scores columns", {"no features (the mean)": 0, "quote": 8}.get(r["model"], len(cum) if "all columns" in r["model"] else {"+ size": 16, "+ type": 27, "+ place": 69, "+ host and reviews": 76, "+ amenities": 89}.get(r["model"], -1)), r["columns"])
        for m_ in ("r2", "mdape", "within10", "within25", "within50"):
            same(f"{key} {m_}", got[m_], r[m_], 1e-7)
    cumr2 = {g: score(cumpred[g])["r2"] for g in GROUPS}; before = 0.0
    for r in table("groups"):
        g = r["group"]; al = score(ols_pred(gidx[g]))["r2"]; lo = cumr2["amenities"] - score(ols_pred([c for c in cum if c not in gidx[g]]))["r2"]
        same(f"{g} columns", len(gidx[g]), r["columns"]); same(f"{g} alone", al, r["alone_r2"], 1e-7); same(f"{g} added", cumr2[g] - before, r["added_r2"], 1e-7); same(f"{g} cumulative", cumr2[g], r["cumulative_r2"], 1e-7); same(f"{g} lost if removed", lo, r["lost_if_removed_r2"], 1e-7); before = cumr2[g]

    # the full fit on every row, with robust (HC1) errors
    fold, Af, bf, TA, Tb = out["by host"]; Mfull = [[TA[min(i, j)][max(i, j)] for j in range(k)] for i in range(k)]; beta = solve(Mfull, Tb); inv = invert(Mfull)
    meat = [[0.0] * k for _ in range(k)]
    for r, yi in zip(rowsp, y):
        e2 = (yi - sum(beta[j] * v for j, v in r)) ** 2
        for a, (i, vi) in enumerate(r):
            Mi = meat[i]
            for j, vj in r[a:]:
                Mi[j] += e2 * vi * vj
    meat = [[meat[min(i, j)][max(i, j)] for j in range(k)] for i in range(k)]
    tmp = [[sum(inv[i][c] * meat[c][j] for c in range(k)) for j in range(k)] for i in range(k)]; se = [math.sqrt(n / (n - k) * sum(tmp[i][c] * inv[c][i] for c in range(k))) for i in range(k)]
    tn = ["intercept"] + names; cf = dict(zip(tn, zip(beta, se)))
    for r in table("coefficients"):
        b_, s_ = cf[r["term"]]; same(f"coefficient {r['term']}", b_, r["coefficient"], 1e-6); same(f"se {r['term']}", s_, r["se_hc1"], 1e-6); eq(f"group {r['term']}", r["group"], "intercept" if r["term"] == "intercept" else group_of[r["term"]])
    for r in table("effects"):
        b_, s_ = cf[r["term"]]; sc_ = float(r["scale"])
        same(f"effect {r['label']}", 100 * (math.exp(sc_ * b_) - 1), r["effect_pct"], 1e-6); same("ci low", 100 * (math.exp(sc_ * (b_ - 1.96 * s_)) - 1), r["ci_low_pct"], 1e-6); same("ci high", 100 * (math.exp(sc_ * (b_ + 1.96 * s_)) - 1), r["ci_high_pct"], 1e-6)

    # the location premium and the errors
    res = [a - b for a, b in zip(y, cumpred["type"])]
    for r in table("location"):
        sel = [e for e, s in zip(res, S) if s["neighbourhood_cleansed"] == r["zip"]]; m = quantile(sel, 0.5)
        same(f"zip {r['zip']} listings", len(sel), r["listings"]); same("median residual", m, r["median_residual"], 1e-7); same("premium", 100 * (math.exp(m) - 1), r["premium_pct"], 1e-6)
    edges = [quantile(price, q) for q in (0.2, 0.4, 0.6, 0.8)]; band = [sum(e < p for e in edges) + 1 for p in price]
    def error_rows(pred, name):
        for r in [x for x in table("errors") if x["model"] == name]:
            b = int(r["price_fifth"]); idx = [i for i in range(n) if band[i] == b]; ape = [abs(math.exp(pred[i]) - price[i]) / price[i] for i in idx]
            same(f"{name} fifth {b} listings", len(idx), r["listings"]); same("price min", min(price[i] for i in idx), r["price_min"]); same("price max", max(price[i] for i in idx), r["price_max"]); same("mdape", quantile(ape, 0.5), r["mdape"], 1e-7)
            same("fit over price", quantile([math.exp(pred[i]) / price[i] for i in idx], 0.5), r["median_fit_over_price"], 1e-7)
    error_rows(cumpred["amenities"], "least squares")
    sm = {r["key"]: float(r["value"]) for r in table("summary")}
    same("summary sample", n, sm["sample"]); same("summary median", quantile(price, 0.5), sm["median_price"]); same("summary mean", math.fsum(price) / n, sm["mean_price"]); same("summary p99", quantile(price, 0.99), sm["price_p99"]); same("summary max", max(price), sm["price_max"])
    same("summary full ols r2", cumr2["amenities"], sm["r2_full_ols_host"], 1e-7)

    # the gradient-boosted model: the library's fit on the matrix built above
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        print("scikit-learn not installed: the gradient-boosted model was not rechecked")
    else:
        Xg = np.array([v + [float(r["latitude"]), float(r["longitude"])] for v, r in zip(X, S)]); yv = np.array(y); gb = {}
        for nm, fold in (("by host", fold_h), ("by listing", fold_l)):
            fa = np.array(fold); pred = np.zeros(n)
            for f in range(5):
                te = fa == f; pred[te] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=0).fit(Xg[~te], yv[~te]).predict(Xg[te])
            gb[nm] = [float(v) for v in pred]
        for r in table("scores"):
            if r["model"] == "gradient boosting":
                got = score(gb[r["folds"]]); same("gbm columns", 91, r["columns"])
                for m_ in ("r2", "mdape", "within10", "within25", "within50"):
                    same(f"gradient boosting {r['folds']} {m_}", got[m_], r[m_], 1e-6)
        error_rows(gb["by host"], "gradient boosting"); same("summary gbm r2", score(gb["by host"])["r2"], sm["r2_gbm_host"], 1e-6)
    print(f"check.py: {n_ok} recomputed numbers agree with results/ (every count, feature, fit and score written out in plain Python; the boosted model's fit is scikit-learn's)")


if __name__ == "__main__":
    main()
