"""Steps 2-4: audit and clean the Austin listings, then ask what explains the price a listing is quoted.

    ./venv/bin/python analyze.py
        -> results/audit.csv  prices.csv  scores.csv  groups.csv  coefficients.csv  effects.csv  location.csv  errors.csv  summary.csv

Sample: listings with a price, a minimum stay under 30 nights and a price quote for a stay of 1 to 7 nights (8,425 of 11,295). The price is the quote's total divided by its nights, quoted for one particular stay.
Target: the natural log of the price. Features, in six groups: the quote (nights, days ahead, weekday), size (guests, bedrooms, beds, bathrooms), type (room type, kind of place), place (zip code), host and reviews, and amenities.
Missing numbers are replaced by the sample median, with a flag. Scores come from 5-fold cross-validation with folds made of hosts (every listing of a host in the same fold, so a host's other listings never help predict a listing),
and, for contrast, with folds made of listings. R squared is 1 - sum of squared errors / sum of squares around the mean of the log price; the error of a prediction exp(fit) is |exp(fit) - price| / price.
Models: ordinary least squares on the log price (fitted by least squares, errors robust to non-constant variance: HC1) and a gradient-boosted tree model (scikit-learn, fixed settings, no randomness) given the same columns plus latitude and longitude.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).parent
D, R = HERE / "data", HERE / "results"
PRICE_TEXT = r"^\$\d{1,3}(,\d{3})*\.\d{2}$"
AMENITIES = [("pool", r"\bpool\b(?! table| view)"), ("hot_tub", r"hot tub"), ("waterfront", r"waterfront"), ("lake_access", r"lake access"), ("ev_charger", r"ev charger"), ("gym", r"\bgym\b"), ("air_conditioning", r"air conditioning"),
             ("workspace", r"dedicated workspace"), ("free_parking", r"free parking on premises"), ("bbq_grill", r"bbq grill"), ("elevator", r"elevator"), ("washer", r"washer")]
GROUPS = ["quote", "size", "type", "place", "host and reviews", "amenities"]
GBM = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=0)
MIN_PLACE = 100          # a kind of place gets its own column if at least this many sample listings are of it
MIN_ZIP = 30             # a zip code is mapped if at least this many sample listings are in it


def bathrooms(text):
    if not isinstance(text, str):
        return np.nan
    m = re.search(r"[0-9.]+", text)
    return float(m.group()) if m else 0.5            # "Half-bath"


def kind_of_place(t):
    return re.sub(r"^(Entire |Private room in |Shared room in |Room in )", "", t).lower()


def build(S):
    """-> the feature columns, as a DataFrame, in group order, and which group each belongs to"""
    n = len(S); cols, group_of = {}, {}
    def put(name, values, group):
        cols[name] = np.asarray(values, float); group_of[name] = group
    put("nights", S["nights"], "quote"); put("days_ahead", np.log1p(S["lead"]), "quote")
    for d in range(1, 7):
        put(f"checkin_day_{d}", S["weekday"] == d, "quote")
    put("log_guests", np.log(S["accommodates"].astype(float)), "size")
    for c in ("bedrooms", "beds", "bathrooms_all"):
        put(c, S[c].fillna(S[c].median()), "size")
    for c in ("bedrooms", "beds", "bathrooms_all"):
        put(f"{c}_missing", S[c].isna(), "size")
    put("shared_bathroom", S["bathrooms_text"].fillna("").str.contains("shared", case=False), "size")
    for r in ("Private room", "Hotel room", "Shared room"):
        put(f"room_{r.split()[0].lower()}", S["room_type"] == r, "type")
    kind = S["property_type"].map(kind_of_place); counts = kind.value_counts(); big = sorted(k for k in counts.index if counts[k] >= MIN_PLACE and k != "home")
    for k in big:
        put(f"place_{k.replace(' ', '_')}", kind == k, "type")
    put("place_other", ~kind.isin(big + ["home"]), "type")
    zips = S["neighbourhood_cleansed"].astype(str); zc = zips.value_counts(); ref = zc.index[0]
    for z in sorted(zc.index):
        if z != ref:
            put(f"zip_{z}", zips == z, "place")
    put("superhost", S["host_is_superhost"] == "t", "host and reviews"); put("host_listings", np.log1p(S["calculated_host_listings_count"]), "host and reviews")
    put("host_years", S["hosts_time_as_host_years"].fillna(S["hosts_time_as_host_years"].median()), "host and reviews"); put("host_verified", S["host_identity_verified"] == "t", "host and reviews")
    put("log_reviews", np.log1p(S["number_of_reviews"]), "host and reviews"); put("review_score", S["review_scores_rating"].fillna(S["review_scores_rating"].median()), "host and reviews"); put("review_score_missing", S["review_scores_rating"].isna(), "host and reviews")
    am = S["amenities"].map(lambda s: [a.lower() for a in json.loads(s)])
    put("log_amenities", np.log(am.map(len).clip(lower=1)), "amenities")
    for name, rx in AMENITIES:
        put(name, am.map(lambda a: any(re.search(rx, x) for x in a)), "amenities")
    return pd.DataFrame(cols), group_of, ref, big


def ols(X, y, tr):
    beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    return beta


def oof_ols(X, y, fold):
    pred = np.zeros(len(y))
    for f in range(5):
        te = fold == f; pred[te] = X[te] @ ols(X, y, ~te)
    return pred


def oof_gbm(X, y, fold):
    pred = np.zeros(len(y))
    for f in range(5):
        te = fold == f; pred[te] = HistGradientBoostingRegressor(**GBM).fit(X[~te], y[~te]).predict(X[te])
    return pred


def score(y, price, pred):
    ape = np.abs(np.exp(pred) - price) / price
    return {"r2": 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), "mdape": float(np.median(ape)), "within10": float((ape <= 0.10).mean()), "within25": float((ape <= 0.25).mean()), "within50": float((ape <= 0.50).mean())}


def main():
    R.mkdir(exist_ok=True)
    L = pd.read_csv(D / "listings.csv.gz", low_memory=False); N = pd.read_csv(D / "neighbourhoods.csv")
    audit = []
    def add(name, value):
        audit.append((name, value))
    add("rows: listings", len(L)); add("columns: listings", L.shape[1]); add("rows: neighbourhoods", len(N)); add("duplicate ids", int(L["id"].duplicated().sum()))
    add("scrape dates", int(L["last_scraped"].nunique())); add("first scrape date", int(L["last_scraped"].min().replace("-", ""))); add("last scrape date", int(L["last_scraped"].max().replace("-", "")))
    txt = L["price"].dropna().astype(str); L["p"] = pd.to_numeric(L["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    add("prices written as text", len(txt)); add("prices not in the form $1,234.56", int((~txt.str.match(PRICE_TEXT)).sum())); add("prices blank", int(L["price"].isna().sum())); add("prices that fail to convert", int((L["p"].isna() & L["price"].notna()).sum()))
    L["nights"] = (pd.to_datetime(L["price_quote_checkout_date"]) - pd.to_datetime(L["price_quote_checkin_date"])).dt.days; L["lead"] = (pd.to_datetime(L["price_quote_checkin_date"]) - pd.to_datetime(L["last_scraped"])).dt.days
    L["weekday"] = pd.to_datetime(L["price_quote_checkin_date"]).dt.dayofweek
    add("listings with a price", int(L["p"].notna().sum())); add("priced, minimum stay under 30 nights", int((L["p"].notna() & (L["minimum_nights"] < 30)).sum()))
    q1 = L["nights"].between(1, 7); add("priced, short minimum stay, quote of 1 to 7 nights (the sample)", int((L["p"].notna() & (L["minimum_nights"] < 30) & q1).sum()))
    add("priced, short minimum stay, quote of 8 nights or more", int((L["p"].notna() & (L["minimum_nights"] < 30) & (L["nights"] >= 8)).sum()))
    S = L[L["p"].notna() & (L["minimum_nights"] < 30) & q1].reset_index(drop=True); n = len(S); price = S["p"].to_numpy(); y = np.log(price)
    add("sample: quote shorter than 1 day or negative lead", int(((S["nights"] < 1) | (S["lead"] < 0)).sum())); add("sample: median days from scrape to check-in", float(S["lead"].median())); add("sample: hosts", int(S["host_id"].nunique()))
    add("sample: listings of hosts with more than one listing", int((S.groupby("host_id")["id"].transform("size") > 1).sum())); add("sample: most listings of one host", int(S.groupby("host_id").size().max()))
    S["bathrooms_all"] = S["bathrooms"].where(S["bathrooms"].notna(), S["bathrooms_text"].map(bathrooms))
    add("sample: bathrooms blank", int(S["bathrooms"].isna().sum())); add("sample: of which read from the bathrooms text", int((S["bathrooms"].isna() & S["bathrooms_all"].notna()).sum())); add("sample: bathrooms still missing", int(S["bathrooms_all"].isna().sum()))
    add("sample: bedrooms missing", int(S["bedrooms"].isna().sum())); add("sample: beds missing", int(S["beds"].isna().sum())); add("sample: review score missing", int(S["review_scores_rating"].isna().sum())); add("sample: host years missing", int(S["hosts_time_as_host_years"].isna().sum()))
    add("sample: superhost field missing", int(S["host_is_superhost"].isna().sum())); add("sample: kinds of property listed", int(S["property_type"].nunique())); add("sample: zip codes", int(S["neighbourhood_cleansed"].nunique()))
    add("sample: price quote equals the price", int(np.isclose(pd.to_numeric(S["price_quote_price_per_night"], errors="coerce").round(2), S["p"], rtol=0, atol=0.011).sum()))
    X, group_of, ref_zip, big_places = build(S); names = list(X.columns); Xa = np.c_[np.ones(n), X.to_numpy(float)]; gidx = {g: [1 + i for i, c in enumerate(names) if group_of[c] == g] for g in GROUPS}
    add("sample: zip code used as the reference", int(ref_zip)); add("sample: kinds of place with their own column", len(big_places)); add("sample: columns of the model (without the intercept)", len(names)); add("sample: amenity entries read", int(S["amenities"].map(lambda s: len(json.loads(s))).sum()))
    for g in GROUPS:
        add(f"columns: {g}", len(gidx[g]))
    pd.DataFrame(audit, columns=["check", "value"]).to_csv(R / "audit.csv", index=False)

    # prices (the course questions): by room type
    pr = []
    for nm, mk in (("All listings", np.ones(n, bool)), *((r, (S["room_type"] == r).to_numpy()) for r in ("Entire home/apt", "Private room", "Hotel room", "Shared room"))):
        v = price[mk]; pr.append({"group": nm, "listings": int(mk.sum()), "median": float(np.median(v)), "q1": float(np.quantile(v, 0.25)), "q3": float(np.quantile(v, 0.75)), "mean": float(v.mean()), "p99": float(np.quantile(v, 0.99)), "max": float(v.max())})
    pd.DataFrame(pr).to_csv(R / "prices.csv", index=False)

    # models: scores by host-grouped folds, with the groups added one at a time, alone, and removed from the full model
    fold_h = (S["host_id"].astype("int64") % 5).to_numpy(); fold_l = (S["id"].astype("int64") % 5).to_numpy(); rows = []
    def spec(cols_):
        return [0] + cols_
    def run(label, cols_, fold, folds_name, columns=None):
        pred = oof_ols(Xa[:, spec(cols_)], y, fold); rows.append({"model": label, "folds": folds_name, "columns": len(cols_) if columns is None else columns, **score(y, price, pred)}); return pred
    run("no features (the mean)", [], fold_h, "by host", 0)
    cum, cumr2 = [], {}
    for g in GROUPS:
        cum = cum + gidx[g]; pred = run(g if g == "quote" else f"+ {g}", cum, fold_h, "by host"); cumr2[g] = rows[-1]["r2"]
    full_cols = cum; full_pred_h = pred
    alone = {g: score(y, price, oof_ols(Xa[:, spec(gidx[g])], y, fold_h))["r2"] for g in GROUPS}
    lost = {g: cumr2["amenities"] - score(y, price, oof_ols(Xa[:, spec([c for c in full_cols if c not in gidx[g]])], y, fold_h))["r2"] for g in GROUPS}
    gb_h = oof_gbm(np.c_[X.to_numpy(float), S[["latitude", "longitude"]].to_numpy(float)], y, fold_h); rows.append({"model": "gradient boosting", "folds": "by host", "columns": len(names) + 2, **score(y, price, gb_h)})
    full_pred_l = run("least squares, all columns", full_cols, fold_l, "by listing")
    gb_l = oof_gbm(np.c_[X.to_numpy(float), S[["latitude", "longitude"]].to_numpy(float)], y, fold_l); rows.append({"model": "gradient boosting", "folds": "by listing", "columns": len(names) + 2, **score(y, price, gb_l)})
    pd.DataFrame(rows).to_csv(R / "scores.csv", index=False)
    gr, before = [], 0.0
    for g in GROUPS:
        gr.append({"group": g, "columns": len(gidx[g]), "alone_r2": alone[g], "added_r2": cumr2[g] - before, "cumulative_r2": cumr2[g], "lost_if_removed_r2": lost[g]}); before = cumr2[g]
    pd.DataFrame(gr).to_csv(R / "groups.csv", index=False)

    # the full least-squares fit on every row, with errors robust to non-constant variance (HC1)
    beta = np.linalg.lstsq(Xa, y, rcond=None)[0]; e = y - Xa @ beta; k = Xa.shape[1]; bread = np.linalg.inv(Xa.T @ Xa); meat = (Xa * e[:, None] ** 2).T @ Xa; V = n / (n - k) * bread @ meat @ bread; se = np.sqrt(np.diag(V))
    tn = ["intercept"] + names; gn = ["intercept"] + [group_of[c] for c in names]
    pd.DataFrame({"term": tn, "group": gn, "coefficient": beta, "se_hc1": se}).to_csv(R / "coefficients.csv", index=False)
    cf = dict(zip(tn, zip(beta, se))); eff = []
    for label, term, scale, kind in (("Doubling the guests it sleeps", "log_guests", np.log(2), "x"), ("One more bathroom", "bathrooms_all", 1, "x"), ("One more bedroom", "bedrooms", 1, "x"), ("A shared bathroom", "shared_bathroom", 1, "x"),
                                     ("A private room, not an entire home", "room_private", 1, "x"), ("Waterfront", "waterfront", 1, "x"), ("A hot tub", "hot_tub", 1, "x"), ("An EV charger", "ev_charger", 1, "x"), ("A pool", "pool", 1, "x"),
                                     ("Air conditioning", "air_conditioning", 1, "x"), ("A superhost badge", "superhost", 1, "x"), ("A review score half a point higher", "review_score", 0.5, "x")):
        b, s = cf[term]; eff.append({"label": label, "term": term, "scale": scale, "coefficient": b, "se_hc1": s, "effect_pct": 100 * (np.exp(scale * b) - 1), "ci_low_pct": 100 * (np.exp(scale * (b - 1.96 * s)) - 1), "ci_high_pct": 100 * (np.exp(scale * (b + 1.96 * s)) - 1)})
    pd.DataFrame(eff).to_csv(R / "effects.csv", index=False)

    # the location premium: how much more or less than its size and type suggest a listing is quoted, by zip code (out-of-fold, before the zip code is used)
    pred_pre = oof_ols(Xa[:, spec(gidx["quote"] + gidx["size"] + gidx["type"])], y, fold_h); res = y - pred_pre; z = S["neighbourhood_cleansed"].astype(str); loc = []
    for zc_ in sorted(z.unique()):
        mk = (z == zc_).to_numpy(); m = float(np.median(res[mk])); loc.append({"zip": zc_, "listings": int(mk.sum()), "median_residual": m, "premium_pct": 100 * (np.exp(m) - 1)})
    pd.DataFrame(loc).to_csv(R / "location.csv", index=False)

    # the errors, by how expensive the listing is
    edges = np.quantile(price, [0.2, 0.4, 0.6, 0.8]); band = np.searchsorted(edges, price, side="left") + 1; er = []
    for nm, pred in (("least squares", full_pred_h), ("gradient boosting", gb_h)):
        for b in range(1, 6):
            mk = band == b; ape = np.abs(np.exp(pred[mk]) - price[mk]) / price[mk]
            er.append({"model": nm, "price_fifth": b, "listings": int(mk.sum()), "price_min": float(price[mk].min()), "price_max": float(price[mk].max()), "mdape": float(np.median(ape)), "median_fit_over_price": float(np.median(np.exp(pred[mk]) / price[mk]))})
    pd.DataFrame(er).to_csv(R / "errors.csv", index=False)
    sm = {"sample": n, "median_price": float(np.median(price)), "mean_price": float(price.mean()), "price_p99": float(np.quantile(price, 0.99)), "price_max": float(price.max()), "r2_full_ols_host": cumr2["amenities"], "r2_gbm_host": score(y, price, gb_h)["r2"]}
    pd.DataFrame(list(sm.items()), columns=["key", "value"]).to_csv(R / "summary.csv", index=False)
    for f in ("audit", "prices", "scores", "groups", "effects", "location", "errors"):
        print("==", f); print(pd.read_csv(R / f"{f}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
