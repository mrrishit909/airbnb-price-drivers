# What makes an Austin Airbnb expensive?

The Austin listings that Inside Airbnb collected from airbnb.com in June 2026, with the price each was quoted for a night. Five questions:
- what is quoted for a night, and what does the quote cover;
- how much of the price do a listing's size, type, place, host and reviews, and amenities explain, on listings the model has never seen;
- which of those groups matters on its own, and which only because it stands in for another;
- how far off is a prediction, and for which listings;
- how much does a naive split flatter a model, when a host has several listings.

The questions follow GeeksforGeeks' "Airbnb Data Analysis" (explore the listings, visualise them, model the price), with the model made into a test that can fail. The data is real, not a course file.

Build log: https://mrrishit909.github.io/projects/airbnb-price-drivers/

## Data

Inside Airbnb (https://insideairbnb.com/get-the-data/), Austin, Texas, 22 June 2026; the listings were collected on 4 dates from 22 June to 2 July 2026. Licensed under Creative Commons Attribution 4.0; attribution: Inside Airbnb (insideairbnb.com). Public listing information from airbnb.com.
Three files: `listings.csv.gz` (11,295 listings, 90 columns), `neighbourhoods.csv` (44 zip codes, which Inside Airbnb uses as neighbourhoods) and `neighbourhoods.geojson` (their outlines). About 8 MB, so none is committed; `results/pull.json` has the URLs, row counts and checksums.
Host names, listing titles and descriptions are in the listings file and are never used or published: the results are summaries. Inside Airbnb says Airbnb anonymises each listing's location to within 150 metres of the real address.

```
./venv/bin/pip install pandas numpy scikit-learn matplotlib certifi      # in a venv
./venv/bin/python download.py    # data/*
./venv/bin/python analyze.py     # results/*
./venv/bin/python check.py
./venv/bin/python charts.py
```

## Steps

1. **Download** the three files, with their checksums.
2. **Audit, clean and choose the sample.** The price is text ("$1,234.56") and becomes a number. A listing is in the sample if it has a price, a minimum stay under 30 nights and a price quote for a stay of 1 to 7 nights: 8,425 of 11,295. The price is that quote's total divided by its nights. Bathrooms are read from their text where the number is blank (707 of 712); other missing numbers are replaced by the sample median, with a flag.
3. **Build the columns**, in six groups: the quote (nights, days to check-in, weekday of check-in), size (guests, bedrooms, beds, bathrooms, a shared bathroom), type (room type, kind of place), place (zip code), host and reviews (superhost, the host's number of listings and years, verified, number of reviews, review score) and amenities (how many, and twelve picked in advance: pool, hot tub, waterfront, lake access, EV charger, gym, air conditioning, workspace, free parking, BBQ grill, elevator, washer). 89 columns in all.
4. **Score out of sample.** Five-fold cross-validation with folds made of hosts, so a host's other listings never help predict a listing; for contrast, folds made of listings. Models: ordinary least squares on the log price, with the groups added one at a time, each alone, and each removed from the full model; and a gradient-boosted tree model (scikit-learn, fixed settings, not tuned) given the same columns plus latitude and longitude. R squared is on the log price; the error of a prediction is |predicted price - price| / price.
5. **Check.** `check.py` rebuilds the columns, the least-squares fits and every score in plain Python (the csv, gzip, json and re modules and loops: no pandas, numpy or scipy): 658 numbers. Only the boosted model's fit is the library's: check.py hands it the matrix it built by hand and recomputes the scores from the predictions. Changing a threshold, a fold rule, a regular expression, the standard-error formula or the boosted model's settings makes it fail.
6. **Charts.**

## Results

- **What is quoted.** The price is not a list price: Airbnb was asked for a quote for one particular stay, here 3 days after the scrape at the median, and the price is the total divided by the nights. The median is $251 a night and the mean $403; the top 1% start at $2,804 and the highest is $58,600.50. The median for entire homes is $263 and for private rooms $129.
- **Size first.** Knowing nothing, or only what was quoted (the nights, the weekday, the days ahead), explains none of the variation in the log price (R squared −0.001 and −0.001, a hair below zero). Adding size takes it to 0.51; type to 0.52; place (the zip code) to 0.58; host and reviews to 0.61; amenities to 0.63. The boosted model reaches 0.67.
- **Order matters.** Size alone explains 0.51, but removing it from the full model costs only 0.15: type, amenities and the rest stand in for it. Type alone explains 0.26 and adds 0.01 after size; amenities alone 0.20 and add 0.02. Place alone explains 0.11, adds 0.07 and costs 0.04 if removed. Host and reviews: 0.05, 0.03 and 0.03.
- **What a feature goes with** (the full least-squares fit on every row, 95% interval from errors robust to non-constant variance; an association with the other columns held fixed, not a cause). Doubling the guests a listing sleeps: +19% (+15% to +22%). One more bathroom: +18% (+14% to +22%); one more bedroom: +5% (+1% to +9%). A private room instead of an entire home: −29% (−33% to −24%); a shared bathroom: −20% (−27% to −13%).
  Waterfront: +41% (+30% to +52%); a hot tub: +17% (+13% to +21%); a pool: +15% (+12% to +18%); an EV charger: +10% (+6% to +15%); air conditioning: +6% (−0.4% to +13%). A superhost badge: −8% (−10% to −6%); a review score half a point higher: +5% (+2% to +7%).
- **Place.** Against what its size and type suggest, the median listing is quoted furthest above in zip codes 78746 (+42%, 144 listings), 78733 (+37%, 66 listings), 78732 (+30%, 50 listings), 78701 (+27%, 842 listings), and furthest below in 78717 (−35%, 31 listings), 78728 (−33%, 72 listings), 78754 (−30%, 90 listings), 78752 (−29%, 146 listings) (zip codes with at least 30 sample listings; out of fold, before the zip code is used).
- **How far off.** Out of sample (by host) the full least-squares model misses by a median 24% and has 52% of listings within 25% of the price; the boosted model 21% and 57%. Both overprice the cheapest fifth of listings (the median prediction is 1.27 times the price for least squares) and underprice the dearest fifth (0.73); the dearest fifth is also the hardest to predict (median error 33% against 19% in the fifth below).
- **A naive split flatters the model.** 5,268 of the 8,425 listings (62.5%) belong to a host with another listing in the sample (one host has 178). With folds made of listings the boosted model scores R squared 0.77 and misses by a median 17%, and least squares 0.66, against 0.67 and 0.63 by host: the difference is what a model gains from having seen the same host's other listings.

## Not done, and caveats

- The price is one quote for one stay of 1 to 7 nights, made in June or July 2026: not an annual price, and it says nothing about weekends away from the quoted dates, events or seasons. Listings with a minimum stay of 30 nights or more (1,432 priced) and 464 priced short-stay listings whose quote was for 8 nights or more are left out, and the results do not apply to them.
- The effects are associations. A pool goes with a higher price partly because houses with pools are better in ways the columns do not capture; an EV charger stands in for a newer building; a superhost badge goes with a lower price after the other columns are held fixed, which is not evidence that the badge lowers it. Review count and score are consequences of demand as well as drivers of price.
- Folds made of hosts stop a host's other listings from helping, but not neighbouring listings of different hosts in the same building; the boosted model, given latitude and longitude, can still learn a building's prices from the training folds, so its score is somewhat optimistic for a truly new location. Its settings were fixed, not tuned, so it is not an upper bound.
- The listing data cannot see photos, the view, the quality of the furnishing, the floor or the host's pricing strategy. About a third of the variation in the log price is left unexplained by the best model (R squared 0.67).
- Predictions are exp(fit) of a log-price model, so they estimate the median price, not the mean. Missing bedrooms, beds, bathrooms, host years and review scores are replaced by the sample median (with a flag where it matters); 982 listings have no bedroom count.
- The amenity flags come from text rules chosen in advance, not a search; zip codes with fewer than 30 sample listings are not mapped. Coordinates are anonymised by Airbnb to within 150 metres.
- Free to use with attribution (CC BY 4.0). Inside Airbnb itself says the accuracy of the information compiled from the Airbnb site is not its responsibility; this is a portfolio project.
