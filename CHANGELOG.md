# Changelog

## [0.0.2] - 2026-11-09

- Target tab is clearer: two routes, **A. look it up online** and **B. paste an ephemeris**. Clicking or pasting in a section picks it, so a pasted ephemeris is no longer ignored when the source list says Horizons.
- Fixed pasted Find_Orb ephemerides (e.g. from projectpluto.com): rows like `1982 03 18  13 44 21.3` read the RA hour as the time of day, giving wrong dates and positions. Find_Orb's sigma, PA, magnitude and object name are now kept.
- New orbit sketch beside the ephemeris: the track seen from above the ecliptic, with the planets' orbits, Earth, and a two-body orbit (a, e, i, q) through the track. Works for open orbits too.
- Horizons errors for NEOCP ids now say to use Scout, Find_Orb or a pasted Find_Orb ephemeris.

## [0.0.1] - 2026-11-08

First release.

- Tracks from JPL Horizons (with 3σ ellipse), JPL Scout (orbit samples), Find_Orb (local or web) and pasted ephemerides (Ephemeral parser); open any track in Ephemeral.
- Plate search at 1–6σ: SuperCOSMOS (offline plate list, +1 day date fix), DASCH, CADC SSOIS, STScI DSS (POSS-I blue).
- Stamps with median background, Gaussian matched filter, template subtraction, least-squares star removal and deblending, Gaussian centroids, empirical noise.
- Stacking, shift-stacking and rate stacking with control-track noise and null tests.
- Deterministic stamp grids in the stamp-grid house style (`stamp_grid.json` compatible).
- Rust core (PyO3) with NumPy fallbacks; device tiers for Pi 3B+ to desktop.
- Tkinter GUI, `srs` command line, GitHub Releases updater; Windows installer, Linux and Raspberry Pi OS bundles.
