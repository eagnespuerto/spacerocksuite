# Changelog

## [0.0.1] - 2026-11-08

First release.

- Tracks from JPL Horizons (with 3σ ellipse), JPL Scout (orbit samples), Find_Orb (local or web) and pasted ephemerides (Ephemeral parser); open any track in Ephemeral.
- Plate search at 1–6σ: SuperCOSMOS (offline plate list, +1 day date fix), DASCH, CADC SSOIS, STScI DSS (POSS-I blue).
- Stamps with median background, Gaussian matched filter, template subtraction, least-squares star removal and deblending, Gaussian centroids, empirical noise.
- Stacking, shift-stacking and rate stacking with control-track noise and null tests.
- Deterministic stamp grids in the stamp-grid house style (`stamp_grid.json` compatible).
- Rust core (PyO3) with NumPy fallbacks; device tiers for Pi 3B+ to desktop.
- Tkinter GUI, `srs` command line, GitHub Releases updater; Windows installer, Linux and Raspberry Pi OS bundles.
