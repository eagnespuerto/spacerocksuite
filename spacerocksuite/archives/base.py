"""Common pieces for plate/image archive searches."""
from __future__ import annotations

import csv
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

import numpy as np

from ..device import PROFILE


@dataclass
class PlateHit:
    archive: str
    plate_id: str
    telescope: str
    band: str
    mid_jd: float
    exptime_s: float
    ra_obj: float               # predicted object position at mid-exposure (deg)
    dec_obj: float
    vmag_pred: float | None
    sigma_a: float              # n-sigma ellipse used for the test (arcsec)
    sigma_b: float
    sigma_pa: float
    limit_mag: float | None = None
    margin_arcsec: float | None = None   # >0: ellipse fully inside the plate by this much
    covered_fraction: float = 1.0        # fraction of the ellipse on the plate
    trail_arcsec: float | None = None    # motion during the exposure
    url: str = ""
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def depth_margin(self):
        """limit - predicted V (positive means the plate should reach the object)."""
        if self.limit_mag is None or self.vmag_pred is None or np.isnan(self.vmag_pred):
            return None
        return self.limit_mag - self.vmag_pred

    def row(self):
        d = asdict(self)
        d.pop("extra")
        d["depth_margin"] = self.depth_margin
        return d


class Archive:
    name = "archive"
    description = ""
    years = (1880, 2100)

    def search(self, track, nsigma: float, jd0=None, jd1=None, progress=None, stop=None) -> list[PlateHit]:
        raise NotImplementedError


def pmap(fn, items, workers=None, progress=None, stop=None):
    """Thread-pool map sized for this device, with progress callback and cancel flag."""
    workers = workers or PROFILE.net_workers
    out = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            if stop is not None and stop.is_set():
                ex.shutdown(wait=False, cancel_futures=True)
                break
            i = futs[f]
            try:
                out[i] = f.result()
            except Exception as e:  # keep going; report per item
                out[i] = e
            done += 1
            if progress:
                progress(done / len(items))
    return out


def write_csv(hits: list[PlateHit], path):
    rows = [h.row() for h in hits]
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def tsv_rows(text: str):
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))
