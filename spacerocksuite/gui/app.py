"""SpaceRockSuite GUI (Tkinter). Plain ttk widgets, so it runs on a Pi 3B+.

Tabs: Target -> Plates -> Stack -> Stamp grid, plus Settings. Network and
image jobs run on a worker thread; results come back through a queue polled
by the Tk main loop.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
import traceback
import webbrowser
from tkinter import filedialog, messagebox, ttk

from .. import APP_NAME, __version__, config, core
from ..device import PROFILE

ARCHIVES = ["SuperCOSMOS", "DASCH", "SSOIS", "DSS"]
PAD = {"padx": 6, "pady": 4}


class Worker:
    """One background job at a time, with progress, log lines and a cancel flag."""

    def __init__(self, app):
        self.app = app
        self.q = queue.Queue()
        self.stop = threading.Event()
        self.busy = False

    def run(self, label, fn, done=None):
        if self.busy:
            messagebox.showinfo(APP_NAME, "Please wait for the current job to finish (or press Cancel).")
            return
        self.busy = True
        self.stop.clear()
        self.app.set_status(label + " ...")
        self.app.progress["value"] = 0

        def target():
            try:
                res = fn()
                self.q.put(("done", done, res))
            except Exception as e:  # shown to the user, full trace in the log
                self.q.put(("log", f"Error: {e}\n{traceback.format_exc(limit=3)}"))
                self.q.put(("done", None, None))
        threading.Thread(target=target, daemon=True).start()

    def log(self, msg):
        self.q.put(("log", str(msg)))

    def progress(self, f):
        self.q.put(("progress", f))

    def poll(self):
        try:
            while True:
                kind, *rest = self.q.get_nowait()
                if kind == "log":
                    self.app.log(rest[0])
                elif kind == "progress":
                    self.app.progress["value"] = max(0, min(100, 100 * rest[0]))
                elif kind == "done":
                    cb, res = rest
                    self.busy = False
                    self.app.progress["value"] = 100 if res is not None else 0
                    self.app.set_status("Ready")
                    if cb and res is not None:
                        cb(res)
        except queue.Empty:
            pass
        self.app.root.after(150, self.poll)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.settings = config.load()
        config.apply(self.settings)
        self.track = None
        self.hits = []
        self.obs80 = None
        self.last_spec = None
        self._img = None
        root.title(f"{APP_NAME} {__version__}")
        try:
            from importlib import resources
            self._icon = tk.PhotoImage(file=str(resources.files("spacerocksuite.data").joinpath("icon.png")))
            root.iconphoto(True, self._icon)
        except Exception:
            pass
        # scale everything to the screen (Windows display scaling, HiDPI Pi screens)
        self.k = max(1.0, root.winfo_fpixels("1i") / 96.0)
        w, h = (900, 640) if PROFILE.tier == "light" else (1100, 760)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        W, H = min(int(w * self.k), int(sw * 0.95)), min(int(h * self.k), int(sh * 0.9))
        root.geometry(f"{W}x{H}")
        root.minsize(int(700 * self.k), int(500 * self.k))
        from tkinter import font as tkfont
        line = tkfont.nametofont("TkDefaultFont").metrics("linespace")
        ttk.Style().configure("Treeview", rowheight=int(line * 1.35))
        self.worker = Worker(self)
        self._build()
        self.worker.poll()
        if self.settings.get("check_updates"):
            root.after(2500, lambda: self.check_updates(quiet=True))

    # -------------------------------------------------------------- layout
    def _build(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)
        self.nb = nb
        for name, fn in (("1 Target", self._tab_target), ("2 Plates", self._tab_plates),
                         ("3 Stack", self._tab_stack), ("4 Stamp grid", self._tab_grid),
                         ("Settings", self._tab_settings)):
            f = ttk.Frame(nb)
            nb.add(f, text=name)
            fn(f)
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, length=220, maximum=100)
        self.progress.pack(side="left", **PAD)
        ttk.Button(bottom, text="Cancel", command=self.worker.stop.set).pack(side="left")
        self.status = ttk.Label(bottom, text="Ready")
        self.status.pack(side="left", **PAD)
        ttk.Label(bottom, text=f"{core.backend()} · {PROFILE.tier} device").pack(side="right", **PAD)
        self.logbox = tk.Text(self.root, height=7, wrap="word", font=("TkFixedFont", 9))
        self.logbox.pack(fill="x")

    def log(self, msg):
        self.logbox.insert("end", time.strftime("%H:%M:%S ") + str(msg).rstrip() + "\n")
        self.logbox.see("end")

    def set_status(self, s):
        self.status.config(text=s)

    def _row(self, parent, r, label, widget):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", **PAD)
        widget.grid(row=r, column=1, sticky="we", **PAD)
        return widget

    # -------------------------------------------------------------- target
    SERVICES = [("JPL Horizons  (numbered or designated objects)", "horizons"),
                ("JPL Scout  (NEOCP objects, with Monte Carlo uncertainty)", "scout"),
                ("Find_Orb  (fits MPC or your own observations)", "findorb")]

    def _tab_target(self, f):
        # left: how to get the track; right: orbit sketch of the track
        pw = ttk.PanedWindow(f, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        right = ttk.LabelFrame(pw, text="Orbit (from the track)")
        pw.add(left, weight=3)
        pw.add(right, weight=2)
        left.columnconfigure(0, weight=1)

        src = self.settings["source"]
        self.v_route = tk.StringVar(value="paste" if src == "ephemeral" else "lookup")
        self.v_target = tk.StringVar(value="2026 ED138")
        labels = dict((v, k) for k, v in self.SERVICES)
        self.v_service = tk.StringVar(value=labels.get(src, self.SERVICES[0][0]))
        self.v_start = tk.StringVar(value="1950-01-01")
        self.v_stop = tk.StringVar(value="2003-12-31")
        self.v_step = tk.StringVar(value=str(self.settings["step_days"]))
        self.v_site = tk.StringVar(value=self.settings["site"])

        ttk.Label(left, text="Object name or designation (also used as the name for a pasted ephemeris)").grid(
            row=0, column=0, sticky="w", **PAD)
        ttk.Entry(left, textvariable=self.v_target).grid(row=1, column=0, sticky="we", **PAD)

        # A. look up online
        fa = ttk.Frame(left)
        ttk.Radiobutton(left, text="A.  Look it up online", value="lookup", variable=self.v_route).grid(
            row=2, column=0, sticky="w", padx=6, pady=(10, 0))
        fa.grid(row=3, column=0, sticky="we", padx=(28, 6))
        fa.columnconfigure(1, weight=1)
        cb = ttk.Combobox(fa, textvariable=self.v_service, state="readonly", values=[k for k, _ in self.SERVICES])
        self._row(fa, 0, "Service", cb)
        span = ttk.Frame(fa)
        for i, (lab, var, w) in enumerate((("from", self.v_start, 11), ("to", self.v_stop, 11),
                                           ("every", self.v_step, 5))):
            if i:
                ttk.Label(span, text=lab).pack(side="left", padx=(6, 2))
            ttk.Entry(span, textvariable=var, width=w).pack(side="left")
        ttk.Label(span, text="days").pack(side="left", padx=2)
        self._row(fa, 1, "Dates (UTC)", span)
        self._row(fa, 2, "Observatory code", ttk.Entry(fa, textvariable=self.v_site, width=8)).grid(sticky="w")
        ob = ttk.Frame(fa)
        ttk.Button(ob, text="Use my observations...", command=self.load_obs).pack(side="left")
        self.l_obs = ttk.Label(ob, text="  none: Find_Orb downloads the MPC's", foreground="#666")
        self.l_obs.pack(side="left")
        self._row(fa, 3, "Find_Orb only", ob)

        # B. paste
        ttk.Radiobutton(left, text="B.  Paste an ephemeris  (Find_Orb / projectpluto, Horizons, MPC, "
                                   "or time-RA-Dec table)", value="paste", variable=self.v_route).grid(
            row=4, column=0, sticky="w", padx=6, pady=(10, 0))
        self.t_ephem = tk.Text(left, height=8, font=("TkFixedFont", 9), wrap="none")
        self.t_ephem.grid(row=5, column=0, sticky="nsew", padx=(28, 6), pady=4)
        left.rowconfigure(5, weight=1)

        # picking a section by using it selects its radio button
        # (clicks and typing only: focus also returns when you switch back from the browser)
        for w in [cb] + list(fa.winfo_children()) + list(span.winfo_children()):
            for ev in ("<Button-1>", "<Key>"):
                w.bind(ev, lambda e: self.v_route.set("lookup"), add="+")
        cb.bind("<<ComboboxSelected>>", lambda e: self.v_route.set("lookup"), add="+")
        for ev in ("<Button-1>", "<<Paste>>", "<Key>"):
            self.t_ephem.bind(ev, lambda e: self.v_route.set("paste"), add="+")

        b = ttk.Frame(left)
        b.grid(row=6, column=0, sticky="w", **PAD)
        ttk.Button(b, text="Get track", command=self.get_track).pack(side="left", padx=3)
        ttk.Button(b, text="Save track...", command=self.save_track).pack(side="left", padx=3)
        ttk.Button(b, text="Load track...", command=self.load_track).pack(side="left", padx=3)
        ttk.Button(b, text="Open in Ephemeral", command=self.open_ephemeral).pack(side="left", padx=3)
        self.l_track = ttk.Label(left, text="No track yet.", justify="left")
        self.l_track.grid(row=7, column=0, sticky="w", **PAD)

        self.orbit_canvas = tk.Canvas(right, highlightthickness=0, width=int(320 * self.k), height=int(320 * self.k))
        self.orbit_canvas.pack(fill="both", expand=True)
        self._orbit = {"error": "Get a track to see its orbit here."}
        self.orbit_canvas.bind("<Configure>", lambda e: self._draw_orbit())

    def _draw_orbit(self):
        from .. import orbit_view
        orbit_view.draw(self.orbit_canvas, self._orbit)

    def get_track(self):
        from .. import precovery as P
        paste = self.v_route.get() == "paste"
        text = self.t_ephem.get("1.0", "end").strip()
        if paste and not text:
            messagebox.showinfo(APP_NAME, "Paste an ephemeris into box B first, or choose A to look it up online.")
            return
        try:
            step = float(self.v_step.get())
        except ValueError:
            messagebox.showinfo(APP_NAME, "The step must be a number of days, e.g. 1 or 0.25.")
            return
        service = dict(self.SERVICES).get(self.v_service.get(), "horizons")
        a = dict(target=self.v_target.get().strip(), start=self.v_start.get(), stop=self.v_stop.get(),
                 source="ephemeral" if paste else service, step_days=step, site=self.v_site.get().strip(),
                 obs80=self.obs80, ephem_text=text or None,
                 findorb_binary=self.settings.get("findorb_binary") or None, progress=self.worker.progress)
        label = "the pasted ephemeris" if paste else self.v_service.get().split("  ")[0]
        self.worker.run(f"Getting track from {label}", lambda: P.get_track(**a), self._got_track)

    def _got_track(self, t):
        import numpy as np
        from .. import orbit_view
        from ..precovery import iso
        self.track = t
        fmt = t.meta.get("format")
        s = f"{t.name}: {len(t)} rows from {t.source}{f' ({fmt})' if fmt else ''}, " \
            f"{iso(t.jd[0])[:16]} to {iso(t.jd[-1])[:16]}"
        if t.smaa is not None:
            a, _, _ = t.ellipse(t.jd, 3)
            s += f"\n3-sigma major axis: {np.nanmin(a):.1f}\" to {np.nanmax(a):.1f}\""
        if t.vmag is not None:
            s += f"\nPredicted V: {np.nanmin(t.vmag):.1f} to {np.nanmax(t.vmag):.1f}"
        self.l_track.config(text=s)
        self.log(s.replace("\n", "; "))
        try:
            self._orbit = orbit_view.build(t)
        except Exception as e:  # the sketch is a bonus; never block the track
            self._orbit = {"error": f"Could not draw the orbit: {e}"}
        self._draw_orbit()

    def load_obs(self):
        self.v_route.set("lookup")
        p = filedialog.askopenfilename(title="MPC 80-column or ADES observations")
        if p:
            self.obs80 = open(p, encoding="utf-8", errors="replace").read()
            n = len(self.obs80.splitlines())
            self.l_obs.config(text=f"  {os.path.basename(p)} ({n} lines)")
            self.log(f"Loaded {n} observation lines from {os.path.basename(p)}")

    def open_ephemeral(self):
        if not self._need_track():
            return
        from ..services import ephemeral
        self.log("Opened in Ephemeral: " + ephemeral.open_in_browser(self.track)[:80] + "...")

    def save_track(self):
        if not self._need_track():
            return
        p = filedialog.asksaveasfilename(defaultextension=".json", initialfile="track.json")
        if p:
            open(p, "w", encoding="utf-8").write(self.track.to_json())

    def load_track(self):
        from ..track import Track
        p = filedialog.askopenfilename(filetypes=[("Track JSON", "*.json")])
        if p:
            self._got_track(Track.from_json(open(p, encoding="utf-8").read()))

    def _need_track(self):
        if self.track is None:
            messagebox.showinfo(APP_NAME, "Get a track first (tab 1).")
            return False
        return True

    # -------------------------------------------------------------- plates
    def _tab_plates(self, f):
        top = ttk.Frame(f)
        top.pack(fill="x")
        self.v_sigma = tk.DoubleVar(value=self.settings["nsigma"])
        ttk.Label(top, text="Uncertainty").pack(side="left", **PAD)
        sc = ttk.Scale(top, from_=1, to=6, variable=self.v_sigma, length=160,
                       command=lambda v: self.l_sigma.config(text=f"{float(v):.1f} σ"))
        sc.pack(side="left")
        self.l_sigma = ttk.Label(top, text=f"{self.v_sigma.get():.1f} σ", width=6)
        self.l_sigma.pack(side="left")
        self.v_arch = {a: tk.BooleanVar(value=a in self.settings["archives"]) for a in ARCHIVES}
        for a, v in self.v_arch.items():
            ttk.Checkbutton(top, text=a, variable=v).pack(side="left", padx=2)
        ttk.Button(top, text="Search", command=self.search).pack(side="left", padx=8)
        cols = ("date", "archive", "plate", "band", "exp", "V", "limit", "margin", "cover", "ellipse", "note")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", selectmode="extended")
        widths = (120, 120, 170, 110, 50, 45, 45, 55, 50, 80, 220)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=int(w * self.k), minwidth=int(30 * self.k), stretch=c == "note")
        ys = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ys.pack(side="left", fill="y")
        side = ttk.Frame(f)
        side.pack(side="left", fill="y")
        ttk.Button(side, text="Export CSV...", command=self.export_hits).pack(fill="x", **PAD)
        ttk.Button(side, text="Stamps of selected", command=self.stamps_selected).pack(fill="x", **PAD)
        ttk.Button(side, text="Open archive page", command=self.open_hit).pack(fill="x", **PAD)
        self.v_matched = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Gaussian matched filter", variable=self.v_matched).pack(anchor="w", **PAD)

    def search(self):
        if not self._need_track():
            return
        from .. import precovery as P
        arch = [a for a, v in self.v_arch.items() if v.get()]
        ns = round(float(self.v_sigma.get()), 1)
        self.worker.run(f"Searching {', '.join(arch)} at {ns} σ",
                        lambda: P.search_plates(self.track, ns, arch, obs80=self.obs80,
                                                date_fix_days=float(self.settings["ssa_date_fix_days"]),
                                                progress=self.worker.progress, stop=self.worker.stop,
                                                log=self.worker.log), self._got_hits)

    def _got_hits(self, hits):
        from ..precovery import iso
        self.hits = hits
        self.tree.delete(*self.tree.get_children())
        for i, h in enumerate(hits):
            dm = h.depth_margin
            self.tree.insert("", "end", iid=str(i), values=(
                iso(h.mid_jd), h.archive, h.plate_id, h.band, f"{h.exptime_s / 60:.0f}m",
                "" if h.vmag_pred is None else f"{h.vmag_pred:.1f}", "" if h.limit_mag is None else f"{h.limit_mag:.1f}",
                "" if dm is None else f"{dm:+.1f}", f"{h.covered_fraction:.2f}",
                f"{h.sigma_a:.0f}x{h.sigma_b:.0f}\"", h.note))
        self.log(f"{len(hits)} plates/frames found")

    def _selected(self):
        return [self.hits[int(i)] for i in self.tree.selection()]

    def export_hits(self):
        from ..archives import write_csv
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="plates.csv")
        if p:
            write_csv(self.hits, p)

    def open_hit(self):
        for h in self._selected()[:3]:
            if h.url:
                webbrowser.open(h.url)

    def stamps_selected(self):
        sel = [h for h in self._selected() if h.archive.startswith(("SuperCOSMOS", "DSS"))]
        if not sel:
            messagebox.showinfo(APP_NAME, "Select SuperCOSMOS or DSS rows (other archives: use 'Open archive page').")
            return
        from .. import precovery as P
        out = os.path.join(self.settings["output_dir"], "".join(self.track.name.split()), time.strftime("%Y%m%d-%H%M%S"))
        matched = self.v_matched.get()

        def job():
            res = []
            for k, h in enumerate(sel):
                if self.worker.stop.is_set():
                    break
                try:
                    blob = P.fetch_cutout(h)
                    r = (P.measure_with_date_probe(blob, h, self.track, matched=matched)
                         if h.archive.startswith("SuperCOSMOS") else P.measure_stamp(blob, h, matched=matched))
                    res.append(r)
                    self.worker.log(f"{h.plate_id}: S/N {r.snr:.1f} {r.note}")
                except Exception as e:
                    self.worker.log(f"{h.plate_id}: skipped ({e})")
                self.worker.progress((k + 1) / len(sel))
            if not res:
                raise RuntimeError("no stamps could be made")
            from ..imaging import stampgrid
            spec = P.grid_spec(self.track.name, res, out)
            return spec, stampgrid.render_file(spec)
        self.worker.run("Making stamps", job, self._got_grid)

    # -------------------------------------------------------------- stack
    def _tab_stack(self, f):
        f.columnconfigure(1, weight=1)
        self.frames_paths = []
        self.l_frames = ttk.Label(f, text="No frames.")
        b = ttk.Frame(f)
        b.grid(row=0, column=0, columnspan=2, sticky="w", **PAD)
        ttk.Button(b, text="Add FITS frames...", command=self.add_frames).pack(side="left", padx=3)
        ttk.Button(b, text="Clear", command=lambda: (self.frames_paths.clear(),
                                                     self.l_frames.config(text="No frames."))).pack(side="left")
        self.l_frames.grid(row=1, column=0, columnspan=2, sticky="w", **PAD)
        self.v_mode = tk.StringVar(value="track")
        self.v_method = tk.StringVar(value="clipmean")
        self.v_size = tk.StringVar(value=str(PROFILE.stamp_px))
        self.v_scale = tk.StringVar(value="1.0")
        self.v_bkg = tk.StringVar(value="41")
        self.v_fwhm = tk.StringVar(value="0")
        self.v_rap = tk.StringVar(value="3")
        self.v_null = tk.BooleanVar(value=True)
        self._row(f, 2, "Mode (track = shift-stack on the prediction)",
                  ttk.Combobox(f, textvariable=self.v_mode, values=["track", "static"], state="readonly"))
        self._row(f, 3, "Combine", ttk.Combobox(f, textvariable=self.v_method, state="readonly",
                                                values=["clipmean", "median", "mean"]))
        self._row(f, 4, "Stamp size (px)", ttk.Entry(f, textvariable=self.v_size))
        self._row(f, 5, "Pixel scale (arcsec)", ttk.Entry(f, textvariable=self.v_scale))
        self._row(f, 6, "Median background box (px, 0 = off)", ttk.Entry(f, textvariable=self.v_bkg))
        self._row(f, 7, "Gaussian matched filter FWHM (px, 0 = off)", ttk.Entry(f, textvariable=self.v_fwhm))
        self._row(f, 8, "Aperture radius (px)", ttk.Entry(f, textvariable=self.v_rap))
        ttk.Checkbutton(f, text="Null tests (reversed / stationary / rotated / double rate)",
                        variable=self.v_null).grid(row=9, column=0, columnspan=2, sticky="w", **PAD)
        ttk.Button(f, text="Stack", command=self.run_stack).grid(row=10, column=0, sticky="w", **PAD)
        self.c_stack = tk.Label(f)
        self.c_stack.grid(row=11, column=0, columnspan=2, **PAD)

    def add_frames(self):
        ps = filedialog.askopenfilenames(filetypes=[("FITS", "*.fits *.fit *.fts *.fits.gz *.fz"), ("All", "*")])
        self.frames_paths += list(ps)
        self.l_frames.config(text=f"{len(self.frames_paths)} frames")

    def run_stack(self):
        if not self._need_track() or not self.frames_paths:
            if self.track is not None:
                messagebox.showinfo(APP_NAME, "Add some FITS frames first.")
            return
        from ..imaging import stack as S, cleanup
        mode, method = self.v_mode.get(), self.v_method.get()
        size, scale = int(self.v_size.get()), float(self.v_scale.get())
        bkg, fwhm, r = int(self.v_bkg.get()), float(self.v_fwhm.get()), float(self.v_rap.get())
        paths, null = list(self.frames_paths), self.v_null.get()
        out = os.path.join(self.settings["output_dir"], "".join(self.track.name.split()))
        os.makedirs(out, exist_ok=True)

        def job():
            import numpy as np
            fr = []
            for k, p in enumerate(paths):
                fr.append(S.Frame.from_fits(p, name=os.path.basename(p)))
                self.worker.progress(0.3 * (k + 1) / len(paths))
            kw = dict(size_px=size, scale_arcsec=scale, method=method, bkg_size=bkg or None)
            res = S.stack(fr, self.track, mode, **kw)
            img = res["stack"]
            if fwhm > 0:
                img = cleanup.matched(np.nan_to_num(img), fwhm)
            self.worker.progress(0.5)
            noise, _ = S.control_noise(fr, self.track, mode, r_px=r, **kw)
            flux, _ = S.photometry(img, r)
            self.worker.log(f"Stack of {len(fr)} ({mode}): flux {flux:.4g}, control noise {noise:.4g}, "
                            f"S/N {flux / noise if noise else float('nan'):.1f}")
            if null:
                for name, v in S.null_tests(fr, self.track, r_px=r, **kw).items():
                    self.worker.log(f"  {name:<12} {v / noise if noise else float('nan'):+.1f} σ")
            self.worker.progress(0.95)
            path = os.path.join(out, f"stack_{mode}.npz")
            np.savez_compressed(path, stack=img, coverage=res["coverage"])
            spec = {"title": self.track.name, "subtitle": f"{'shift-' if mode == 'track' else ''}stack "
                    f"(centred on predicted position)", "out": f"stack_{mode}.png", "ncols": 1,
                    "panels": [{"file": os.path.basename(path), "key": "stack", "hw": size * scale / 2,
                                "r": r * scale, "title": f"{len(fr)} frames, {method}",
                                "subtitle": f"S/N {flux / noise if noise else float('nan'):.1f} at prediction",
                                "n": len(fr)}]}
            import json
            sp = os.path.join(out, f"stack_{mode}.json")
            json.dump(spec, open(sp, "w", encoding="utf-8"), indent=1)
            from ..imaging import stampgrid
            return sp, stampgrid.render_file(sp)
        self.worker.run("Stacking", job, lambda res: self._show(res[1], self.c_stack, int(360 * self.k)))

    # -------------------------------------------------------------- grid
    def _tab_grid(self, f):
        b = ttk.Frame(f)
        b.pack(fill="x")
        ttk.Button(b, text="Open stamp_grid.json...", command=self.open_spec).pack(side="left", **PAD)
        ttk.Button(b, text="Re-render", command=self.rerender).pack(side="left", **PAD)
        ttk.Button(b, text="Open output folder", command=self.open_folder).pack(side="left", **PAD)
        self.l_grid = ttk.Label(f, text="Grids made from tab 2 show here. Same spec + data = same PNG bytes.")
        self.l_grid.pack(anchor="w", **PAD)
        self.c_grid = tk.Label(f)
        self.c_grid.pack(fill="both", expand=True)

    def open_spec(self):
        p = filedialog.askopenfilename(filetypes=[("Stamp grid spec", "*.json")])
        if p:
            self.last_spec = p
            self.rerender()

    def rerender(self):
        if not self.last_spec:
            return
        from ..imaging import stampgrid
        spec = self.last_spec
        self.worker.run("Rendering grid", lambda: (spec, stampgrid.render_file(spec)), self._got_grid)

    def _got_grid(self, res):
        from ..imaging import stampgrid
        spec, png = res
        self.last_spec = spec
        self.l_grid.config(text=f"{png}   sha256 {stampgrid.sha256(png)[:16]}")
        self.nb.select(3)
        self._show(png, self.c_grid, int(self.c_grid.winfo_height() or 560 * self.k))
        self.log(f"Saved {png}")

    def _show(self, png, label, max_h):
        img = tk.PhotoImage(file=png)
        k = max(1, -(-img.height() // max_h))
        img = img.subsample(k, k)
        label.configure(image=img)
        label.image = img

    def open_folder(self):
        p = os.path.dirname(self.last_spec) if self.last_spec else self.settings["output_dir"]
        os.makedirs(p, exist_ok=True)
        if os.name == "nt":
            os.startfile(p)
        else:
            webbrowser.open("file://" + p)

    # -------------------------------------------------------------- settings
    def _tab_settings(self, f):
        f.columnconfigure(1, weight=1)
        self.v_set = {}
        rows = [("net_workers", "Parallel archive queries"), ("mc_samples", "Monte Carlo orbit samples (Scout)"),
                ("stamp_px", "Default stamp size (px)"), ("ssa_date_fix_days", "SuperCOSMOS date fix (days)"),
                ("findorb_binary", "Local Find_Orb (fo / fo64.exe), blank = web service"),
                ("output_dir", "Output folder")]
        for r, (k, label) in enumerate(rows):
            v = tk.StringVar(value=str(self.settings.get(k, "")))
            self.v_set[k] = v
            self._row(f, r, label, ttk.Entry(f, textvariable=v))
        self.v_upd = tk.BooleanVar(value=bool(self.settings.get("check_updates")))
        ttk.Checkbutton(f, text="Check for updates at start", variable=self.v_upd).grid(
            row=len(rows), column=0, sticky="w", **PAD)
        b = ttk.Frame(f)
        b.grid(row=len(rows) + 1, column=0, columnspan=2, sticky="w", **PAD)
        ttk.Button(b, text="Save settings", command=self.save_settings).pack(side="left", padx=3)
        ttk.Button(b, text="Check for updates", command=self.check_updates).pack(side="left", padx=3)
        ttk.Button(b, text="Refresh SuperCOSMOS plate list", command=self.refresh_ssa).pack(side="left", padx=3)
        info = (f"{APP_NAME} {__version__}\nNumerical backend: {core.backend()}\n"
                f"Device: {PROFILE.tier} ({PROFILE.cores} cores, {PROFILE.ram_mb} MB, {PROFILE.machine}"
                f"{', ' + PROFILE.pi_model if PROFILE.pi_model else ''})")
        ttk.Label(f, text=info, justify="left").grid(row=len(rows) + 2, column=0, columnspan=2, sticky="w", **PAD)

    def save_settings(self):
        for k, v in self.v_set.items():
            val = v.get().strip()
            if k in ("net_workers", "mc_samples", "stamp_px"):
                val = int(val)
            elif k == "ssa_date_fix_days":
                val = float(val)
            self.settings[k] = val
        self.settings["check_updates"] = self.v_upd.get()
        self.settings["nsigma"] = round(float(self.v_sigma.get()), 1)
        self.settings["archives"] = [a for a, v in self.v_arch.items() if v.get()]
        config.save(self.settings)
        config.apply(self.settings)
        self.log("Settings saved")

    def refresh_ssa(self):
        from ..archives.supercosmos import refresh_plates
        self.worker.run("Downloading SSA plate list", refresh_plates,
                        lambda n: self.log(f"SuperCOSMOS plate list refreshed: {n} plates"))

    def check_updates(self, quiet=False):
        from .. import updater
        if quiet and self.worker.busy:          # never interrupt a job; try again shortly
            self.root.after(5000, lambda: self.check_updates(quiet=True))
            return

        def done(u):
            if not u.get("version"):
                if not quiet:
                    messagebox.showinfo(APP_NAME, f"{APP_NAME} {__version__} is up to date.")
                return
            a = u.get("asset")
            if not a:
                if messagebox.askyesno(APP_NAME, f"Version {u['version']} is out. Open the release page?"):
                    webbrowser.open(u["url"])
                return
            size = a.get("size", 0) / 1e6
            if messagebox.askyesno(APP_NAME, f"Version {u['version']} is available.\n\nDownload "
                                             f"{a['name']} ({size:.1f} MB) from GitHub and install it?"):
                self.worker.run("Downloading update", lambda: updater.install(updater.download(a)),
                                lambda msg: (messagebox.showinfo(APP_NAME, msg),
                                             self.root.destroy() if a["name"].endswith(".exe") else None))
        self.worker.run("Checking for updates", lambda: updater.check() or {}, done)


def main():
    if os.name == "nt":  # crisp text under Windows display scaling
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
