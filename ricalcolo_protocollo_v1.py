#!/usr/bin/env python3
"""
Ricalcolo secondo il protocollo v1.0 (brief A. Rampado, 11/09/2026)
Terremoto M7.6 Sand Point, Alaska - 2020-10-19 20:54:39 UTC

Skill di riferimento: ~/.claude/skills/rampado-protocollo-polarizzazione-rz

Output in una cartella nuova, protocollo_v1_sand_point/:
    consegna/   6 file grezzi: per stazione 1 miniSEED (BH1+BH2+BHZ,
                record originali concatenati) + 1 StationXML, MANIFEST
    analisi/    report.txt, risultati.csv, parametri.json, figura

Regole congelate (dal brief):
    Dati            BH1, BH2, BHZ + StationXML
    Trasformazione  rotazione Z/R/T; banda 0.03-0.10 Hz
    Ricerca         finestra di velocita' 2.6-3.6 km/s; picco con media mobile 60 s
    Valutazione     ellitticita' R-Z, energia trasversa, punteggio Q, distanza dai bordi

Il brief NON definisce formula di Q, soglie, margine di bordo, rimozione
della risposta e location. I valori usati qui sono PROVVISORI (vedi
PARAMETRI_PROVVISORI): la classificazione ufficiale spetta al catalog
runner v1.0 di Andrea.
"""

import csv
import hashlib
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

from obspy import read, read_inventory, UTCDateTime
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "seismic_data")
OUT = os.path.join(BASE, "protocollo_v1_sand_point")
CONSEGNA = os.path.join(OUT, "consegna")
ANALISI = os.path.join(OUT, "analisi")

# Evento (USGS us6000c9hg)
ORIGIN = UTCDateTime("2020-10-19T20:54:39")
EV_LAT, EV_LON, EV_DEP = 54.662, -159.675, 28.4

# Copertura richiesta dal brief
START = UTCDateTime("2020-10-19T20:54:00")
END = UTCDateTime("2020-10-19T22:20:00")

NET = "IU"
STATIONS = ["ANMO", "TUC", "HRV"]
CHANNELS = ["BH1", "BH2", "BHZ"]

# --- Regole congelate v1.0 (dal brief) --------------------------------------
BAND_HZ = (0.03, 0.10)
VEL_WINDOW_KMS = (2.6, 3.6)
MOVING_AVG_S = 60.0

# --- Parametri NON definiti nel brief: PROVVISORI ---------------------------
LOC = "00"                              # sensore primario GSN
PRE_FILT = (0.01, 0.02, 0.2, 0.4)       # Hz, per remove_response
SEGMENT_HALF_S = 60.0                   # metriche su cresta +/- 60 s
EDGE_MARGIN_S = 60.0                    # "troppo vicina al bordo"
RETRO_DPHI_DEG = (-120.0, -60.0)        # Rayleigh retrogrado: -90 +/- 30
MAX_ET = 0.5                            # energia trasversa massima
MIN_Q = 0.5

PARAMETRI_PROVVISORI = {
    "location": f"{LOC} (sensore primario GSN)",
    "risposta_strumentale": f"rimossa -> velocita' (m/s), pre_filt {PRE_FILT} Hz",
    "curva_energia_cresta": "R^2 + Z^2, media mobile 60 s (regola v1.0)",
    "segmento_metriche": f"cresta +/- {SEGMENT_HALF_S:.0f} s, dentro la finestra",
    "margine_bordo_s": EDGE_MARGIN_S,
    "sfasamento_retrogrado_deg": list(RETRO_DPHI_DEG),
    "max_frazione_energia_T": MAX_ET,
    "min_Q": MIN_Q,
    "formula_Q": "coerenza di fase Z-R x |sin(dphi)| x (1 - E_T/E_tot)",
    "classi": {
        "replica interna": "metriche favorevoli e distanza dal bordo >= margine",
        "marginale": "metriche favorevoli ma distanza dal bordo < margine",
        "non replica": "dati validi ma metriche non favorevoli",
        "non classificabile": "verifica dei dati fallita o finestra fuori copertura",
    },
}


# ============================================================================
# 1. CONSEGNA: sei file grezzi
# ============================================================================

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_delivery(sta):
    """Concatena i record miniSEED originali di BH1/BH2/BHZ e copia lo StationXML."""
    src_dir = os.path.join(DATA, f"{NET}_{sta}")
    sources = [os.path.join(src_dir, f"{NET}_{sta}_{LOC}_{c}.mseed") for c in CHANNELS]

    mseed = os.path.join(CONSEGNA, f"{NET}_{sta}_{LOC}_BH1_BH2_BHZ.mseed")
    with open(mseed, "wb") as out:
        for src in sources:
            with open(src, "rb") as f:
                out.write(f.read())

    xml = os.path.join(CONSEGNA, f"{NET}_{sta}_{LOC}_StationXML.xml")
    shutil.copyfile(os.path.join(src_dir, f"{NET}_{sta}_{LOC}_StationXML.xml"), xml)

    return mseed, xml, sources


def verify(mseed, xml):
    """Verifica alla ricezione: componenti, orari, risposta, continuita'."""
    st = read(mseed)
    inv = read_inventory(xml)
    problems = []
    lines = []

    chans = sorted(tr.stats.channel for tr in st)
    if chans != sorted(CHANNELS):
        problems.append(f"canali {chans} invece di {CHANNELS}")

    gaps = st.get_gaps()
    if gaps:
        problems.append(f"{len(gaps)} gap/overlap")

    for tr in st:
        d = tr.stats.delta
        if tr.stats.starttime > START + d:
            problems.append(f"{tr.id} inizia alle {tr.stats.starttime}")
        if tr.stats.endtime < END - 2 * d:
            problems.append(f"{tr.id} finisce alle {tr.stats.endtime}")
        try:
            inv.get_response(tr.id, ORIGIN)
            resp = "risposta OK"
        except Exception as e:
            resp = "risposta MANCANTE"
            problems.append(f"{tr.id} senza risposta: {e}")
        lines.append(
            f"    {tr.id}  {tr.stats.starttime} -> {tr.stats.endtime}  "
            f"{tr.stats.sampling_rate:g} Hz  {tr.stats.npts} campioni  {resp}"
        )

    return st, inv, problems, lines


# ============================================================================
# 2. ANALISI: protocollo v1.0
# ============================================================================

def analyse(sta, st, inv):
    st = st.copy()
    st.detrend("demean")
    st.detrend("linear")
    st.taper(0.05)
    st.remove_response(inventory=inv, output="VEL", pre_filt=PRE_FILT)
    st.rotate("->ZNE", inventory=inv)

    coords = inv.get_coordinates(st[0].id, ORIGIN)
    dist_m, _, baz = gps2dist_azimuth(EV_LAT, EV_LON,
                                      coords["latitude"], coords["longitude"])
    dist_km = dist_m / 1000.0
    st.rotate("NE->RT", back_azimuth=baz)
    st.filter("bandpass", freqmin=BAND_HZ[0], freqmax=BAND_HZ[1],
              corners=4, zerophase=True)

    Z, R, T = (st.select(component=c)[0] for c in "ZRT")
    n = min(len(Z.data), len(R.data), len(T.data))
    t = Z.times(reftime=ORIGIN)[:n]
    z, r, tt = Z.data[:n], R.data[:n], T.data[:n]
    fs = Z.stats.sampling_rate

    # Finestra di velocita' 2.6-3.6 km/s
    w0, w1 = dist_km / VEL_WINDOW_KMS[1], dist_km / VEL_WINDOW_KMS[0]
    res = dict(sta=sta, dist_km=dist_km, deg=kilometers2degrees(dist_km),
               baz=baz, w0=w0, w1=w1, fs=fs)

    if w1 > t[-1] or w0 < t[0]:
        res.update(classe="non classificabile",
                   motivo="finestra di velocita' fuori dalla copertura dei dati")
        return res, None

    # Cresta: massimo della media mobile 60 s della curva R^2 + Z^2
    energy = r ** 2 + z ** 2
    smooth = uniform_filter1d(energy, size=int(round(MOVING_AVG_S * fs)), mode="nearest")
    inside = np.where((t >= w0) & (t <= w1))[0]
    ic = inside[np.argmax(smooth[inside])]
    tc = t[ic]
    edge = min(tc - w0, w1 - tc)

    # Metriche sul segmento attorno alla cresta
    seg = (t >= max(w0, tc - SEGMENT_HALF_S)) & (t <= min(w1, tc + SEGMENT_HALF_S))
    hz, hr = hilbert(z), hilbert(r)                 # segnale analitico su tutta la traccia
    hz, hr = hz[seg], hr[seg]
    weight = np.abs(hz) * np.abs(hr)
    resultant = np.sum(weight * np.exp(1j * np.angle(hz * np.conj(hr)))) / np.sum(weight)
    dphi = float(np.degrees(np.angle(resultant)))   # fase Z - fase R
    coherence = float(np.abs(resultant))

    zs, rs, ts = z[seg], r[seg], tt[seg]
    ez, er, et = np.sum(zs ** 2), np.sum(rs ** 2), np.sum(ts ** 2)
    hv = float(np.sqrt(er / ez))
    frac_t = float(et / (ez + er + et))
    q = coherence * abs(np.sin(np.radians(dphi))) * (1.0 - frac_t)

    # Classificazione PROVVISORIA
    reasons = []
    if not RETRO_DPHI_DEG[0] <= dphi <= RETRO_DPHI_DEG[1]:
        reasons.append(f"sfasamento {dphi:+.0f}° fuori da {RETRO_DPHI_DEG}")
    if frac_t > MAX_ET:
        reasons.append(f"energia T {frac_t:.2f} > {MAX_ET}")
    if q < MIN_Q:
        reasons.append(f"Q {q:.2f} < {MIN_Q}")

    if reasons:
        classe, motivo = "non replica", "; ".join(reasons)
    elif edge < EDGE_MARGIN_S:
        classe = "marginale"
        motivo = f"cresta a {edge:.0f} s dal bordo (< {EDGE_MARGIN_S:.0f} s)"
    else:
        classe = "replica interna"
        motivo = f"metriche favorevoli, cresta a {edge:.0f} s dal bordo"

    res.update(tc=tc, t_cresta_utc=str(ORIGIN + tc), v_cresta=dist_km / tc,
               edge=edge, dphi=dphi, coherence=coherence, hv=hv,
               frac_t=frac_t, q=q, senso="retrogrado" if dphi < 0 else "progrado",
               classe=classe, motivo=motivo)

    plot = dict(t=t, z=z, r=r, tt=tt, smooth=smooth, seg=seg)
    return res, plot


def make_figure(results, plots):
    fig, axes = plt.subplots(len(results), 2, figsize=(15, 4.2 * len(results)),
                             gridspec_kw=dict(width_ratios=[3, 1]))
    for i, (res, p) in enumerate(zip(results, plots)):
        ax, axp = axes[i]
        if p is None:
            ax.set_title(f"IU.{res['sta']} - {res['classe']}: {res['motivo']}")
            continue
        t = p["t"]
        view = (t >= res["w0"] - 400) & (t <= res["w1"] + 400)
        scale = max(np.max(np.abs(x[view])) for x in (p["z"], p["r"], p["tt"]))
        for k, (x, lab, col) in enumerate(((p["z"], "Z", "k"),
                                          (p["r"], "R", "tab:blue"),
                                          (p["tt"], "T", "tab:red"))):
            ax.plot(t[view], x[view] / scale - 2.2 * k, col, lw=0.6)
        sm = p["smooth"][view] / np.max(p["smooth"][view])
        ax.plot(t[view], sm * 1.8 - 6.8, "tab:green", lw=1.2)
        ax.axvspan(res["w0"], res["w1"], color="0.85", zorder=0)
        ax.axvspan(res["w0"], res["w0"] + EDGE_MARGIN_S, color="tab:orange", alpha=0.25, zorder=0)
        ax.axvspan(res["w1"] - EDGE_MARGIN_S, res["w1"], color="tab:orange", alpha=0.25, zorder=0)
        ax.axvline(res["tc"], color="tab:purple", lw=1.2)
        ax.set_yticks([0, -2.2, -4.4, -6.0])
        ax.set_yticklabels(["Z", "R", "T", "E media 60 s"])
        ax.set_title(f"IU.{res['sta']}  Δ={res['deg']:.1f}°  finestra 3,6-2,6 km/s "
                     f"({res['w0']:.0f}-{res['w1']:.0f} s)  cresta {res['tc']:.0f} s "
                     f"({res['v_cresta']:.2f} km/s)  ->  {res['classe']}", fontsize=9)
        ax.set_xlabel("tempo dall'origine (s)")

        seg = p["seg"]
        axp.plot(p["r"][seg] * 1e6, p["z"][seg] * 1e6, lw=0.7, c="tab:blue")
        axp.set_aspect("equal", "datalim")
        axp.set_xlabel("R (µm/s)")
        axp.set_ylabel("Z (µm/s)")
        axp.set_title(f"R-Z cresta ±{SEGMENT_HALF_S:.0f} s\nΔφ={res['dphi']:+.0f}°  "
                      f"H/V={res['hv']:.2f}  Q={res['q']:.2f}", fontsize=9)
    fig.suptitle("Protocollo v1.0 - M7.6 Sand Point 2020-10-19 - banda 0,03-0,10 Hz "
                 "(arancio = margine di bordo PROVVISORIO)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(ANALISI, "figura_protocollo_v1.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(CONSEGNA, exist_ok=True)
    os.makedirs(ANALISI, exist_ok=True)

    manifest = ["MANIFEST - consegna dati grezzi (brief Rampado, protocollo v1.0)",
                "Evento: M7.6 Sand Point, Alaska - 2020-10-19, finestra 20:54 -> 22:20 UTC",
                f"Fonte: EarthScope FDSN (URL in seismic_data/download_urls.txt), location {LOC}",
                "miniSEED = record originali di BH1, BH2, BHZ concatenati, nessuna elaborazione",
                ""]
    verifica = []
    results, plots = [], []

    for sta in STATIONS:
        mseed, xml, sources = build_delivery(sta)
        for path in (mseed, xml):
            manifest.append(f"{sha256(path)}  {os.path.getsize(path):>9,} B  "
                            f"{os.path.basename(path)}")
        for src in sources:
            manifest.append(f"    sorgente: {os.path.relpath(src, BASE)}  sha256 {sha256(src)[:16]}...")

        st, inv, problems, lines = verify(mseed, xml)
        verifica.append(f"IU.{sta}: {'OK' if not problems else 'PROBLEMI'}")
        verifica += lines
        verifica += [f"    PROBLEMA: {p}" for p in problems]

        if problems:
            res, plot = dict(sta=sta, classe="non classificabile",
                             motivo="; ".join(problems)), None
        else:
            res, plot = analyse(sta, st, inv)
        results.append(res)
        plots.append(plot)
        print(f"IU.{sta}: {res['classe']} - {res['motivo']}")

    with open(os.path.join(CONSEGNA, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")

    fig_path = make_figure(results, plots)

    # CSV
    cols = ["sta", "dist_km", "deg", "baz", "w0", "w1", "tc", "t_cresta_utc",
            "v_cresta", "edge", "dphi", "coherence", "hv", "frac_t", "q",
            "senso", "classe", "motivo"]
    with open(os.path.join(ANALISI, "risultati.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for res in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in res.items() if k in cols})

    with open(os.path.join(ANALISI, "parametri.json"), "w", encoding="utf-8") as f:
        json.dump({"regole_congelate_v1.0": {"canali": CHANNELS,
                                              "banda_hz": BAND_HZ,
                                              "finestra_velocita_kms": VEL_WINDOW_KMS,
                                              "media_mobile_s": MOVING_AVG_S},
                   "parametri_provvisori": PARAMETRI_PROVVISORI},
                  f, indent=2, ensure_ascii=False)

    # Report
    counts = {}
    for res in results:
        counts[res["classe"]] = counts.get(res["classe"], 0) + 1

    lines = [
        "RICALCOLO PROTOCOLLO v1.0 - M7.6 Sand Point, Alaska - 2020-10-19 20:54:39 UTC",
        "Skill: rampado-protocollo-polarizzazione-rz (brief A. Rampado, 11/09/2026)",
        "",
        "Regole congelate: BH1/BH2/BHZ + StationXML; rotazione Z/R/T; banda 0,03-0,10 Hz;",
        "finestra 2,6-3,6 km/s; cresta con media mobile 60 s; ellitticita' R-Z,",
        "energia trasversa, punteggio Q, distanza dai bordi.",
        "",
        "ATTENZIONE: formula di Q, soglie, margine di bordo, rimozione della risposta e",
        "location NON sono definiti nel brief. Qui sono PROVVISORI (analisi/parametri.json).",
        "La classificazione ufficiale spetta al catalog runner v1.0.",
        "",
        "VERIFICA DELLA CONSEGNA (componenti, orari, risposta, continuita')",
        *verifica,
        "",
        "RISULTATI",
    ]
    for res in results:
        lines.append(f"IU.{res['sta']}: {res['classe'].upper()} - {res['motivo']}")
        if "tc" in res:
            lines += [
                f"  Δ={res['deg']:.1f}° ({res['dist_km']:.0f} km)  BAZ={res['baz']:.0f}°  "
                f"finestra {res['w0']:.0f}-{res['w1']:.0f} s",
                f"  cresta {res['tc']:.0f} s ({res['t_cresta_utc'][11:19]} UTC, "
                f"{res['v_cresta']:.2f} km/s)  distanza dal bordo {res['edge']:.0f} s",
                f"  ellitticita' R-Z: Δφ(Z-R)={res['dphi']:+.1f}° ({res['senso']})  "
                f"H/V={res['hv']:.2f}  coerenza di fase={res['coherence']:.2f}",
                f"  energia trasversa E_T/E_tot={res['frac_t']:.2f}  Q={res['q']:.2f}",
            ]
        lines.append("")
    lines.append("RIEPILOGO: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    lines += ["",
              "Il risultato descrive una polarizzazione coerente (o no) con onde di",
              "superficie di Rayleigh; non dimostra nuova fisica."]
    report = "\n".join(lines)
    with open(os.path.join(ANALISI, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print()
    print(report)
    print()
    print(f"Output: {OUT}")
    print(f"Figura: {fig_path}")


if __name__ == "__main__":
    main()
