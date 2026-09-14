#!/usr/bin/env python3
"""
Analisi della polarizzazione delle onde di superficie
Terremoto M7.6 Alaska (Sand Point), 2020-10-19 20:54:39 UTC.

Per ogni stazione (location 00):
  1. rimozione risposta strumentale -> velocità (m/s)
  2. rotazione BH1/BH2/BHZ -> Z/N/E -> Z/R/T
  3. filtro passa-banda 20-100 s (onde di superficie)
  4. finestre Love (4.6-3.9 km/s) e Rayleigh (4.1-3.2 km/s)
  5. polarizzazione: sfasamento Z-R (Hilbert), ellitticità H/V,
     senso di rotazione (retrogrado/progrado), rapporto energia T/(R+Z)
  6. figure: sismogrammi ZRT, particle motion 2D, grafico 3D.
"""

import os
import sys
import glob

sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252: "Δ", "µ"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from obspy import read, read_inventory, UTCDateTime, Stream
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.taup import TauPyModel

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "seismic_data")
OUT = os.path.join(BASE, "risultati_polarizzazione")
os.makedirs(OUT, exist_ok=True)

# Evento (USGS us6000c9hg)
ORIGIN = UTCDateTime("2020-10-19T20:54:39")
EV_LAT, EV_LON, EV_DEP = 54.662, -159.675, 28.4

STATIONS = ["ANMO", "TUC", "HRV"]
LOC = "00"
FMIN, FMAX = 0.01, 0.05          # 100 s - 20 s
LOVE_V = (4.6, 3.9)              # km/s (finestra di velocità di gruppo)
RAYL_V = (4.1, 3.2)

model = TauPyModel("iasp91")


def load_station(sta):
    d = os.path.join(DATA, f"IU_{sta}")
    st = Stream()
    for f in sorted(glob.glob(os.path.join(d, f"IU_{sta}_{LOC}_BH?.mseed"))):
        st += read(f)
    inv = read_inventory(os.path.join(d, f"IU_{sta}_{LOC}_StationXML.xml"))
    return st, inv


def process(st, inv):
    st = st.copy()
    st.merge(fill_value=0)
    st.detrend("demean")
    st.detrend("linear")
    st.taper(0.05)
    st.remove_response(inventory=inv, output="VEL",
                       pre_filt=(0.005, 0.008, 5.0, 8.0))
    st.rotate("->ZNE", inventory=inv)
    coords = inv.get_coordinates(st[0].id)
    dist_m, az, baz = gps2dist_azimuth(EV_LAT, EV_LON,
                                       coords["latitude"], coords["longitude"])
    st.rotate("NE->RT", back_azimuth=baz)
    st.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=True)
    return st, dist_m / 1000.0, baz, coords


def window(tr, t0, t1):
    t = tr.times(reftime=ORIGIN)
    m = (t >= t0) & (t <= t1)
    return t[m], tr.data[m]


def polarization(z, r):
    """Sfasamento Z-R e senso di rotazione dal segnale analitico."""
    hz, hr = hilbert(z), hilbert(r)
    env = np.abs(hz) * np.abs(hr)
    dphi = np.angle(hz * np.conj(hr))            # fase Z - fase R
    # media circolare pesata con l'ampiezza
    mean_dphi = np.degrees(np.angle(np.sum(env * np.exp(1j * dphi))))
    hv = np.sqrt(np.sum(r ** 2) / np.sum(z ** 2))
    # Convenzione ObsPy: R positivo = da sorgente verso stazione, Z su.
    # Moto retrogrado: in cima all'ellisse la particella torna verso la
    # sorgente -> Z=cos(wt), R=-sin(wt) -> fase(Z)-fase(R) = -90°.
    sense = "retrogrado" if mean_dphi < 0 else "progrado"
    return mean_dphi, hv, sense


def linearity(x, y, z):
    """Grado di polarizzazione dalla matrice di covarianza (Jurkevics)."""
    c = np.cov(np.vstack([x, y, z]))
    w, v = np.linalg.eigh(c)
    w = w[::-1]
    v = v[:, ::-1]
    rect = 1 - (w[1] + w[2]) / (2 * w[0])
    planarity = 1 - 2 * w[2] / (w[0] + w[1])
    return rect, planarity, v[:, 0]


results = []
fig_all, axes_all = plt.subplots(len(STATIONS), 1, figsize=(13, 10), sharex=False)

for i, sta in enumerate(STATIONS):
    raw, inv = load_station(sta)
    st, dist_km, baz, coords = process(raw, inv)
    deg = kilometers2degrees(dist_km)
    Z = st.select(component="Z")[0]
    R = st.select(component="R")[0]
    T = st.select(component="T")[0]

    arr = model.get_travel_times(EV_DEP, deg, phase_list=["P", "S"])
    tP = next((a.time for a in arr if a.name == "P"), None)
    tS = next((a.time for a in arr if a.name == "S"), None)

    l0, l1 = dist_km / LOVE_V[0], dist_km / LOVE_V[1]
    r0, r1 = dist_km / RAYL_V[0], dist_km / RAYL_V[1]

    tz, z = window(Z, r0, r1)
    _, r = window(R, r0, r1)
    _, tt = window(T, r0, r1)
    tl, tlove = window(T, l0, l1)
    _, rl = window(R, l0, l1)
    _, zl = window(Z, l0, l1)

    dphi, hv, sense = polarization(z, r)
    rect_R, plan_R, vR = linearity(r, tt, z)
    rect_L, plan_L, vL = linearity(rl, tlove, zl)
    eT_love = np.sum(tlove ** 2) / (np.sum(rl ** 2) + np.sum(zl ** 2) + np.sum(tlove ** 2))
    eT_rayl = np.sum(tt ** 2) / (np.sum(r ** 2) + np.sum(z ** 2) + np.sum(tt ** 2))

    res = dict(sta=sta, dist_km=dist_km, deg=deg, baz=baz, tP=tP, tS=tS,
               love=(l0, l1), rayl=(r0, r1), dphi=dphi, hv=hv, sense=sense,
               rect_R=rect_R, plan_R=plan_R, rect_L=rect_L,
               eT_love=eT_love, eT_rayl=eT_rayl,
               pgv={c: np.max(np.abs(tr.data)) for c, tr in zip("ZRT", (Z, R, T))},
               data=(tz, r, tt, z, tl, rl, tlove, zl), traces=(Z, R, T))
    results.append(res)

    # --- Sismogrammi ZRT ---------------------------------------------------
    ax = axes_all[i]
    t = Z.times(reftime=ORIGIN)
    scale = max(res["pgv"].values())
    for k, (tr, col) in enumerate(zip((Z, R, T), ("k", "tab:blue", "tab:red"))):
        ax.plot(t, tr.data / scale - 2.2 * k, col, lw=0.6, label=tr.stats.channel)
    ax.axvspan(l0, l1, color="tab:red", alpha=0.12, label="finestra Love")
    ax.axvspan(r0, r1, color="tab:blue", alpha=0.12, label="finestra Rayleigh")
    if tP:
        ax.axvline(tP, color="g", ls="--", lw=0.8)
        ax.text(tP, 1.0, " P", color="g")
    if tS:
        ax.axvline(tS, color="m", ls="--", lw=0.8)
        ax.text(tS, 1.0, " S", color="m")
    ax.set_yticks([0, -2.2, -4.4])
    ax.set_yticklabels(["Z", "R", "T"])
    ax.set_title(f"IU.{sta}  Δ={deg:.1f}°  ({dist_km:.0f} km)  BAZ={baz:.0f}°  "
                 f"banda {1/FMAX:.0f}-{1/FMIN:.0f} s")
    ax.set_xlim(max(0, (tP or 0) - 60), t[-1])
    if i == 0:
        ax.legend(loc="upper right", fontsize=7, ncol=5)
axes_all[-1].set_xlabel("Tempo dall'origine (s)")
fig_all.tight_layout()
fig_all.savefig(os.path.join(OUT, "01_sismogrammi_ZRT.png"), dpi=140)
plt.close(fig_all)

# --- Particle motion 2D ---------------------------------------------------
fig, axes = plt.subplots(len(results), 3, figsize=(13, 4 * len(results)))
for i, res in enumerate(results):
    tz, r, tt, z, tl, rl, tlove, zl = res["data"]
    a = axes[i]
    for ax, (x, y, xl, yl, ttl, tc) in zip(a, [
            (r, z, "R", "Z", "Rayleigh: piano R-Z", tz),
            (r, tt, "R", "T", "Rayleigh: piano R-T", tz),
            (rl, tlove, "R", "T", "Love: piano R-T", tl)]):
        sc = ax.scatter(x * 1e6, y * 1e6, c=tc, s=2, cmap="viridis")
        ax.set_xlabel(f"{xl} (µm/s)")
        ax.set_ylabel(f"{yl} (µm/s)")
        ax.set_title(f"IU.{res['sta']} – {ttl}", fontsize=9)
        ax.set_aspect("equal", "datalim")
        ax.axhline(0, c="0.7", lw=0.5)
        ax.axvline(0, c="0.7", lw=0.5)
    a[0].text(0.02, 0.97, f"Δφ(Z-R)={res['dphi']:+.0f}°\nH/V={res['hv']:.2f}\n{res['sense']}",
              transform=a[0].transAxes, va="top", fontsize=8,
              bbox=dict(fc="w", alpha=0.8))
    a[2].text(0.02, 0.97, f"E_T/E_tot={res['eT_love']:.2f}\nrettilinearità={res['rect_L']:.2f}",
              transform=a[2].transAxes, va="top", fontsize=8,
              bbox=dict(fc="w", alpha=0.8))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_particle_motion_2D.png"), dpi=130)
plt.close(fig)

# --- Grafico 3D del moto delle particelle -------------------------------
fig = plt.figure(figsize=(16, 6))
for i, res in enumerate(results):
    Z, R, T = res["traces"]
    t = Z.times(reftime=ORIGIN)
    t0, t1 = res["love"][0], res["rayl"][1]
    m = (t >= t0) & (t <= t1)
    ax = fig.add_subplot(1, len(results), i + 1, projection="3d")
    s = 1e6
    # colore: rosso = finestra Love, blu = finestra Rayleigh
    tl = t[m]
    colors = np.where(tl < res["rayl"][0], "tab:red", "tab:blue")
    ax.scatter(R.data[m] * s, T.data[m] * s, Z.data[m] * s, c=colors, s=1.5, alpha=0.7)
    ax.set_xlabel("R (µm/s)")
    ax.set_ylabel("T (µm/s)")
    ax.set_zlabel("Z (µm/s)")
    ax.set_title(f"IU.{res['sta']}  Δ={res['deg']:.0f}°\n"
                 "rosso=Love (T)  blu=Rayleigh (R-Z)", fontsize=9)
    ax.view_init(elev=20, azim=-60)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_particle_motion_3D.png"), dpi=130)
plt.close(fig)

# --- 3D tempo-ampiezza (traiettoria elicoidale R-Z nel tempo) ------------
fig = plt.figure(figsize=(16, 6))
for i, res in enumerate(results):
    Z, R, T = res["traces"]
    t = Z.times(reftime=ORIGIN)
    r0, r1 = res["rayl"]
    m = (t >= r0) & (t <= r1)
    ax = fig.add_subplot(1, len(results), i + 1, projection="3d")
    ax.plot(t[m], R.data[m] * 1e6, Z.data[m] * 1e6, lw=0.6, c="tab:blue")
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("R (µm/s)")
    ax.set_zlabel("Z (µm/s)")
    ax.set_title(f"IU.{res['sta']} – Rayleigh: elica R-Z nel tempo\n"
                 f"Δφ={res['dphi']:+.0f}°  ({res['sense']})", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_rayleigh_elica_3D.png"), dpi=130)
plt.close(fig)

# --- Report -------------------------------------------------------------
lines = ["ANALISI POLARIZZAZIONE ONDE DI SUPERFICIE - M7.6 Alaska 2020-10-19",
         f"Banda: {1/FMAX:.0f}-{1/FMIN:.0f} s   Location: {LOC}", ""]
for res in results:
    lines += [
        f"IU.{res['sta']}: Δ={res['deg']:.1f}° ({res['dist_km']:.0f} km), BAZ={res['baz']:.0f}°",
        f"  P={res['tP']:.0f}s  S={res['tS']:.0f}s  Love {res['love'][0]:.0f}-{res['love'][1]:.0f}s"
        f"  Rayleigh {res['rayl'][0]:.0f}-{res['rayl'][1]:.0f}s",
        f"  Rayleigh: Δφ(Z-R)={res['dphi']:+.1f}°  H/V={res['hv']:.2f}  moto {res['sense']}"
        f"  planarità={res['plan_R']:.2f}  E_T/E_tot={res['eT_rayl']:.2f}",
        f"  Love:     E_T/E_tot={res['eT_love']:.2f}  rettilinearità={res['rect_L']:.2f}",
        f"  PGV (banda): Z={res['pgv']['Z']*1e6:.1f}  R={res['pgv']['R']*1e6:.1f}"
        f"  T={res['pgv']['T']*1e6:.1f} µm/s", ""]
report = "\n".join(lines)
print(report)
with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
