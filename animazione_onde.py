#!/usr/bin/env python3
"""
Animazione 3D delle onde sismiche - M7.6 Alaska 2020-10-19.

Pannelli:
  - Globo 3D: epicentro, stazioni, percorsi, fronti d'onda P, S, Love, Rayleigh
    che si espandono; il marker di ogni stazione pulsa con l'energia ricevuta.
  - Per ogni stazione: moto 3D del suolo (R, T, Z) con scia recente e
    freccia = vettore velocità istantaneo; barre = ripartizione energia Z/R/T.
  - Striscia in basso: inviluppo dell'energia nel tempo con cursore.

Output: risultati_polarizzazione/05_animazione_onde_3D.gif
"""

import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import hilbert

from obspy import read, read_inventory, UTCDateTime, Stream
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "seismic_data")
OUT = os.path.join(BASE, "risultati_polarizzazione")
os.makedirs(OUT, exist_ok=True)

ORIGIN = UTCDateTime("2020-10-19T20:54:39")
EV_LAT, EV_LON = 54.662, -159.675
STATIONS = ["ANMO", "TUC", "HRV"]
COLORS = {"ANMO": "tab:orange", "TUC": "tab:green", "HRV": "tab:purple"}
LOC = "00"
FMIN, FMAX = 0.01, 0.05
R_EARTH = 6371.0

EV_DEP = 28.4
# Fronti: P e S dalle curve tempo-distanza TauP (onde di volume, viaggiano in
# profondità); Love e Rayleigh con velocità di gruppo indicative (km/s).
FRONTS = {"P": (None, "tab:green"), "S": (None, "tab:pink"),
          "Love": (4.25, "tab:red"), "Rayleigh": (3.6, "tab:blue")}
VIEW_ELEV, VIEW_AZIM = 45, -125     # vista fissa centrata su Alaska - Nord America

T_START, T_END, DT_FRAME = 300, 2300, 10     # secondi dall'origine
TRAIL = 150                                  # s di scia nel moto del suolo
FS = 1.0                                     # campionamento per l'animazione


def load(sta):
    d = os.path.join(DATA, f"IU_{sta}")
    st = Stream()
    for f in sorted(glob.glob(os.path.join(d, f"IU_{sta}_{LOC}_BH?.mseed"))):
        st += read(f)
    inv = read_inventory(os.path.join(d, f"IU_{sta}_{LOC}_StationXML.xml"))
    st.merge(fill_value=0)
    st.detrend("demean"); st.detrend("linear"); st.taper(0.05)
    st.remove_response(inventory=inv, output="VEL",
                       pre_filt=(0.005, 0.008, 5.0, 8.0))
    st.rotate("->ZNE", inventory=inv)
    c = inv.get_coordinates(st[0].id)
    dist_m, _, baz = gps2dist_azimuth(EV_LAT, EV_LON, c["latitude"], c["longitude"])
    st.rotate("NE->RT", back_azimuth=baz)
    st.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=True)
    st.resample(FS)
    sd = st.copy().integrate()                      # spostamento (µm) per la superficie
    sd.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=True)
    tr = st.select(component="Z")[0]
    t = tr.times(reftime=ORIGIN)
    Z = tr.data * 1e6
    R = st.select(component="R")[0].data * 1e6
    T = st.select(component="T")[0].data * 1e6
    env = np.sqrt(np.abs(hilbert(Z)) ** 2 + np.abs(hilbert(R)) ** 2 + np.abs(hilbert(T)) ** 2)
    return dict(sta=sta, lat=c["latitude"], lon=c["longitude"], dist=dist_m / 1000,
                baz=baz, t=t, Z=Z, R=R, T=T, env=env,
                Zd=sd.select(component="Z")[0].data * 1e6,
                Rd=sd.select(component="R")[0].data * 1e6)


def xyz(lat, lon, r=1.0):
    la, lo = np.radians(lat), np.radians(lon)
    return r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)


def small_circle(lat0, lon0, delta, n=180, az=None):
    """Punti a distanza angolare delta (rad) dall'epicentro."""
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    if az is None:
        az = np.linspace(0, 2 * np.pi, n)
    la = np.arcsin(np.sin(la0) * np.cos(delta) + np.cos(la0) * np.sin(delta) * np.cos(az))
    lo = lo0 + np.arctan2(np.sin(az) * np.sin(delta) * np.cos(la0),
                          np.cos(delta) - np.sin(la0) * np.sin(la))
    return xyz(np.degrees(la), np.degrees(lo), 1.005)


def great_circle(lat1, lon1, lat2, lon2, n=100):
    p1, p2 = np.array(xyz(lat1, lon1)), np.array(xyz(lat2, lon2))
    om = np.arccos(np.clip(p1 @ p2, -1, 1))
    f = np.linspace(0, 1, n)
    pts = (np.sin((1 - f) * om)[:, None] * p1 + np.sin(f * om)[:, None] * p2) / np.sin(om)
    return pts.T * 1.005


def travel_curve(phase):
    """Distanza angolare (rad) raggiunta dal fronte di volume al tempo t."""
    degs = np.arange(1, 98, 1.0)
    times = []
    for g in degs:
        a = model.get_travel_times(EV_DEP, g, phase_list=[phase])
        times.append(min(x.time for x in a) if a else np.nan)
    times = np.array(times)
    ok = np.isfinite(times)
    return lambda t: np.radians(np.interp(t, times[ok], degs[ok], left=0, right=np.nan))


def facing(x, y, z):
    """Nasconde (NaN) i punti sull'emisfero non visibile."""
    e, a = np.radians(VIEW_ELEV), np.radians(VIEW_AZIM)
    v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    hide = x * v[0] + y * v[1] + z * v[2] < 0
    x, y, z = (np.where(hide, np.nan, c) for c in (x, y, z))
    return x, y, z


def rotation(d, t_now, win=40):
    """Verso di rotazione nel piano R-Z negli ultimi `win` s.
    Con R positivo verso la stazione e Z in alto:
    R*dZ - Z*dR > 0  -> retrogrado (tipico Rayleigh),  < 0 -> progrado.
    Restituisce un valore in [-1, 1] (coerenza della rotazione)."""
    m = (d["t"] > t_now - win) & (d["t"] <= t_now)
    if m.sum() < 5:
        return 0.0
    R, Z = d["R"][m], d["Z"][m]
    dR, dZ = np.gradient(R), np.gradient(Z)
    num = np.sum(R * dZ - Z * dR)
    den = np.sum(np.abs(R * dZ) + np.abs(Z * dR)) + 1e-12
    return num / den


def phase_label(d, t):
    """Quale onda sta arrivando alla stazione al tempo t."""
    if t < d["tP"]:
        return "rumore / attesa", "0.5"
    if t < d["tS"]:
        return "onde P (volume)", "tab:green"
    if t < d["dist"] / 4.6:
        return "onde S (volume)", "tab:pink"
    if t < d["dist"] / 4.0:
        return "LOVE (orizz. trasversale)", "tab:red"
    if t < d["dist"] / 3.0:
        return "RAYLEIGH (ellisse R-Z)", "tab:blue"
    return "coda", "0.4"


print("Carico e processo i dati...")
model = TauPyModel("iasp91")
data = [load(s) for s in STATIONS]
for d in data:
    g = d["dist"] / 111.195
    d["tP"] = model.get_travel_times(EV_DEP, g, phase_list=["P"])[0].time
    d["tS"] = model.get_travel_times(EV_DEP, g, phase_list=["S"])[0].time
    print(f"  IU.{d['sta']}: {d['dist']:.0f} km, BAZ {d['baz']:.0f}°, "
          f"P {d['tP']:.0f} s, S {d['tS']:.0f} s")
CURVES = {"P": travel_curve("P"), "S": travel_curve("S")}

# ------------------------------------------------------------------ figura
fig = plt.figure(figsize=(20, 11), facecolor="#0e1117")
gs = fig.add_gridspec(3, 3, width_ratios=[1.15, 1.45, 0.9], height_ratios=[1, 1, 0.7],
                      left=0.045, right=0.99, top=0.94, bottom=0.06, wspace=0.04, hspace=0.22)
ax_g = fig.add_subplot(gs[0:2, 0], projection="3d")
ax_r = fig.add_subplot(gs[0:2, 1], projection="3d")
ax_s = [fig.add_subplot(gs[i, 2], projection="3d") for i in range(3)]
ax_e = fig.add_subplot(gs[2, 0:2])
title = fig.suptitle("", color="w", fontsize=16)

TXT = "#e6e6e6"
for a in [ax_g, ax_r] + ax_s:
    a.set_facecolor("#0e1117")
    a.xaxis.set_pane_color((0, 0, 0, 0)); a.yaxis.set_pane_color((0, 0, 0, 0))
    a.zaxis.set_pane_color((0, 0, 0, 0))

# --- globo statico: reticolo solo sull'emisfero visibile
ll = np.linspace(-180, 180, 361)
for lat in range(-80, 90, 10):
    ax_g.plot(*facing(*xyz(np.full_like(ll, lat), ll)), color="#2b3a55", lw=0.5)
lt = np.linspace(-90, 90, 181)
for lon in range(-180, 180, 15):
    ax_g.plot(*facing(*xyz(lt, np.full_like(lt, lon))), color="#2b3a55", lw=0.5)
th = np.linspace(0, 2 * np.pi, 361)       # contorno del disco
e_, a_ = np.radians(VIEW_ELEV), np.radians(VIEW_AZIM)
vv = np.array([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_), np.sin(e_)])
b1 = np.cross(vv, [0, 0, 1]); b1 /= np.linalg.norm(b1); b2 = np.cross(vv, b1)
rim = np.outer(np.cos(th), b1) + np.outer(np.sin(th), b2)
ax_g.plot(rim[:, 0], rim[:, 1], rim[:, 2], color="#5a7bb5", lw=1)
ex, ey, ez = xyz(EV_LAT, EV_LON, 1.01)
ax_g.scatter([ex], [ey], [ez], marker="*", s=350, c="yellow", edgecolor="k", zorder=10)
ax_g.text(ex, ey, ez + 0.08, "M7.6 Alaska", color="yellow", fontsize=9)
st_markers = {}
for d in data:
    gx, gy, gz = great_circle(EV_LAT, EV_LON, d["lat"], d["lon"])
    ax_g.plot(gx, gy, gz, color=COLORS[d["sta"]], lw=1, ls=":")
    sx, sy, sz = xyz(d["lat"], d["lon"], 1.01)
    st_markers[d["sta"]] = ax_g.scatter([sx], [sy], [sz], marker="^", s=60,
                                        c=COLORS[d["sta"]], edgecolor="w")
    ax_g.text(sx, sy, sz - 0.1, d["sta"], color=COLORS[d["sta"]], fontsize=9)
front_lines = {k: ax_g.plot([], [], [], color=c, lw=2.2,
                            label=f"fronte {k} " + ("(TauP, volume)" if vel is None
                                                    else f"({vel} km/s, superficie)"))[0]
               for k, (vel, c) in FRONTS.items()}
ax_g.legend(loc="lower left", fontsize=8, facecolor="#0e1117", labelcolor=TXT, framealpha=0.6)
ax_g.set_xlim(-1, 1); ax_g.set_ylim(-1, 1); ax_g.set_zlim(-1, 1)
ax_g.set_box_aspect((1, 1, 1), zoom=1.45)
ax_g.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
ax_g.set_axis_off()

# --- energia nel tempo
ax_e.set_facecolor("#0e1117")
for d in data:
    ax_e.plot(d["t"], d["env"], color=COLORS[d["sta"]], lw=1, label=f"IU.{d['sta']}")
ax_e.set_xlim(T_START, T_END)
ax_e.set_ylim(0, max(d["env"].max() for d in data) * 1.1)
ax_e.set_xlabel("secondi dall'origine (20:54:39 UTC)", color=TXT)
ax_e.set_ylabel("|v| inviluppo (µm/s)", color=TXT)
ax_e.tick_params(colors=TXT)
for s in ax_e.spines.values():
    s.set_color("#444")
ax_e.legend(loc="upper left", fontsize=8, facecolor="#0e1117", labelcolor=TXT)
cursor = ax_e.axvline(T_START, color="w", lw=1.2)

LIM = {d["sta"]: np.max(np.abs(np.r_[d["Z"], d["R"], d["T"]])) * 0.9 for d in data}
ENV_MAX = max(d["env"].max() for d in data)

# --- pannello onda di Rayleigh: superficie deformata con i dati reali di TUC
REF = next(d for d in data if d["sta"] == "TUC")
C_R = 3.6                                     # km/s, velocità di fase indicativa
XS = np.linspace(-270, 270, 61)               # km lungo la direzione di propagazione
YS = np.linspace(-80, 80, 9)                  # km in direzione trasversale
SC = 45.0 / np.max(np.abs(np.r_[REF["Zd"], REF["Rd"]]))   # esagerazione ampiezza
PROP_AZ = (REF["baz"] + 180) % 360            # azimut di propagazione alla stazione
TRACERS = (-180, -90, 0, 90, 180)
ray_arrows = []


def rotation_text(rot, in_ray):
    if in_ray and abs(rot) > 0.3:
        return (f"rotazione R-Z: RETROGRADA ↺  ({rot:+.2f})", "tab:blue") if rot > 0 \
            else (f"rotazione R-Z: PROGRADA ↻  ({rot:+.2f})", "tab:orange")
    return f"rotazione R-Z: {rot:+.2f}", "0.5"


def draw_rayleigh(ax, t_now):
    ax.cla()
    ax.set_facecolor("#0e1117")
    ax.xaxis.set_pane_color((0, 0, 0, 0)); ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    # u(x, t) = u_TUC(t - x / c): il sismogramma reale che scorre lungo x
    tt = t_now - XS / C_R
    rz = np.interp(tt, REF["t"], REF["Zd"]) * SC
    rr = np.interp(tt, REF["t"], REF["Rd"]) * SC
    X = np.broadcast_to(XS + rr, (len(YS), len(XS)))
    Y = np.broadcast_to(YS[:, None], (len(YS), len(XS)))
    Zs = np.broadcast_to(rz, (len(YS), len(XS)))
    ax.plot_surface(X, Y, Zs, cmap="coolwarm", vmin=-45, vmax=45, alpha=0.85,
                    linewidth=0.3, edgecolor="#1b2a41")
    # particelle traccianti sul bordo anteriore: ellissi degli ultimi 30 s
    y0 = YS[0]
    for x0 in TRACERS:
        ts = np.arange(t_now - 30, t_now + 1) - x0 / C_R
        px = x0 + np.interp(ts, REF["t"], REF["Rd"]) * SC
        pz = np.interp(ts, REF["t"], REF["Zd"]) * SC
        ax.plot(px, np.full_like(px, y0), pz, color="yellow", lw=1.4)
        ax.scatter([px[-1]], [y0], [pz[-1]], color="yellow", s=30, depthshade=False)
        v = np.array([px[-1] - px[-2], pz[-1] - pz[-2]])
        n = np.hypot(*v)
        if n > 0.05:
            v = v / n * 20
            ax.quiver(px[-1], y0, pz[-1], v[0], 0, v[1], color="w", lw=1.6,
                      arrow_length_ratio=0.5)
    # direzione e verso di propagazione
    ax.quiver(-250, YS[-1], 80, 330, 0, 0, color="lime", lw=3, arrow_length_ratio=0.1)
    ax.text(-250, YS[-1], 95, f"direzione di propagazione: azimut {PROP_AZ:.0f}° "
            f"(dall'Alaska verso SE)", color="lime", fontsize=10)
    ax.set_xlim(-290, 290); ax.set_ylim(-90, 90); ax.set_zlim(-90, 110)
    ax.set_box_aspect((2.6, 0.9, 1.0), zoom=1.15)
    ax.set_xlabel("x lungo il percorso (km)", color=TXT)
    ax.set_ylabel("trasversale (km)", color=TXT)
    ax.set_zlabel("Z (esagerato)", color=TXT)
    ax.tick_params(colors="#777", labelsize=7)
    ax.view_init(elev=16, azim=-78)

    in_ray = REF["dist"] / 4.0 <= t_now <= REF["dist"] / 3.0
    txt, col = rotation_text(rotation(REF, t_now), in_ray)
    ax.text2D(0.02, 0.97, "ONDA DI RAYLEIGH a IU.TUC — dati reali R-Z (spostamento, 20-100 s)",
              transform=ax.transAxes, color="tab:blue", fontsize=12, weight="bold")
    ax.text2D(0.02, 0.92, phase_label(REF, t_now)[0], transform=ax.transAxes,
              color=phase_label(REF, t_now)[1], fontsize=11, weight="bold")
    ax.text2D(0.02, 0.87, txt, transform=ax.transAxes, color=col, fontsize=11, weight="bold")
    ax.text2D(0.02, 0.03,
              "Giallo: particelle del suolo (scia 30 s)   Bianco: verso istantaneo del moto\n"
              "Moto retrogrado: in cima all'ellisse il suolo va verso l'Alaska, "
              "contro la propagazione (↺ con propagazione a destra)",
              transform=ax.transAxes, color=TXT, fontsize=9)


def draw_station(ax, d, t_now):
    ax.cla()
    ax.set_facecolor("#0e1117")
    ax.xaxis.set_pane_color((0, 0, 0, 0)); ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    L = LIM[d["sta"]]
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
    ax.set_xlabel("R", color=TXT, labelpad=-10); ax.set_ylabel("T", color=TXT, labelpad=-10)
    ax.set_zlabel("Z", color=TXT, labelpad=-10)
    ax.tick_params(colors="#777", labelsize=6, pad=-3)
    # assi di riferimento: R verso stazione (propagazione), T, Z
    ax.plot([-L, L], [0, 0], [0, 0], color="tab:blue", lw=0.5, alpha=0.5)
    ax.plot([0, 0], [-L, L], [0, 0], color="tab:red", lw=0.5, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-L, L], color="w", lw=0.5, alpha=0.5)
    ax.quiver(-L, -L, -L, L * 0.6, 0, 0, color="yellow", lw=1.5, arrow_length_ratio=0.25)
    ax.text(-L * 0.4, -L, -L, "propagazione", color="yellow", fontsize=6)

    t = d["t"]
    m = (t > t_now - TRAIL) & (t <= t_now)
    lab, col = phase_label(d, t_now)
    if m.sum() > 2:
        R, T, Z = d["R"][m], d["T"][m], d["Z"][m]
        alpha = np.linspace(0.05, 1, m.sum())
        cols = np.zeros((m.sum(), 4))
        cols[:, :3] = matplotlib.colors.to_rgb(COLORS[d["sta"]])
        cols[:, 3] = alpha
        ax.scatter(R, T, Z, c=cols, s=4, depthshade=False)
        ax.plot(R[-40:], T[-40:], Z[-40:], color=COLORS[d["sta"]], lw=1.2)
        ax.quiver(0, 0, 0, R[-1], T[-1], Z[-1], color="w", lw=1.6, arrow_length_ratio=0.15)
        # proiezione dell'ellisse R-Z (piano Rayleigh) sulla parete T = +L
        ax.plot(R[-40:], np.full(len(R[-40:]), L), Z[-40:], color="tab:blue", lw=1.1, alpha=0.8)
        in_ray = d["dist"] / 4.0 <= t_now <= d["dist"] / 3.0
        rtxt, rcol = rotation_text(rotation(d, t_now), in_ray)
        ax.text2D(0.0, 0.76, rtxt, transform=ax.transAxes, color=rcol, fontsize=8)
        # ripartizione energia nell'ultima finestra (60 s)
        w = slice(-60, None)
        e = np.array([np.sum(R[w] ** 2), np.sum(T[w] ** 2), np.sum(Z[w] ** 2)])
        e = e / e.sum() if e.sum() > 0 else e
        ax.text2D(0.0, 0.02, f"energia  R {e[0]:.0%}  T {e[1]:.0%}  Z {e[2]:.0%}",
                  transform=ax.transAxes, color=TXT, fontsize=7)
    ax.text2D(0.0, 0.93, f"IU.{d['sta']}  {d['dist']:.0f} km", transform=ax.transAxes,
              color=COLORS[d["sta"]], fontsize=9, weight="bold")
    ax.text2D(0.0, 0.84, lab, transform=ax.transAxes, color=col, fontsize=8, weight="bold")
    ax.view_init(elev=18, azim=-55)


def update(i):
    t_now = T_START + i * DT_FRAME
    title.set_text(f"M7.6 Alaska 2020-10-19 — propagazione onde sismiche   "
                   f"t = {t_now:4.0f} s  ({(ORIGIN + t_now).strftime('%H:%M:%S')} UTC)")
    for k, (vel, _) in FRONTS.items():
        if vel is None:
            delta = float(CURVES[k](t_now))
        else:
            delta = min(vel * t_now / R_EARTH, np.pi * 0.999)
        if not np.isfinite(delta) or delta <= 0:
            front_lines[k].set_data_3d([], [], [])
            continue
        x, y, z = facing(*small_circle(EV_LAT, EV_LON, delta))
        front_lines[k].set_data_3d(x, y, z)
        if k == "Rayleigh":
            # frecce sul fronte Rayleigh: verso di propagazione (verso l'esterno)
            for q in ray_arrows:
                q.remove()
            ray_arrows.clear()
            az = np.radians(np.arange(0, 360, 30))
            p1 = np.array(small_circle(EV_LAT, EV_LON, delta, az=az))
            p2 = np.array(small_circle(EV_LAT, EV_LON, min(delta + 0.13, np.pi), az=az))
            vis = ~np.isnan(facing(*p1)[0])
            if vis.any():
                ray_arrows.append(ax_g.quiver(*p1[:, vis], *(p2 - p1)[:, vis],
                                              color="deepskyblue", lw=2,
                                              arrow_length_ratio=0.45))
    for d in data:
        idx = np.searchsorted(d["t"], t_now)
        e = d["env"][min(idx, len(d["env"]) - 1)] / ENV_MAX
        st_markers[d["sta"]].set_sizes([60 + 900 * e])
    for a, d in zip(ax_s, data):
        draw_station(a, d, t_now)
    draw_rayleigh(ax_r, t_now)
    cursor.set_xdata([t_now, t_now])
    return []


frames = int((T_END - T_START) / DT_FRAME) + 1
print(f"Genero {frames} fotogrammi...")
anim = FuncAnimation(fig, update, frames=frames, blit=False)
out = os.path.join(OUT, "05_animazione_onde_3D.gif")
anim.save(out, writer=PillowWriter(fps=12), dpi=90,
          progress_callback=lambda k, n: print(f"  {k}/{n}", end="\r") if k % 20 == 0 else None)
print(f"\nSalvato: {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")

# fotogramma statico di anteprima (arrivo Rayleigh a TUC)
update(int((1250 - T_START) / DT_FRAME))
fig.savefig(os.path.join(OUT, "05_animazione_anteprima.png"), dpi=100, facecolor=fig.get_facecolor())
