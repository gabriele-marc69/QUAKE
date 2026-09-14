# QUAKE — Analisi della polarizzazione delle onde di superficie

Terremoto **M7.6 Alaska (Sand Point)** — 19 ottobre 2020, 20:54:39 UTC
(evento USGS `us6000c9hg`, 54.662°N 159.675°W, profondità 28.4 km).

Pipeline in Python/ObsPy che scarica i dati sismici dalla rete globale IU,
analizza la polarizzazione delle onde di superficie (Love e Rayleigh) e
produce figure e un'animazione 3D della propagazione.

## Contenuto

| File | Descrizione |
|---|---|
| `Download dati sismici M7.6 Alaska 19 ottobre 2020.py` | Scarica forme d'onda (BH1/BH2/BHZ) e StationXML per IU.ANMO, IU.TUC, IU.HRV dai servizi FDSN EarthScope |
| `analisi_polarizzazione.py` | Rimozione risposta strumentale, rotazione ZNE→ZRT, filtro 20–100 s, finestre Love/Rayleigh, misura di sfasamento Z–R, ellitticità H/V, senso di rotazione, energia T/(R+Z) |
| `animazione_onde.py` | Animazione 3D: globo con fronti d'onda P/S/Love/Rayleigh, moto del suolo per stazione, inviluppo di energia |
| `seismic_data/` | Dati grezzi scaricati (miniSEED + StationXML) |
| `risultati_polarizzazione/` | Figure PNG, GIF animata e `report.txt` con i risultati numerici |

## Stazioni

| Stazione | Distanza | Back-azimuth |
|---|---|---|
| IU.ANMO (Albuquerque, NM) | Δ = 41.4° — 4603 km | 315° |
| IU.TUC (Tucson, AZ) | Δ = 40.8° — 4540 km | 318° |
| IU.HRV (Harvard, MA) | Δ = 55.7° — 6198 km | 315° |

## Risultati principali

In tutte e tre le stazioni le onde di Rayleigh mostrano uno sfasamento
Z–R vicino a −90° (−92.5°, −93.2°, −82.9°) e un rapporto H/V ≈ 0.82–0.85:
**moto ellittico retrogrado**, come atteso per il modo fondamentale di Rayleigh.
Le onde di Love sono quasi puramente trasversali (E_T/E_tot = 0.93–0.97,
rettilinearità ≥ 0.97).

Dettagli completi in [`risultati_polarizzazione/report.txt`](risultati_polarizzazione/report.txt).

## Requisiti

```bash
pip install obspy numpy scipy matplotlib pillow
```

## Uso

```bash
python "Download dati sismici M7.6 Alaska 19 ottobre 2020.py"   # scarica i dati
python analisi_polarizzazione.py                                 # analisi + figure
python animazione_onde.py                                        # animazione GIF 3D
```

## Fonte dati

EarthScope FDSN Web Services — rete IU (Global Seismograph Network, IRIS/USGS).
