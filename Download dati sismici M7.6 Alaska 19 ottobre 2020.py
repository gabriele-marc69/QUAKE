#!/usr/bin/env python3

"""
Scarica dati sismici per il terremoto M7.6 dell'Alaska
19 ottobre 2020, circa 20:54 UTC.

Stazioni:
    IU.ANMO
    IU.TUC
    IU.HRV

Canali richiesti:
    BH1
    BH2
    BH3   (richiesto, ma non esiste per IU.ANMO/TUC/HRV: segnalato
           come non disponibile, le componenti orizzontali sono BH1/BH2)
    BHZ

Intervallo:
    20:54:00 UTC -> 22:20:00 UTC

Ogni miniSEED viene verificato: deve essere continuo (una sola traccia,
nessun gap/overlap) e coprire tutto l'intervallo richiesto.

Fonte:
    EarthScope FDSN Web Services
        station/1    -> ricerca canali + StationXML (con risposta)
        dataselect/1 -> forme d'onda miniSEED

Tutte le richieste sono fatte con URL espliciti, e ogni URL usato
viene registrato in seismic_data/download_urls.txt (con esito HTTP,
dimensione e file di destinazione), cosi' il download e' riproducibile
anche senza lo script (browser, wget, curl).

Output:
    seismic_data/
        download_urls.txt
        IU_ANMO/
            IU_ANMO_<LOC>_BH1.mseed
            IU_ANMO_<LOC>_BH2.mseed
            IU_ANMO_<LOC>_BHZ.mseed
            IU_ANMO_<LOC>_StationXML.xml
        IU_TUC/
            ...
        IU_HRV/
            ...
"""

import io
import os
import sys
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests

from obspy import UTCDateTime
from obspy import read
from obspy import read_inventory


# ============================================================
# CONFIGURAZIONE
# ============================================================

START = UTCDateTime("2020-10-19T20:54:00")
END   = UTCDateTime("2020-10-19T22:20:00")

STATIONS = {
    "IU": ["ANMO", "TUC", "HRV"]
}

# BH3 e' richiesto esplicitamente: se la stazione non lo ha,
# viene riportato tra i canali non disponibili
CHANNELS = ["BH1", "BH2", "BH3", "BHZ"]

# Cartella di output accanto allo script (indipendente dalla directory corrente)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "seismic_data"
)

# File con l'elenco degli URL usati per il download
URL_LOG_FILE = os.path.join(OUTPUT_DIR, "download_urls.txt")

# EarthScope FDSN
EARTHSCOPE_URL = "https://service.earthscope.org"
STATION_URL    = f"{EARTHSCOPE_URL}/fdsnws/station/1/query"
DATASELECT_URL = f"{EARTHSCOPE_URL}/fdsnws/dataselect/1/query"

TIMEOUT = 120

# Registro di tutte le richieste effettuate (scritto in URL_LOG_FILE)
URL_LOG = []

# Canali richiesti ma non presenti nei metadati: "NET.STA.LOC.CHA"
MISSING_CHANNELS = []


# ============================================================
# FUNZIONI
# ============================================================


def make_output_dir(path):
    """Crea la directory se non esiste."""
    os.makedirs(path, exist_ok=True)


def fdsn_time(t):
    """Formato data/ora accettato dai servizi FDSN."""
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def location_query(location):
    """Nei servizi FDSN la location vuota si indica con '--'."""
    return location if location else "--"


def location_filename(location):
    """Nome della location da usare nei nomi dei file."""
    return location if location else "BLANK"


def fdsn_get(base_url, params, description, output_file=None):
    """
    Esegue una GET verso un servizio FDSN e registra l'URL usato.

    L'URL completo viene costruito qui (':' lasciati leggibili nelle
    date), cosi' quello registrato e' identico a quello richiesto.

    Restituisce la response se HTTP 200 con contenuto, altrimenti None.
    """

    url = f"{base_url}?{urlencode(params, safe=':,*?')}"

    entry = {
        "description": description,
        "url": url,
        "status": None,
        "bytes": 0,
        "output_file": output_file,
        "note": "",
        "check": "",
    }
    URL_LOG.append(entry)

    print(f"  URL: {url}")

    try:
        response = requests.get(url, timeout=TIMEOUT)

    except requests.exceptions.RequestException as e:
        entry["note"] = f"errore di rete: {e}"
        print(f"  ERRORE di rete: {e}")
        return None

    entry["status"] = response.status_code
    entry["bytes"] = len(response.content)

    if response.url != url:
        entry["note"] = f"reindirizzato a {response.url}"

    if response.status_code == 204 or response.status_code == 404:
        entry["note"] = entry["note"] or "nessun dato disponibile"
        print(f"  Nessun dato (HTTP {response.status_code}).")
        return None

    if not response.ok:
        entry["note"] = entry["note"] or response.reason
        print(f"  HTTP ERROR {response.status_code}: {response.reason}")
        return None

    if not response.content:
        entry["note"] = entry["note"] or "risposta vuota"
        print("  Nessun dato ricevuto.")
        return None

    return response


def find_channels(network, station):
    """
    Cerca i canali BH1/BH2/BHZ realmente disponibili
    nell'intervallo temporale richiesto (station service,
    level=channel, formato testo).

    Restituisce una lista di tuple:
        (network, station, location, channel)
    """

    print()
    print("=" * 70)
    print(f"Cerco i canali disponibili per {network}.{station}")
    print("=" * 70)

    params = {
        "net": network,
        "sta": station,
        "loc": "*",
        "cha": "BH?",
        "starttime": fdsn_time(START),
        "endtime": fdsn_time(END),
        "level": "channel",
        "format": "text",
        "nodata": "404",
    }

    response = fdsn_get(
        STATION_URL,
        params,
        f"Ricerca canali {network}.{station} (station, level=channel)"
    )

    if response is None:
        return []

    found = []

    # Formato testo FDSN:
    # #Network|Station|Location|Channel|Latitude|...
    for line in response.text.splitlines():

        if not line.strip() or line.startswith("#"):
            continue

        fields = line.split("|")

        if len(fields) < 4:
            continue

        item = (
            fields[0].strip(),
            fields[1].strip(),
            fields[2].strip(),
            fields[3].strip()
        )

        if item[3] not in CHANNELS:
            continue

        if item not in found:
            found.append(item)

    if not found:
        print(f"Nessun canale {'/'.join(CHANNELS)} trovato.")
        return []

    print("Canali trovati:")

    for net, sta, loc, cha in found:
        loc_display = loc if loc else "--"
        print(f"  {net}.{sta}.{loc_display}.{cha}")

    # Canali richiesti che la stazione non ha (per ogni location)
    locations = sorted(set(item[2] for item in found))

    missing = [
        f"{network}.{station}.{location_query(loc)}.{cha}"
        for loc in locations
        for cha in CHANNELS
        if (network, station, loc, cha) not in found
    ]

    if missing:
        print("Canali richiesti NON disponibili:")

        for item in missing:
            print(f"  {item}")

        MISSING_CHANNELS.extend(missing)

    return found


def download_stationxml(network, station, location):
    """
    Scarica StationXML al livello response, quindi includendo
    la risposta strumentale. Il file salvato e' l'XML originale
    restituito dal servizio.
    """

    station_dir = os.path.join(OUTPUT_DIR, f"{network}_{station}")
    make_output_dir(station_dir)

    filename = os.path.join(
        station_dir,
        f"{network}_{station}_{location_filename(location)}_StationXML.xml"
    )

    print()
    print(f"Scarico StationXML: {filename}")

    params = {
        "net": network,
        "sta": station,
        "loc": location_query(location),
        "cha": "BH?",
        "starttime": fdsn_time(START),
        "endtime": fdsn_time(END),
        "level": "response",
        "format": "xml",
        "nodata": "404",
    }

    response = fdsn_get(
        STATION_URL,
        params,
        f"StationXML {network}.{station}.{location_query(location)} "
        f"(station, level=response)",
        output_file=filename
    )

    if response is None:
        return None

    with open(filename, "wb") as f:
        f.write(response.content)

    # Controllo rapido dello StationXML
    try:
        inv = read_inventory(io.BytesIO(response.content))
        n_cha = sum(len(sta) for net in inv for sta in net)
        print(f"  OK - {len(response.content):,} bytes, {n_cha} canali")

    except Exception as e:
        print(
            f"  ATTENZIONE: file scaricato ma "
            f"non riesco a leggerlo con ObsPy: {e}"
        )

    return filename


def check_continuity(st):
    """
    Verifica che lo stream sia un miniSEED continuo sull'intervallo
    richiesto: una sola traccia, nessun gap/overlap, inizio e fine
    entro un campione da START/END.

    Restituisce (continuo, descrizione).
    """

    if len(st) == 0:
        return False, "nessuna traccia"

    gaps = st.get_gaps()
    delta = max(tr.stats.delta for tr in st)
    t0 = min(tr.stats.starttime for tr in st)
    t1 = max(tr.stats.endtime for tr in st)

    problems = []

    if gaps:
        n_gap = sum(1 for g in gaps if g[6] > 0)
        n_ovl = len(gaps) - n_gap
        problems.append(f"{n_gap} gap, {n_ovl} overlap")

    if len(st) > 1:
        problems.append(f"{len(st)} tracce")

    if t0 > START + delta:
        problems.append(f"inizia alle {t0} (dopo {START})")

    if t1 < END - 2 * delta:
        problems.append(f"finisce alle {t1} (prima di {END})")

    if problems:
        return False, "NON continuo: " + "; ".join(problems)

    return True, f"continuo {t0} -> {t1}"


def download_waveform(network, station, location, channel):
    """
    Scarica il waveform in formato miniSEED
    usando il servizio dataselect di EarthScope
    e ne verifica la continuita'.
    """

    station_dir = os.path.join(OUTPUT_DIR, f"{network}_{station}")
    make_output_dir(station_dir)

    filename = os.path.join(
        station_dir,
        f"{network}_{station}_{location_filename(location)}_{channel}.mseed"
    )

    print()
    print(
        f"Scarico {network}.{station}."
        f"{location_filename(location)}.{channel}"
    )

    params = {
        "net": network,
        "sta": station,
        "loc": location_query(location),
        "cha": channel,
        "start": fdsn_time(START),
        "end": fdsn_time(END),
        "format": "miniseed",
        "nodata": "404",
    }

    response = fdsn_get(
        DATASELECT_URL,
        params,
        f"Waveform {network}.{station}.{location_query(location)}.{channel} "
        f"(dataselect, miniSEED)",
        output_file=filename
    )

    if response is None:
        return None

    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"  OK - {len(response.content):,} bytes")

    entry = URL_LOG[-1]

    # Controllo del miniSEED: lettura + continuita'
    try:
        st = read(filename)
        print(f"  Trace ricevute: {len(st)}")

        for tr in st:
            print(
                f"    {tr.id} | "
                f"{tr.stats.starttime} -> "
                f"{tr.stats.endtime} | "
                f"{tr.stats.sampling_rate} Hz"
            )

        continuous, description = check_continuity(st)
        entry["check"] = description
        entry["continuous"] = continuous

        if continuous:
            print(f"  Continuita': OK ({description})")
        else:
            print(f"  ATTENZIONE - {description}")

    except Exception as e:
        entry["check"] = f"illeggibile con ObsPy: {e}"
        entry["continuous"] = False
        print(
            f"  ATTENZIONE: file scaricato ma "
            f"non riesco a leggerlo con ObsPy: {e}"
        )

    return filename


def write_url_log():
    """
    Scrive in URL_LOG_FILE tutti gli URL usati per il download:
    prima un blocco dettagliato per ogni richiesta, poi l'elenco
    semplice degli URL (uno per riga, pronto per copia/incolla).
    """

    make_output_dir(OUTPUT_DIR)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ok = [e for e in URL_LOG if e["status"] == 200 and e["bytes"] > 0]

    lines = [
        "# URL usati per il download dei dati sismici",
        "# Terremoto M7.6 Alaska (Sand Point) - 19 ottobre 2020, 20:54:39 UTC",
        f"# Intervallo richiesto: {START} -> {END}",
        f"# Servizio: EarthScope FDSN Web Services ({EARTHSCOPE_URL})",
        f"# Generato il: {now}",
        f"# Canali richiesti: {', '.join(CHANNELS)}",
        f"# Richieste totali: {len(URL_LOG)} - riuscite: {len(ok)}",
        "",
    ]

    if MISSING_CHANNELS:
        lines.append(
            "# Canali richiesti NON disponibili "
            "(assenti nei metadati; download dataselect tentato, "
            "vedi esito HTTP sotto):"
        )
        lines += [f"#   {item}" for item in MISSING_CHANNELS]
        lines.append("")

    lines += [
        "# " + "=" * 68,
        "# DETTAGLIO RICHIESTE",
        "# " + "=" * 68,
        "",
    ]

    for i, e in enumerate(URL_LOG, start=1):

        status = e["status"] if e["status"] is not None else "nessuna risposta"

        lines.append(f"# [{i:02d}] {e['description']}")
        lines.append(f"#      HTTP {status} - {e['bytes']:,} bytes")

        if e["output_file"] and e["status"] == 200:
            rel = os.path.relpath(e["output_file"], OUTPUT_DIR)
            lines.append(f"#      salvato in: seismic_data/{rel.replace(os.sep, '/')}")

        if e["check"]:
            lines.append(f"#      miniSEED: {e['check']}")

        if e["note"]:
            lines.append(f"#      nota: {e['note']}")

        lines.append(e["url"])
        lines.append("")

    lines += [
        "# " + "=" * 68,
        "# ELENCO URL (solo richieste riuscite, uno per riga)",
        "# " + "=" * 68,
        "",
    ]
    lines += [e["url"] for e in ok]

    with open(URL_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return URL_LOG_FILE


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 70)
    print(" DOWNLOAD DATI SISMICI - ALASKA M7.6")
    print(" 19 ottobre 2020")
    print("=" * 70)

    print()
    print(
        f"Intervallo UTC:\n"
        f"  {START}\n"
        f"  {END}"
    )

    print()
    print(f"Servizio FDSN: {EARTHSCOPE_URL}")

    make_output_dir(OUTPUT_DIR)

    all_results = []

    try:

        # ----------------------------------------------------
        # Ciclo sulle stazioni
        # ----------------------------------------------------

        for network, stations in STATIONS.items():

            for station in stations:

                channels_found = find_channels(network, station)

                if not channels_found:
                    continue

                # Location disponibili
                locations = sorted(set(item[2] for item in channels_found))

                # StationXML per ogni location
                for location in locations:
                    download_stationxml(network, station, location)

                # Canali BH1/BH2/BHZ
                for net, sta, loc, channel in channels_found:

                    result = download_waveform(net, sta, loc, channel)

                    if result:
                        all_results.append(result)

                # Canali richiesti ma assenti nei metadati (es. BH3):
                # tentativo diretto su dataselect, cosi' URL ed esito
                # restano registrati in download_urls.txt
                prefix = f"{network}.{station}."

                for item in MISSING_CHANNELS:

                    if not item.startswith(prefix):
                        continue

                    net, sta, loc, channel = item.split(".")
                    loc = "" if loc == "--" else loc

                    result = download_waveform(net, sta, loc, channel)

                    if result:
                        all_results.append(result)

    finally:

        # Il file degli URL viene scritto anche se il download
        # si interrompe a meta' (errore o Ctrl+C)
        url_file = write_url_log()

    # --------------------------------------------------------
    # RIEPILOGO
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(" DOWNLOAD TERMINATO")
    print("=" * 70)

    waveforms = [e for e in URL_LOG if "continuous" in e]
    not_continuous = [e for e in waveforms if not e["continuous"]]

    print()
    print(f"File miniSEED scaricati: {len(all_results)}")
    print(
        f"  continui sull'intervallo richiesto: "
        f"{len(waveforms) - len(not_continuous)}/{len(waveforms)}"
    )

    for e in not_continuous:
        print(f"  ATTENZIONE {e['description']}: {e['check']}")

    if MISSING_CHANNELS:
        print()
        print("Canali richiesti NON disponibili:")

        for item in MISSING_CHANNELS:
            print(f"  {item}")

    print()
    print(
        f"I dati si trovano nella directory:\n"
        f"  {os.path.abspath(OUTPUT_DIR)}"
    )

    print()
    print(
        f"URL usati ({len(URL_LOG)} richieste) salvati in:\n"
        f"  {url_file}"
    )

    print()

    if all_results:

        print("File miniSEED:")

        for filename in all_results:
            print(f"  {filename}")

    print()

    if not all_results:
        sys.exit(1)


if __name__ == "__main__":
    main()
