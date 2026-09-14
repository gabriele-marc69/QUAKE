#!/usr/bin/env python3

"""
Scarica dati sismici per il terremoto M7.6 dell'Alaska
19 ottobre 2020, circa 20:54 UTC.

Stazioni:
    IU.ANMO
    IU.TUC
    IU.HRV

Canali:
    BH1
    BH2
    BHZ

Intervallo:
    20:54:00 UTC -> 22:20:00 UTC

Fonte:
    EarthScope FDSN Web Services

Output:
    seismic_data/
        IU_ANMO/
            IU_ANMO_<LOC>_BH1.mseed
            IU_ANMO_<LOC>_BH2.mseed
            IU_ANMO_<LOC>_BHZ.mseed
            StationXML.xml
        IU_TUC/
            ...
        IU_HRV/
            ...
"""

import os
import sys
import requests

from obspy import UTCDateTime
from obspy import read
from obspy.clients.fdsn import Client


# ============================================================
# CONFIGURAZIONE
# ============================================================

START = UTCDateTime("2020-10-19T20:54:00")
END   = UTCDateTime("2020-10-19T22:20:00")

STATIONS = {
    "IU": ["ANMO", "TUC", "HRV"]
}

CHANNELS = ["BH1", "BH2", "BHZ"]

# Cartella di output accanto allo script (indipendente dalla directory corrente)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "seismic_data"
)

# EarthScope FDSN
EARTHSCOPE_URL = "https://service.earthscope.org"

# ============================================================
# FUNZIONI
# ============================================================


def make_output_dir(path):
    """Crea la directory se non esiste."""
    os.makedirs(path, exist_ok=True)


def get_client():
    """
    Crea il client ObsPy per EarthScope.
    """
    try:
        # ObsPy >= 1.5
        return Client("EARTHSCOPE")

    except Exception:
        # Compatibilità con versioni precedenti
        return Client(EARTHSCOPE_URL)


def find_channels(client, network, station):
    """
    Cerca i canali BH1/BH2/BHZ realmente disponibili
    nell'intervallo temporale richiesto.

    Restituisce una lista di tuple:
        (network, station, location, channel)
    """

    print()
    print("=" * 70)
    print(f"Cerco i canali disponibili per {network}.{station}")
    print("=" * 70)

    try:
        inventory = client.get_stations(
            network=network,
            station=station,
            location="*",
            channel="BH?",
            starttime=START,
            endtime=END,
            level="channel"
        )

    except Exception as e:
        print(f"ERRORE nella ricerca dei metadati: {e}")
        return []

    found = []

    for net in inventory:
        for sta in net:
            for cha in sta:

                if cha.code not in CHANNELS:
                    continue

                item = (
                    net.code,
                    sta.code,
                    cha.location_code,
                    cha.code
                )

                if item not in found:
                    found.append(item)

    if not found:
        print("Nessun canale BH1/BH2/BHZ trovato.")
        return []

    print("Canali trovati:")

    for net, sta, loc, cha in found:

        if loc == "":
            loc_display = "--"
        else:
            loc_display = loc

        print(
            f"  {net}.{sta}.{loc_display}.{cha}"
        )

    return found


def download_stationxml(client, network, station, location):
    """
    Scarica StationXML al livello response, quindi includendo
    la risposta strumentale.
    """

    station_dir = os.path.join(
        OUTPUT_DIR,
        f"{network}_{station}"
    )

    make_output_dir(station_dir)

    if location == "":
        location_for_filename = "BLANK"
    else:
        location_for_filename = location

    filename = os.path.join(
        station_dir,
        f"{network}_{station}_{location_for_filename}_StationXML.xml"
    )

    print()
    print(f"Scarico StationXML: {filename}")

    try:

        inventory = client.get_stations(
            network=network,
            station=station,
            location=location if location else "--",
            channel="BH?",
            starttime=START,
            endtime=END,
            level="response"
        )

        inventory.write(
            filename,
            format="STATIONXML"
        )

        print("  OK")

        return filename

    except Exception as e:

        print(f"  ERRORE StationXML: {e}")

        return None


def download_waveform(
    network,
    station,
    location,
    channel
):
    """
    Scarica il waveform in formato miniSEED
    usando direttamente il servizio EarthScope.
    """

    station_dir = os.path.join(
        OUTPUT_DIR,
        f"{network}_{station}"
    )

    make_output_dir(station_dir)

    if location == "":
        location_for_filename = "BLANK"
        location_query = "--"
    else:
        location_for_filename = location
        location_query = location

    filename = os.path.join(
        station_dir,
        f"{network}_{station}_{location_for_filename}_{channel}.mseed"
    )

    print()
    print(
        f"Scarico {network}.{station}."
        f"{location_for_filename}.{channel}"
    )

    url = (
        f"{EARTHSCOPE_URL}/fdsnws/dataselect/1/query"
    )

    params = {
        "net": network,
        "sta": station,
        "loc": location_query,
        "cha": channel,
        "start": START.strftime("%Y-%m-%dT%H:%M:%S"),
        "end": END.strftime("%Y-%m-%dT%H:%M:%S"),
        "format": "miniseed",
        "nodata": "404"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=120
        )

        response.raise_for_status()

        if len(response.content) == 0:
            print("  Nessun dato ricevuto.")
            return None

        with open(filename, "wb") as f:
            f.write(response.content)

        print(
            f"  OK - {len(response.content):,} bytes"
        )

        # Controllo rapido del miniSEED
        try:

            st = read(filename)

            print(
                f"  Trace ricevute: {len(st)}"
            )

            for tr in st:

                print(
                    f"    {tr.id} | "
                    f"{tr.stats.starttime} -> "
                    f"{tr.stats.endtime} | "
                    f"{tr.stats.sampling_rate} Hz"
                )

        except Exception as e:

            print(
                f"  ATTENZIONE: file scaricato ma "
                f"non riesco a leggerlo con ObsPy: {e}"
            )

        return filename

    except requests.exceptions.HTTPError as e:

        print(
            f"  HTTP ERROR: {e}"
        )

        return None

    except Exception as e:

        print(
            f"  ERRORE download: {e}"
        )

        return None


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

    make_output_dir(OUTPUT_DIR)

    # --------------------------------------------------------
    # Client EarthScope
    # --------------------------------------------------------

    try:

        client = get_client()

        print()
        print("Connessione a EarthScope: OK")

    except Exception as e:

        print()
        print("Impossibile creare il client EarthScope.")
        print(e)
        sys.exit(1)

    # --------------------------------------------------------
    # Ciclo sulle stazioni
    # --------------------------------------------------------

    all_results = []

    for network, stations in STATIONS.items():

        for station in stations:

            channels_found = find_channels(
                client,
                network,
                station
            )

            if not channels_found:
                continue

            # ------------------------------------------------
            # Troviamo le location disponibili
            # ------------------------------------------------

            locations = sorted(
                set(
                    item[2]
                    for item in channels_found
                )
            )

            # ------------------------------------------------
            # Scarica StationXML per ogni location
            # ------------------------------------------------

            for location in locations:

                download_stationxml(
                    client,
                    network,
                    station,
                    location
                )

            # ------------------------------------------------
            # Scarica i tre canali
            # ------------------------------------------------

            for (
                net,
                sta,
                loc,
                channel
            ) in channels_found:

                result = download_waveform(
                    net,
                    sta,
                    loc,
                    channel
                )

                if result:
                    all_results.append(result)

    # --------------------------------------------------------
    # RIEPILOGO
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(" DOWNLOAD TERMINATO")
    print("=" * 70)

    print()
    print(
        f"File scaricati: {len(all_results)}"
    )

    print()
    print(
        f"I dati si trovano nella directory:\n"
        f"  {os.path.abspath(OUTPUT_DIR)}"
    )

    print()

    if all_results:

        print("File miniSEED:")

        for filename in all_results:
            print(
                f"  {filename}"
            )

    print()


if __name__ == "__main__":
    main()
