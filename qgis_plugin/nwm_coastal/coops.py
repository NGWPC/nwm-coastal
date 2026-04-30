"""NOAA CO-OPS station metadata: download, cross-session cache, and layer build.

The metadata is fetched from the public CO-OPS endpoint and cached as a JSON
file under the user's QGIS profile so subsequent sessions skip the network
round-trip. If a cached copy is fresh (younger than ``MAX_CACHE_AGE_DAYS``)
we return it immediately; otherwise we re-download and refresh the cache. If
the network call fails but a stale cache exists, we fall back to the stale
copy so the plugin still works offline.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Final

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsBlockingNetworkRequest,
    QgsFeature,
    QgsGeometry,
    QgsMarkerSymbol,
    QgsPointXY,
    QgsSimpleMarkerSymbolLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtNetwork import QNetworkRequest

COOPS_URL: Final[str] = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=waterlevels"
)
MAX_CACHE_AGE_DAYS: Final[int] = 30
DOWNLOAD_TIMEOUT_MS: Final[int] = 60_000


def cache_file() -> Path:
    """Path to the on-disk cache under the active QGIS profile."""
    cache_dir = Path(QgsApplication.qgisSettingsDirPath()) / "cache" / "nwm_coastal"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "coops_stations_metadata.json"


def _download(force_refresh: bool = True) -> bytes | None:
    """Fetch the CO-OPS stations JSON payload synchronously.

    Returns the raw response bytes on success, ``None`` on any failure
    (network error, timeout, HTTP error). Uses ``QgsBlockingNetworkRequest``
    so QGIS proxy/auth settings are honored.
    """
    request = QNetworkRequest(QUrl(COOPS_URL))
    request.setRawHeader(b"User-Agent", b"NWM Coastal QGIS Plugin")

    blocking = QgsBlockingNetworkRequest()
    err = blocking.get(request, forceRefresh=force_refresh)
    if err != QgsBlockingNetworkRequest.NoError:
        return None

    reply = blocking.reply()
    # ``error()`` returns QNetworkReply.NetworkError; NoError == 0 in both Qt5/Qt6.
    if int(reply.error()) != 0:
        return None

    return bytes(reply.content())


def _read_cache(cache_path: Path) -> dict | None:
    try:
        return json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_stations(*, refresh: bool = False) -> tuple[dict | None, str]:
    """Return ``(stations_json, source_label)``, downloading if needed.

    Parameters
    ----------
    refresh : bool, optional
        Force a network fetch even if the cache is fresh.

    Returns
    -------
    tuple[dict | None, str]
        ``stations_json`` is the parsed top-level JSON dict (with a
        ``"stations"`` key), or ``None`` if no data could be obtained.
        ``source_label`` is one of ``"network"``, ``"cache"``,
        ``"stale-cache"``, or ``"unavailable"`` for diagnostics.
    """
    path = cache_file()

    if not refresh and path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86_400.0
        if age_days < MAX_CACHE_AGE_DAYS:
            cached = _read_cache(path)
            if cached is not None:
                return cached, "cache"

    payload = _download(force_refresh=True)
    if payload is not None:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            # Cache write failures shouldn't block the layer load.
            with contextlib.suppress(OSError):
                path.write_text(json.dumps(parsed))
            return parsed, "network"

    # Network failed — fall back to whatever cache exists, even if stale.
    if path.exists():
        cached = _read_cache(path)
        if cached is not None:
            return cached, "stale-cache"

    return None, "unavailable"


def _coerce_lonlat(station: dict) -> tuple[float, float] | None:
    try:
        lng = float(station.get("lng"))
        lat = float(station.get("lat"))
    except (TypeError, ValueError):
        return None
    if not (-180.0 <= lng <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return lng, lat


def build_layer(stations_json: dict, layer_name: str = "coops") -> QgsVectorLayer:
    """Build a styled in-memory point layer from the CO-OPS stations JSON.

    Parameters
    ----------
    stations_json : dict
        The decoded JSON returned by :func:`load_stations`.
    layer_name : str, optional
        Name shown in the layer panel.

    Returns
    -------
    QgsVectorLayer
        Memory layer with one point feature per valid station.
    """
    uri = (
        "Point?crs=EPSG:4326"
        "&field=station_id:string"
        "&field=station_name:string"
        "&field=state:string"
        "&field=tidal:string"
        "&field=greatlakes:string"
        "&field=time_zone:string"
        "&field=time_zone_offset:string"
    )
    layer = QgsVectorLayer(uri, layer_name, "memory")

    feats: list[QgsFeature] = []
    for station in stations_json.get("stations", []):
        lonlat = _coerce_lonlat(station)
        if lonlat is None:
            continue
        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*lonlat)))
        feat.setAttribute("station_id", str(station.get("id", "")))
        feat.setAttribute("station_name", str(station.get("name", "")))
        feat.setAttribute("state", str(station.get("state", "")))
        feat.setAttribute("tidal", str(station.get("tidal", "")))
        feat.setAttribute("greatlakes", str(station.get("greatlakes", "")))
        feat.setAttribute("time_zone", str(station.get("timezone", "")))
        feat.setAttribute("time_zone_offset", str(station.get("timezonecorr", "")))
        feats.append(feat)

    layer.dataProvider().addFeatures(feats)

    symbol = QgsMarkerSymbol.createSimple({})
    symbol.deleteSymbolLayer(0)
    marker = QgsSimpleMarkerSymbolLayer(Qgis.MarkerShape.Star)
    marker.setColor(QColor("orange"))
    marker.setSize(7.0)
    symbol.appendSymbolLayer(marker)
    layer.renderer().setSymbol(symbol)

    return layer
