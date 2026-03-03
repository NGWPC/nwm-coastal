"""NWM Coastal QGIS Plugin."""


def classFactory(iface):  # noqa: N802
    """Load the plugin class.

    Parameters
    ----------
    iface : QgsInterface
        QGIS interface instance.
    """
    from .plugin_main import NWMCoastalPlugin

    return NWMCoastalPlugin(iface)
