"""Main plugin module for NWM Coastal."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsField,
    QgsFillSymbol,
    QgsGeometry,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsMeshLayer,
    QgsMessageLog,
    QgsPalLayerSettings,
    QgsPointXY,
    QgsProject,
    QgsProviderRegistry,
    QgsRasterLayer,
    QgsRendererCategory,
    QgsSimpleFillSymbolLayer,
    QgsSimpleLineSymbolLayer,
    QgsSimpleMarkerSymbolLayer,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QSizePolicy, QWidget

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgis.gui import QgisInterface

PLUGIN_NAME = "NWM Coastal"


class NWMCoastalPlugin:
    """Main QGIS plugin class for NWM Coastal."""

    # ------------------------------------------------------------------
    # Button 1: Add Basemap
    # ------------------------------------------------------------------

    _BASEMAP_LAYER_NAMES = ("OpenStreetMap", "divides", "flowpaths", "gages", "nexus", "coops")

    # ------------------------------------------------------------------
    # Button 15: Clear Map
    # ------------------------------------------------------------------

    _MANAGED_LAYER_ATTRS = (
        "_mesh_layer",
        "_boundary_layer",
        "_reaches_layer",
        "_sketcher_layer",
        "_merged_layer",
        "_line_layer",
        "_obs_layer",
    )

    # ------------------------------------------------------------------
    # Button 7: Load SCHISM Mesh
    # ------------------------------------------------------------------

    _GR3_BUFFER = 1024 * 1024  # 1 MiB streaming buffer for large meshes

    def __init__(self, iface: QgisInterface) -> None:
        self.iface = iface
        self._toolbar = None
        self._actions: list[QAction] = []
        # SFINCS workflow layers
        self._sketcher_layer: QgsVectorLayer | None = None
        self._merged_layer: QgsVectorLayer | None = None
        # SCHISM workflow layers
        self._mesh_layer: QgsMeshLayer | None = None
        self._boundary_layer: QgsVectorLayer | None = None
        self._reaches_layer: QgsVectorLayer | None = None
        self._line_layer: QgsVectorLayer | None = None
        self._obs_layer: QgsVectorLayer | None = None
        # SCHISM mesh-related state
        self._gr3_path: Path | None = None
        self._temp_2dm_path: Path | None = None
        self._temp_dat_path: Path | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_action(self, icon_name: str, text: str, callback) -> None:
        action = QAction(
            QgsApplication.getThemeIcon(icon_name),
            text,
            self.iface.mainWindow(),
        )
        action.triggered.connect(callback)
        self._toolbar.addAction(action)
        self._actions.append(action)

    def _log(
        self,
        message: str,
        level: Qgis.MessageLevel = Qgis.MessageLevel.Info,
        push: bool = False,
    ) -> None:
        QgsMessageLog.logMessage(message, PLUGIN_NAME, level=level)
        if push:
            self.iface.messageBar().pushMessage(PLUGIN_NAME, message, level=level)

    @staticmethod
    def _is_layer_alive(layer) -> bool:
        """Return ``True`` iff *layer* is a live, registered project layer.

        Removing a layer from the QGIS Layers panel destroys the underlying
        C++ object; the Python proxy we cached in ``self._..._layer`` is
        still bound, but touching it raises ``RuntimeError: wrapped C/C++
        object has been deleted``. This guard returns ``False`` for
        ``None``, deleted-C++, and not-in-project cases in one place.
        """
        if layer is None:
            return False
        try:
            layer_id = layer.id()
        except RuntimeError:
            return False
        return QgsProject.instance().mapLayer(layer_id) is not None

    def _add_gpkg_layer(
        self,
        gpkg_path: Path,
        layer_name: str,
        project: QgsProject,
        style_fn,
        *,
        optional: bool = False,
        display_name: str | None = None,
    ) -> None:
        uri = f"{gpkg_path}|layername={layer_name}"
        layer = QgsVectorLayer(uri, display_name or layer_name, "ogr")
        if not layer.isValid():
            if optional:
                self._log(f"Optional layer '{layer_name}' not found, skipping.")
            else:
                self._log(
                    f"Failed to load '{layer_name}' from {gpkg_path}",
                    Qgis.MessageLevel.Warning,
                    push=True,
                )
            return
        style_fn(layer)
        project.addMapLayer(layer)

    # -- Layer labeling helper --

    @staticmethod
    def _enable_labels(layer: QgsVectorLayer, field_name: str) -> None:
        """Enable single labels for a layer with a text buffer."""
        text_format = QgsTextFormat()
        text_format.setSize(15)

        buffer_settings = QgsTextBufferSettings()
        buffer_settings.setEnabled(True)
        text_format.setBuffer(buffer_settings)

        label_settings = QgsPalLayerSettings()
        label_settings.fieldName = field_name
        label_settings.setFormat(text_format)

        layer.setLabeling(QgsVectorLayerSimpleLabeling(label_settings))
        layer.setLabelsEnabled(True)

    def _add_coops_layer(self, project: QgsProject) -> None:
        """Download (or load from cache) NOAA CO-OPS stations and add the layer."""
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication

        from . import coops

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            stations, source = coops.load_stations()
        finally:
            QApplication.restoreOverrideCursor()

        if stations is None:
            self._log(
                "Failed to fetch CO-OPS stations and no cached copy available.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        layer = coops.build_layer(stations)
        if not layer.isValid():
            self._log(
                "Failed to build CO-OPS stations layer.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        self._enable_labels(layer, "station_id")
        project.addMapLayer(layer)

        source_msg = {
            "network": "downloaded fresh from NOAA",
            "cache": "loaded from cache",
            "stale-cache": "loaded from stale cache (network unavailable)",
        }.get(source, source)
        self._log(
            f"CO-OPS stations: {layer.featureCount()} stations ({source_msg}).",
            Qgis.MessageLevel.Info,
        )

    # -- Layer styling helpers --

    @staticmethod
    def _style_divides(layer: QgsVectorLayer) -> None:
        symbol = QgsFillSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        fill = QgsSimpleFillSymbolLayer()
        fill.setColor(QColor(0, 0, 0, 0))
        fill.setStrokeColor(QColor(0, 0, 0))
        fill.setStrokeWidth(0.1)
        symbol.appendSymbolLayer(fill)
        layer.renderer().setSymbol(symbol)

    @staticmethod
    def _style_flowpaths(layer: QgsVectorLayer) -> None:
        symbol = QgsLineSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        line = QgsSimpleLineSymbolLayer()
        line.setColor(QColor("blue"))
        line.setWidth(1.0)
        symbol.appendSymbolLayer(line)
        layer.renderer().setSymbol(symbol)

    @staticmethod
    def _style_gages(layer: QgsVectorLayer) -> None:
        symbol = QgsMarkerSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        marker = QgsSimpleMarkerSymbolLayer()
        marker.setColor(QColor("red"))
        marker.setSize(3.0)
        symbol.appendSymbolLayer(marker)
        layer.renderer().setSymbol(symbol)
        NWMCoastalPlugin._enable_labels(layer, "site_no")

    @staticmethod
    def _style_nexus(layer: QgsVectorLayer) -> None:
        symbol = QgsMarkerSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        marker = QgsSimpleMarkerSymbolLayer()
        marker.setColor(QColor("green"))
        marker.setSize(2.0)
        symbol.appendSymbolLayer(marker)
        layer.renderer().setSymbol(symbol)

    def _remove_existing_basemap_layers(self, project: QgsProject) -> None:
        """Remove previously added basemap layers to avoid duplicates."""
        for name in self._BASEMAP_LAYER_NAMES:
            for layer in project.mapLayersByName(name):
                project.removeMapLayer(layer.id())

    def _on_add_basemap(self) -> None:
        from .gui.dlg_basemap import BasemapDialog

        dlg = BasemapDialog(self.iface.mainWindow())
        if dlg.exec() != BasemapDialog.Accepted:
            return

        gpkg_path = dlg.gpkg_path
        divides_layer = dlg.divides_layer
        flowpaths_layer = dlg.flowpaths_layer
        gages_layer = dlg.gages_layer
        nexus_layer = dlg.nexus_layer
        flowpaths_override_path = dlg.flowpaths_override_path
        flowpaths_override_layer = dlg.flowpaths_override_layer
        include_coops = dlg.include_coops
        project = QgsProject.instance()
        root = project.layerTreeRoot()

        self._remove_existing_basemap_layers(project)

        # 1. OpenStreetMap (bottom layer)
        osm_uri = "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0"
        osm = QgsRasterLayer(osm_uri, "OpenStreetMap", "wms")
        if osm.isValid():
            project.addMapLayer(osm, False)
            root.insertLayer(-1, osm)
        else:
            self._log("Failed to load OpenStreetMap.", Qgis.MessageLevel.Warning, push=True)

        # 2. divides (polygon: transparent fill, black edge 0.1)
        if divides_layer:
            self._add_gpkg_layer(
                gpkg_path, divides_layer, project, self._style_divides, display_name="divides"
            )

        # 3. flowpaths (linestring: blue, width 1) — override gpkg takes precedence
        if flowpaths_override_path is not None:
            self._add_gpkg_layer(
                flowpaths_override_path,
                flowpaths_override_layer,
                project,
                self._style_flowpaths,
                display_name="flowpaths",
            )
        elif flowpaths_layer:
            self._add_gpkg_layer(
                gpkg_path,
                flowpaths_layer,
                project,
                self._style_flowpaths,
                display_name="flowpaths",
            )

        # 4. gages (point: red, size 3) — optional
        if gages_layer:
            self._add_gpkg_layer(
                gpkg_path,
                gages_layer,
                project,
                self._style_gages,
                optional=True,
                display_name="gages",
            )

        # 5. nexus (point: green, size 2) — optional
        if nexus_layer:
            self._add_gpkg_layer(
                gpkg_path,
                nexus_layer,
                project,
                self._style_nexus,
                optional=True,
                display_name="nexus",
            )

        # 6. CO-OPS stations (point: orange star, size 7) — fetched + cached
        if include_coops:
            self._add_coops_layer(project)

        self.iface.mapCanvas().refresh()
        self._log("Basemap layers loaded.", Qgis.MessageLevel.Success, push=True)

    # ------------------------------------------------------------------
    # Button 2: Draw Polygon
    # ------------------------------------------------------------------

    def _on_draw_polygon(self) -> None:
        project = QgsProject.instance()

        # Remove previous merged layer if it exists
        if self._is_layer_alive(self._merged_layer):
            project.removeMapLayer(self._merged_layer.id())
        self._merged_layer = None

        crs = project.crs()
        crs_str = crs.authid() if crs.isValid() else "EPSG:4326"

        self._sketcher_layer = QgsVectorLayer(
            f"Polygon?crs={crs_str}", "sketcher_polygon", "memory"
        )
        if not self._sketcher_layer.isValid():
            self._log(
                "Failed to create scratch polygon layer.",
                Qgis.MessageLevel.Critical,
                push=True,
            )
            return

        # Style: semi-transparent fill so underlying layers are visible
        symbol = QgsFillSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        fill = QgsSimpleFillSymbolLayer()
        fill.setColor(QColor(255, 165, 0, 80))  # orange, ~30% opacity
        fill.setStrokeColor(QColor(200, 100, 0))
        fill.setStrokeWidth(0.5)
        symbol.appendSymbolLayer(fill)
        self._sketcher_layer.renderer().setSymbol(symbol)

        project.addMapLayer(self._sketcher_layer)
        self._sketcher_layer.startEditing()
        self.iface.setActiveLayer(self._sketcher_layer)
        self.iface.actionAddFeature().trigger()

        self._log("Draw a polygon on the map. Right-click to finish.", push=True)

    # ------------------------------------------------------------------
    # Button 3: Edit Polygon
    # ------------------------------------------------------------------

    def _on_edit_polygon(self) -> None:
        if self._sketcher_layer is None:
            self._log(
                "No polygon layer to edit. Use 'Draw Polygon' first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        if not self._is_layer_alive(self._sketcher_layer):
            self._log(
                "Sketcher layer was removed. Use 'Draw Polygon' to create a new one.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            self._sketcher_layer = None
            return

        if not self._sketcher_layer.isEditable():
            self._sketcher_layer.startEditing()

        self.iface.setActiveLayer(self._sketcher_layer)
        self.iface.actionVertexTool().trigger()

        self._log("Vertex editing activated. Click vertices to move them.", push=True)

    # ------------------------------------------------------------------
    # Button 4: Union with NHF Divides
    # ------------------------------------------------------------------

    def _collect_intersecting_divides(
        self,
        sketcher_geom: QgsGeometry,
        divides_layer: QgsVectorLayer,
    ) -> list[QgsGeometry]:
        """Return geometries of divides that intersect *sketcher_geom*."""
        request = QgsFeatureRequest().setFilterRect(sketcher_geom.boundingBox())
        return [
            feat.geometry()
            for feat in divides_layer.getFeatures(request)
            if sketcher_geom.intersects(feat.geometry())
        ]

    @staticmethod
    def _union_and_clean(geometries: list[QgsGeometry]) -> QgsGeometry:
        """Union *geometries*, collapse multiparts, and strip interior rings."""
        union_geom = QgsGeometry.unaryUnion(geometries)
        if union_geom.isMultipart():
            union_geom = QgsGeometry.unaryUnion(union_geom.asGeometryCollection())

        if union_geom.isMultipart():
            largest = max(union_geom.asGeometryCollection(), key=lambda g: g.area())
            polygon = largest.asPolygon()[0]
        else:
            polygon = union_geom.asPolygon()[0]
        return QgsGeometry.fromPolygonXY([polygon])

    def _create_merged_layer(self, crs_authid: str, geom: QgsGeometry) -> QgsVectorLayer:
        """Create a styled memory layer containing *geom* and register it."""
        from qgis.core import QgsFeature

        layer = QgsVectorLayer(f"Polygon?crs={crs_authid}", "merged_polygon", "memory")

        symbol = QgsFillSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        fill = QgsSimpleFillSymbolLayer()
        fill.setColor(QColor(50, 205, 50, 80))  # lime green, ~30% opacity
        fill.setStrokeColor(QColor(0, 128, 0))
        fill.setStrokeWidth(0.5)
        symbol.appendSymbolLayer(fill)
        layer.renderer().setSymbol(symbol)

        layer.startEditing()
        feat = QgsFeature()
        feat.setGeometry(geom)
        layer.addFeature(feat)
        layer.commitChanges()
        return layer

    @staticmethod
    def _is_polygon_source_layer(layer) -> bool:
        """Return ``True`` iff *layer* is a valid source for Union/Save.

        Excludes the basemap ``divides`` layer: it's the *target* of
        'Union with NHF Divides', not a valid source, and selecting a
        divide there would be confusing.
        """
        return (
            isinstance(layer, QgsVectorLayer)
            and layer.geometryType() == Qgis.GeometryType.Polygon
            and layer.selectedFeatureCount() > 0
            and layer.name() != "divides"
        )

    def _find_polygon_with_selection(self, project: QgsProject) -> QgsVectorLayer | None:
        """Return a polygon vector layer with at least one selected feature.

        Prefers the currently active layer when it qualifies — that is the
        most explicit signal of user intent. Otherwise falls back to the
        first polygon layer in the project that has a selection. The
        ``divides`` basemap layer is never returned (see
        :meth:`_is_polygon_source_layer`).
        """
        active = self.iface.activeLayer()
        if self._is_polygon_source_layer(active):
            return active

        for layer in project.mapLayers().values():
            if self._is_polygon_source_layer(layer):
                return layer
        return None

    def _resolve_union_source(
        self,
        project: QgsProject,
    ) -> tuple[QgsVectorLayer, list[QgsGeometry], str] | None:
        """Pick the source polygon for 'Union with NHF Divides'.

        Prefers selected features on a polygon layer (active layer first,
        then any polygon layer with a selection). Falls back to all
        features on the sketcher layer. Logs and returns ``None`` when
        no usable source is available.
        """
        selected_layer = self._find_polygon_with_selection(project)

        if selected_layer is not None:
            if selected_layer.isEditable() and not selected_layer.commitChanges():
                self._log(
                    f"Failed to commit edits on '{selected_layer.name()}'. "
                    "Fix geometry errors and try again.",
                    Qgis.MessageLevel.Critical,
                    push=True,
                )
                selected_layer.rollBack()
                return None
            geoms = [
                f.geometry()
                for f in selected_layer.selectedFeatures()
                if f.geometry() and not f.geometry().isEmpty()
            ]
            n = selected_layer.selectedFeatureCount()
            label = (
                f"selected feature on '{selected_layer.name()}'"
                if n == 1
                else f"selected feature(s) on '{selected_layer.name()}'"
            )
            return selected_layer, geoms, label

        if not self._is_layer_alive(self._sketcher_layer):
            self._log(
                "No polygon selected and no sketched polygon. "
                "Use 'Draw Polygon' or select a polygon on any layer first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            self._sketcher_layer = None
            return None

        if self._sketcher_layer.isEditable() and not self._sketcher_layer.commitChanges():
            self._log(
                "Failed to commit sketcher edits. Fix geometry errors and try again.",
                Qgis.MessageLevel.Critical,
                push=True,
            )
            self._sketcher_layer.rollBack()
            return None

        geoms = [
            f.geometry()
            for f in self._sketcher_layer.getFeatures()
            if f.geometry() and not f.geometry().isEmpty()
        ]
        return self._sketcher_layer, geoms, "sketched polygon"

    def _on_union_divides(self) -> None:
        """Union a source polygon with all intersecting NHF divides.

        The source polygon is, in order of preference:

        1. The selected feature(s) on the currently active polygon layer.
        2. The selected feature(s) on any other polygon layer.
        3. All features on the sketcher layer (the original behavior).

        This lets the user union the polygon produced by 'Extract Mesh
        Boundary' (or any other polygon layer) with NHF divides without
        having to redraw it on the sketcher. The source layer itself is
        kept untouched; the result lands in a separate merged layer so
        the source can be edited and the union re-run at any time.
        """
        project = QgsProject.instance()

        resolved = self._resolve_union_source(project)
        if resolved is None:
            return
        source_layer, source_geoms, source_label = resolved

        if not source_geoms:
            self._log(
                f"No usable geometry from {source_label}.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        divides_layers = project.mapLayersByName("divides")
        if not divides_layers:
            self._log(
                "No 'divides' layer found. Load basemap first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        source_crs = source_layer.crs()
        divides_crs = divides_layers[0].crs()

        if source_crs != divides_crs:
            transform = QgsCoordinateTransform(source_crs, divides_crs, project)
            source_geoms_div = []
            for g in source_geoms:
                geom_div = QgsGeometry(g)
                geom_div.transform(transform)
                source_geoms_div.append(geom_div)
        else:
            source_geoms_div = source_geoms

        source_geom_div = self._union_and_clean(source_geoms_div)

        intersecting = self._collect_intersecting_divides(source_geom_div, divides_layers[0])
        if not intersecting:
            self._log(
                f"No divides intersect the {source_label}.",
                Qgis.MessageLevel.Info,
                push=True,
            )
            return

        union_geom = self._union_and_clean([source_geom_div, *intersecting])

        if source_crs != divides_crs:
            union_geom.transform(QgsCoordinateTransform(divides_crs, source_crs, project))

        if self._is_layer_alive(self._merged_layer):
            project.removeMapLayer(self._merged_layer.id())

        self._merged_layer = self._create_merged_layer(source_crs.authid(), union_geom)
        project.addMapLayer(self._merged_layer)
        self.iface.mapCanvas().refresh()
        self._log(
            f"Union complete: merged {source_label} with {len(intersecting)} divides.",
            Qgis.MessageLevel.Success,
            push=True,
        )

    # ------------------------------------------------------------------
    # Button 5: Save Polygon
    # ------------------------------------------------------------------

    def _on_save_polygon(self) -> None:
        """Save a polygon to GeoJSON.

        Source priority (matches 'Union with NHF Divides'):

        1. The selected feature(s) on a polygon layer — active layer
           first, then any other polygon layer with a selection. Only
           the selected features are written.
        2. The merged layer from a previous 'Union with NHF Divides'.
        3. The sketcher layer from 'Draw Polygon'.
        """
        project = QgsProject.instance()

        only_selected = False
        save_layer: QgsVectorLayer | None = None
        save_label = "polygon"

        selected_layer = self._find_polygon_with_selection(project)
        if selected_layer is not None:
            if selected_layer.isEditable():
                selected_layer.commitChanges()
            save_layer = selected_layer
            only_selected = True
            n = selected_layer.selectedFeatureCount()
            save_label = (
                f"selected feature on '{selected_layer.name()}'"
                if n == 1
                else f"selected feature(s) on '{selected_layer.name()}'"
            )
        elif self._is_layer_alive(self._merged_layer) and self._merged_layer.featureCount() > 0:
            save_layer = self._merged_layer
            save_label = "merged polygon"
        elif self._is_layer_alive(self._sketcher_layer):
            if self._sketcher_layer.isEditable():
                self._sketcher_layer.commitChanges()
            if self._sketcher_layer.featureCount() > 0:
                save_layer = self._sketcher_layer
                save_label = "sketched polygon"

        if save_layer is None:
            self._log(
                "No polygon to save. Select a polygon or use 'Draw Polygon' first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self.iface.mainWindow(),
            "Save Polygon as GeoJSON",
            "",
            "GeoJSON (*.geojson)",
        )
        if not save_path:
            return
        if not save_path.endswith(".geojson"):
            save_path += ".geojson"

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GeoJSON"
        options.fileEncoding = "UTF-8"
        if only_selected:
            options.onlySelectedFeatures = True

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        error_code, error_msg, *_ = QgsVectorFileWriter.writeAsVectorFormatV3(
            save_layer,
            save_path,
            project.transformContext(),
            options,
        )

        if error_code == QgsVectorFileWriter.NoError:
            self._log(f"Saved {save_label} to {save_path}", Qgis.MessageLevel.Success, push=True)
        else:
            self._log(f"Error saving polygon: {error_msg}", Qgis.MessageLevel.Critical, push=True)

    # ------------------------------------------------------------------
    # Button 6: Export Selected Flowpaths
    # ------------------------------------------------------------------

    def _on_export_flowpaths(self) -> None:
        """Export selected flowpath features to a GeoJSON file.

        If no features are selected on the flowpaths layer, the layer is
        made active and QGIS's selection tool is activated so the user can
        pick features first.
        """
        project = QgsProject.instance()

        fp_layers = project.mapLayersByName("flowpaths")
        if not fp_layers:
            self._log(
                "No 'flowpaths' layer found. Load basemap first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        fp_layer = fp_layers[0]

        if fp_layer.selectedFeatureCount() == 0:
            self.iface.setActiveLayer(fp_layer)
            self.iface.actionSelect().trigger()
            self._log(
                "Select flowpath features on the map, then click this button again to export.",
                push=True,
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self.iface.mainWindow(),
            "Export Selected Flowpaths as GeoJSON",
            "",
            "GeoJSON (*.geojson)",
        )
        if not save_path:
            return
        if not save_path.endswith(".geojson"):
            save_path += ".geojson"

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GeoJSON"
        options.fileEncoding = "UTF-8"
        options.onlySelectedFeatures = True

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        error_code, error_msg, *_ = QgsVectorFileWriter.writeAsVectorFormatV3(
            fp_layer,
            save_path,
            project.transformContext(),
            options,
        )

        if error_code == QgsVectorFileWriter.NoError:
            n = fp_layer.selectedFeatureCount()
            self._log(
                f"Exported {n} flowpath(s) to {save_path}",
                Qgis.MessageLevel.Success,
                push=True,
            )
        else:
            self._log(
                f"Error exporting flowpaths: {error_msg}",
                Qgis.MessageLevel.Critical,
                push=True,
            )

    def _mesh_load_failure_message(self, out_2dm: Path) -> str:
        """Build a diagnostic-rich message for a failed mesh load.

        ``QgsMeshLayer.isValid() == False`` alone doesn't tell us
        whether the file is missing/empty, the format is wrong, or the
        MDAL provider isn't registered in this QGIS install. On HPC
        clusters all three are plausible failure modes, so collect the
        relevant signals into one message the user can share.
        """
        parts = [f"Failed to load mesh layer from {out_2dm}"]
        if not out_2dm.exists():
            parts.append("file does not exist after conversion")
        else:
            try:
                size = out_2dm.stat().st_size
                parts.append(f"file size: {size:,} bytes")
            except OSError as exc:  # pragma: no cover — diagnostic only
                parts.append(f"stat failed: {exc}")
        try:
            err = self._mesh_layer.error() if self._mesh_layer is not None else None
            if err is not None and not err.isEmpty():
                parts.append(f"MDAL error: {err.summary()}")
        except (RuntimeError, AttributeError):  # pragma: no cover — diagnostic only
            pass
        providers = QgsProviderRegistry.instance().providerList()
        if "mdal" not in providers:
            parts.append(
                "the 'mdal' provider is not registered in this QGIS install "
                f"(available: {', '.join(sorted(providers)) or 'none'})"
            )
        return " — ".join(parts)

    @staticmethod
    def _mesh_cache_dir() -> Path:
        """Return the on-disk mesh cache dir under the active QGIS profile.

        Uses the QGIS profile rather than the system temp dir because on
        HPC clusters ``/tmp`` is often per-job, tmpfs-backed, or cleaned
        aggressively — once QGIS hands the .2dm path off to MDAL, the
        backing file can disappear before the layer is read, leading to
        ``Failed to load mesh layer`` errors. The profile dir is
        guaranteed to be user-writable and persists across sessions, so
        the .2dm/.dat sidecar stay valid for the lifetime of the layer.
        Files are tracked via ``self._temp_*_path`` and removed when the
        mesh is reloaded, the map is cleared, or the plugin is unloaded.
        """
        cache_dir = Path(QgsApplication.qgisSettingsDirPath()) / "cache" / "nwm_coastal" / "meshes"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @staticmethod
    def _convert_gr3_to_2dm(
        gr3_path: Path,
        out_2dm: Path,
        out_dat: Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[int, int]:
        """Convert SCHISM ``hgrid.gr3`` → MDAL 2DM + Bed-Elevation ``.dat`` sidecar.

        Streaming, line-by-line: never holds a full vertex/element/depth
        array in memory, so this stays viable for million-node meshes. Both
        output files are written in lockstep with the gr3 input. The
        ``.dat`` is written in MDAL's XMS/SMS ASCII Dataset format so
        :py:meth:`QgsMeshLayer.addDatasets` can attach it as a vertex
        scalar named ``Bed Elevation``.

        Parameters
        ----------
        progress : Callable[[int, int], None], optional
            Called periodically (≈100 times) with ``(processed, total)``
            counts. Use it to drive a progress bar.

        Returns
        -------
        tuple[int, int]
            ``(n_nodes, n_elements)``.
        """
        buf = NWMCoastalPlugin._GR3_BUFFER
        # fsync both outputs before the ``with`` closes them — on
        # NFS-mounted home directories (common on HPC clusters under
        # ``~/.local/share``) the kernel can otherwise hold writes in
        # the page cache and let MDAL see a truncated or empty file.
        with (
            gr3_path.open("r", buffering=buf) as fin,
            out_2dm.open("w", buffering=buf) as f2dm,
            out_dat.open("w", buffering=buf) as fdat,
        ):
            fin.readline()  # description / header line
            counts = fin.readline().split()
            n_elements, n_nodes = int(counts[0]), int(counts[1])

            f2dm.write("MESH2D\n")

            fdat.write("DATASET\n")
            fdat.write('OBJTYPE "mesh2d"\n')
            fdat.write("BEGSCL\n")
            fdat.write(f"ND {n_nodes}\n")
            fdat.write(f"NC {n_elements}\n")
            fdat.write('NAME "Bed Elevation"\n')
            fdat.write("TS 0 0\n")

            total = n_nodes + n_elements
            step = max(1, total // 100)
            processed = 0

            for _ in range(n_nodes):
                parts = fin.readline().split()
                f2dm.write(f"ND {parts[0]} {parts[1]} {parts[2]} {parts[3]}\n")
                fdat.write(f"{parts[3]}\n")
                processed += 1
                if progress is not None and processed % step == 0:
                    progress(processed, total)

            fdat.write("ENDDS\n")

            for _ in range(n_elements):
                parts = fin.readline().split()
                elem_id = parts[0]
                ncorners = int(parts[1])
                nodes = parts[2 : 2 + ncorners]
                if ncorners == 3:
                    f2dm.write(f"E3T {elem_id} {nodes[0]} {nodes[1]} {nodes[2]}\n")
                elif ncorners == 4:
                    f2dm.write(f"E4Q {elem_id} {nodes[0]} {nodes[1]} {nodes[2]} {nodes[3]}\n")
                else:
                    raise ValueError(
                        f"Unsupported element with {ncorners} corners (only 3 or 4 supported)"
                    )
                processed += 1
                if progress is not None and processed % step == 0:
                    progress(processed, total)

            f2dm.flush()
            os.fsync(f2dm.fileno())
            fdat.flush()
            os.fsync(fdat.fileno())

        if progress is not None:
            progress(total, total)
        return n_nodes, n_elements

    @staticmethod
    def _is_geographic_gr3(gr3_path: Path) -> bool:
        """Return ``True`` if the first mesh node is in lon/lat range."""
        with gr3_path.open("r") as f:
            f.readline()
            f.readline()
            first = f.readline().split()
        x, y = float(first[1]), float(first[2])
        return abs(x) <= 180.0 and abs(y) <= 90.0

    @staticmethod
    def _dataset_groups_hint(mesh_layer: QgsMeshLayer) -> str:
        """Return a hint about available scalar dataset groups (e.g. Bed Elevation)."""
        provider = mesh_layer.dataProvider()
        if provider is None:
            return ""
        n = provider.datasetGroupCount()
        if n <= 0:
            return ""
        names = []
        for i in range(n):
            meta = provider.datasetGroupMetadata(i)
            name = meta.name() if meta else ""
            if name:
                names.append(name)
        listed = ", ".join(names) if names else f"{n} dataset(s)"
        return (
            f" Scalar dataset(s) available [{listed}] — to render, open "
            "Layer Properties → Symbology → Datasets tab and pick a dataset."
        )

    def _push_progress_bar(
        self, title: str, body: str
    ) -> tuple[object, Callable[[int, int], None]]:
        """Show a message-bar entry with a 0-100 progress bar.

        Returns ``(msg_item, update_callback)``. Pop the item via
        ``self.iface.messageBar().popWidget(msg_item)`` when done.
        """
        from qgis.PyQt.QtCore import QCoreApplication, Qt
        from qgis.PyQt.QtWidgets import QProgressBar

        bar = QProgressBar()
        bar.setMaximum(100)
        bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        msg_item = self.iface.messageBar().createMessage(title, body)
        msg_item.layout().addWidget(bar)
        self.iface.messageBar().pushWidget(msg_item, Qgis.MessageLevel.Info)

        def update(done: int, total: int) -> None:
            if total > 0:
                bar.setValue(int((done / total) * 100))
            QCoreApplication.processEvents()

        return msg_item, update

    def _configure_mesh_renderer(self, mesh_layer: QgsMeshLayer) -> None:
        """Apply default native-mesh styling; leave scalar contours off."""
        renderer = mesh_layer.rendererSettings()
        native = renderer.nativeMeshSettings()
        native.setEnabled(True)
        native.setLineWidth(0.3)
        native.setColor(QColor(80, 80, 80))
        renderer.setNativeMeshSettings(native)
        # Bed Elevation contours stay OFF by default — they can be expensive
        # to render on large meshes. To turn them on:
        #   Layer Properties → Symbology → Datasets tab → pick "Bed Elevation"
        # which sets the active scalar; the Color tab + "Show Contours"
        # checkbox become usable from there.
        renderer.setActiveScalarDatasetGroup(-1)
        mesh_layer.setRendererSettings(renderer)

    def _on_load_mesh(self) -> None:
        gr3_path_str, _ = QFileDialog.getOpenFileName(
            self.iface.mainWindow(),
            "Select SCHISM hgrid.gr3 file",
            "",
            "GR3 Files (*.gr3);;All Files (*)",
        )
        if not gr3_path_str:
            return

        gr3_path = Path(gr3_path_str)
        if not gr3_path.is_file():
            self._log(f"File not found: {gr3_path}", Qgis.MessageLevel.Warning, push=True)
            return

        project = QgsProject.instance()

        # Remove previous mesh layer (releases the temp .2dm file handle).
        # User may have deleted it via the Layers panel — `_is_layer_alive`
        # handles that without raising ``RuntimeError`` on the dead proxy.
        if self._is_layer_alive(self._mesh_layer):
            project.removeMapLayer(self._mesh_layer.id())
        self._mesh_layer = None

        # Remove previous temp files now that the mesh layer is gone
        for attr in ("_temp_2dm_path", "_temp_dat_path"):
            old = getattr(self, attr, None)
            if old is not None and old.exists():
                with contextlib.suppress(OSError):
                    old.unlink()
            setattr(self, attr, None)

        out_2dm = self._mesh_cache_dir() / f"{gr3_path.stem}_{id(self)}.2dm"
        out_dat = out_2dm.with_suffix(".dat")

        msg_item, on_progress = self._push_progress_bar(
            "Loading SCHISM Mesh",
            f"Converting {gr3_path.name} (gr3 → 2DM + Bed Elevation)…",
        )
        try:
            n_nodes, n_elements = self._convert_gr3_to_2dm(
                gr3_path, out_2dm, out_dat, progress=on_progress
            )
        except (OSError, ValueError, IndexError) as exc:
            self.iface.messageBar().popWidget(msg_item)
            self._log(
                f"Failed to read SCHISM mesh: {exc}",
                Qgis.MessageLevel.Critical,
                push=True,
            )
            return

        from qgis.PyQt.QtCore import QTimer

        QTimer.singleShot(1500, lambda: self.iface.messageBar().popWidget(msg_item))

        self._gr3_path = gr3_path
        self._temp_2dm_path = out_2dm
        self._temp_dat_path = out_dat
        is_geographic = self._is_geographic_gr3(gr3_path)

        self._mesh_layer = QgsMeshLayer(str(out_2dm), gr3_path.stem, "mdal")
        if not self._mesh_layer.isValid():
            self._log(
                self._mesh_load_failure_message(out_2dm),
                Qgis.MessageLevel.Critical,
                push=True,
            )
            self._mesh_layer = None
            return

        if is_geographic:
            self._mesh_layer.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
        else:
            self._log(
                "Mesh appears to be in projected coordinates. "
                "Set the CRS manually in Layer Properties.",
                Qgis.MessageLevel.Warning,
                push=True,
            )

        # Attach Bed Elevation via the MDAL ASCII .dat sidecar that
        # ``_convert_gr3_to_2dm`` wrote alongside the 2DM. The sidecar is
        # attached regardless of success — failure is non-fatal because
        # the mesh itself loads from the .2dm. ``_dataset_groups_hint``
        # below tells the user whether scalar datasets are available.
        self._mesh_layer.addDatasets(str(out_dat))

        self._configure_mesh_renderer(self._mesh_layer)

        project.addMapLayer(self._mesh_layer)
        self.iface.mapCanvas().setExtent(self._mesh_layer.extent())
        self.iface.mapCanvas().refresh()

        crs_label = "EPSG:4326" if is_geographic else "(unset)"
        ds_hint = self._dataset_groups_hint(self._mesh_layer)
        self._log(
            f"Mesh loaded: {n_nodes:,} nodes, {n_elements:,} elements ({crs_label}).{ds_hint}",
            Qgis.MessageLevel.Success,
            push=True,
        )

    # ------------------------------------------------------------------
    # Button 8: Extract Mesh Boundary
    # ------------------------------------------------------------------

    @staticmethod
    def _read_gr3_boundaries(
        gr3_path: Path,
    ) -> tuple[
        dict[int, tuple[float, float]],
        list[list[int]],
        list[list[int]],
        list[list[int]],
    ]:
        """Parse node coordinates and boundary node sequences from an hgrid.gr3.

        Returns a 4-tuple ``(coords, open_bnds, ext_land_bnds, island_bnds)``
        where ``coords`` maps node id to ``(x, y)``, and each boundary is a
        list of node ids.
        """

        def stripped(s: str) -> str:
            return s.split("!", maxsplit=1)[0].strip()

        coords: dict[int, tuple[float, float]] = {}
        open_bnds: list[list[int]] = []
        ext_bnds: list[list[int]] = []
        isl_bnds: list[list[int]] = []

        with gr3_path.open("r") as f:
            f.readline()  # header
            n_elements, n_nodes = (int(x) for x in f.readline().split()[:2])
            for _ in range(n_nodes):
                parts = f.readline().split()
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            for _ in range(n_elements):
                f.readline()

            # Open boundaries
            try:
                n_open = int(stripped(f.readline()).split()[0])
            except (ValueError, IndexError):
                return coords, [], [], []
            f.readline()  # total open nodes
            for _ in range(n_open):
                n_b = int(stripped(f.readline()).split()[0])
                open_bnds.append([int(stripped(f.readline()).split()[0]) for _ in range(n_b)])

            # Land boundaries (mainland + islands)
            try:
                n_land = int(stripped(f.readline()).split()[0])
            except (ValueError, IndexError):
                return coords, open_bnds, [], []
            f.readline()  # total land nodes
            for _ in range(n_land):
                head = stripped(f.readline()).split()
                n_b = int(head[0])
                btype = int(head[1]) if len(head) > 1 else 0
                bnd = [int(stripped(f.readline()).split()[0]) for _ in range(n_b)]
                (isl_bnds if btype == 1 else ext_bnds).append(bnd)

        return coords, open_bnds, ext_bnds, isl_bnds

    @staticmethod
    def _chain_outer_ring(
        segments: list[list[int]],
    ) -> list[int]:
        """Chain boundary segments into one ring of node ids.

        Handles arbitrary segment order and orientation: each remaining
        segment is appended (or reversed) to extend the current ring at
        either end. Drops shared endpoints to avoid duplicate vertices.

        Subdomain meshes produced by ``extract_mesh`` write original
        boundary segments and new "cut" segments in arbitrary order and
        direction (each is a valid SCHISM boundary on its own, but the
        sequence is not the CCW perimeter chain). A naive concatenation
        would emit straight "jump" lines between non-adjacent segments —
        this routine avoids that by always matching shared endpoints.

        Kept in sync with ``_chain_ring`` in
        ``src/coastal_calibration/schism/stages.py``.
        """
        remaining = [list(s) for s in segments if s]
        if not remaining:
            return []

        ring = remaining.pop(0)
        while remaining:
            last = ring[-1]
            first = ring[0]
            for i, seg in enumerate(remaining):
                if seg[0] == last:
                    ring.extend(seg[1:])
                    remaining.pop(i)
                    break
                if seg[-1] == last:
                    ring.extend(reversed(seg[:-1]))
                    remaining.pop(i)
                    break
                if seg[-1] == first:
                    ring = list(seg) + ring[1:]
                    remaining.pop(i)
                    break
                if seg[0] == first:
                    ring = list(reversed(seg)) + ring[1:]
                    remaining.pop(i)
                    break
            else:
                # No greedy match — fall back to concatenating the next
                # segment in file order. SCHISM hgrid files normally
                # list boundaries in CCW order, but some real meshes have
                # an off-by-one at the open/land transition (e.g. open
                # ends at node N and land starts at N+1, where the two
                # nodes are spatially adjacent but not identical). The
                # pre-greedy implementation handled this by always
                # concatenating; preserve that behavior so the polygon
                # has a one-node visual gap rather than ending the chain
                # early and being closed via a long diagonal.
                ring.extend(remaining.pop(0))

        return ring

    def _on_extract_boundary(self) -> None:
        if self._gr3_path is None:
            self._log(
                "Load a SCHISM mesh first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        try:
            coords, open_bnds, ext_bnds, isl_bnds = self._read_gr3_boundaries(self._gr3_path)
        except (OSError, ValueError, IndexError) as exc:
            self._log(
                f"Failed to parse mesh boundaries: {exc}",
                Qgis.MessageLevel.Critical,
                push=True,
            )
            return

        outer_ids = self._chain_outer_ring(open_bnds + ext_bnds)
        if not outer_ids:
            self._log(
                "Mesh has no exterior boundaries to extract.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        outer = [QgsPointXY(*coords[nid]) for nid in outer_ids]
        if outer[0] != outer[-1]:
            outer.append(outer[0])

        rings: list[list[QgsPointXY]] = [outer]
        for isl in isl_bnds:
            ring = [QgsPointXY(*coords[nid]) for nid in isl]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            rings.append(ring)

        project = QgsProject.instance()
        # Use ``_is_layer_alive`` rather than a plain ``is not None`` check —
        # the user may have removed the mesh from the Layers panel, leaving
        # a dangling Python proxy whose ``.crs()`` would raise RuntimeError.
        if self._is_layer_alive(self._mesh_layer) and self._mesh_layer.crs().isValid():
            crs_authid = self._mesh_layer.crs().authid()
        else:
            crs_authid = "EPSG:4326"
        layer = QgsVectorLayer(f"Polygon?crs={crs_authid}", "mesh_boundary", "memory")

        symbol = QgsFillSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        fill = QgsSimpleFillSymbolLayer()
        fill.setColor(QColor(50, 150, 50, 60))
        fill.setStrokeColor(QColor(0, 100, 0))
        fill.setStrokeWidth(0.6)
        symbol.appendSymbolLayer(fill)
        layer.renderer().setSymbol(symbol)

        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPolygonXY(rings))
        layer.startEditing()
        layer.addFeature(feat)
        layer.commitChanges()

        if self._is_layer_alive(self._boundary_layer):
            project.removeMapLayer(self._boundary_layer.id())

        self._boundary_layer = layer
        project.addMapLayer(layer)
        self._log(
            f"Mesh boundary extracted: {len(outer) - 1} outer vertices, "
            f"{len(isl_bnds)} island hole(s).",
            Qgis.MessageLevel.Success,
            push=True,
        )

    # ------------------------------------------------------------------
    # Button 9: Load NWM Reaches
    # ------------------------------------------------------------------

    @staticmethod
    def _read_nwm_reaches(
        csv_path: Path,
    ) -> list[tuple[str, list[tuple[int, int]]]]:
        """Parse ``nwmReaches.csv`` into ``[(label, [(elem_id, comid), ...]), ...]``.

        SCHISM convention: the first block is sources, the second is sinks.
        Any further blocks are labelled ``block_N`` and treated like sinks
        for styling purposes.
        """
        blocks: list[list[tuple[int, int]]] = []
        with csv_path.open("r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    n = int(stripped)
                except ValueError:
                    continue
                entries: list[tuple[int, int]] = []
                for _ in range(n):
                    entry = f.readline().strip()
                    if not entry:
                        break
                    parts = entry.split()
                    if len(parts) < 2:
                        continue
                    try:
                        entries.append((int(parts[0]), int(parts[1])))
                    except ValueError:
                        continue
                blocks.append(entries)

        labels = ("source", "sink")
        return [
            (labels[i] if i < len(labels) else f"block_{i + 1}", entries)
            for i, entries in enumerate(blocks)
        ]

    @staticmethod
    def _read_gr3_centroids_for(gr3_path: Path, needed: set[int]) -> dict[int, tuple[float, float]]:
        """Compute (x, y) centroids for the requested element ids only.

        Uses a simple arithmetic mean of vertex coordinates — sufficient
        for visualization. The full node-coordinate dict is built in one
        pass; element rows are scanned linearly and centroids are stored
        only for ids in *needed*.
        """
        centroids: dict[int, tuple[float, float]] = {}
        coords: dict[int, tuple[float, float]] = {}
        with gr3_path.open("r") as f:
            f.readline()  # header
            n_elements, n_nodes = (int(x) for x in f.readline().split()[:2])
            for _ in range(n_nodes):
                parts = f.readline().split()
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            for _ in range(n_elements):
                parts = f.readline().split()
                elem_id = int(parts[0])
                if elem_id not in needed:
                    continue
                ncorners = int(parts[1])
                xs = ys = 0.0
                for i in range(ncorners):
                    nid = int(parts[2 + i])
                    x, y = coords[nid]
                    xs += x
                    ys += y
                centroids[elem_id] = (xs / ncorners, ys / ncorners)
        return centroids

    @staticmethod
    def _style_nwm_reaches(layer: QgsVectorLayer) -> None:
        """Apply a categorized source/sink renderer keyed on the ``type`` field."""
        category_specs = (
            ("source", QColor("green")),
            ("sink", QColor("#ff00ee")),
        )
        categories = []
        for label, color in category_specs:
            symbol = QgsMarkerSymbol.createSimple({})
            symbol.deleteSymbolLayer(0)
            marker = QgsSimpleMarkerSymbolLayer(Qgis.MarkerShape.Circle)
            marker.setColor(color)
            marker.setStrokeColor(QColor("white"))
            marker.setStrokeWidth(0.5)
            marker.setSize(4.0)
            symbol.appendSymbolLayer(marker)
            categories.append(QgsRendererCategory(label, symbol, label))
        layer.setRenderer(QgsCategorizedSymbolRenderer("type", categories))

    @staticmethod
    def _build_reach_features(
        layer: QgsVectorLayer,
        blocks: list[tuple[str, list[tuple[int, int]]]],
        centroids: dict[int, tuple[float, float]],
    ) -> tuple[list[QgsFeature], dict[str, int], int]:
        """Build per-reach point features and return (feats, counts, missing)."""
        feats: list[QgsFeature] = []
        counts: dict[str, int] = {}
        missing = 0
        for label, entries in blocks:
            kept = 0
            for elem_id, comid in entries:
                if elem_id not in centroids:
                    missing += 1
                    continue
                x, y = centroids[elem_id]
                feat = QgsFeature(layer.fields())
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                feat.setAttribute("type", label)
                feat.setAttribute("elem_id", elem_id)
                feat.setAttribute("comid", comid)
                feats.append(feat)
                kept += 1
            counts[label] = kept
        return feats, counts, missing

    def _prompt_for_reaches_csv(self) -> Path | None:
        """Show the file picker for nwmReaches.csv and return the chosen path."""
        if self._gr3_path is None:
            return None
        default = self._gr3_path.parent / "nwmReaches.csv"
        start = str(default if default.is_file() else self._gr3_path.parent)
        csv_path_str, _ = QFileDialog.getOpenFileName(
            self.iface.mainWindow(),
            "Select nwmReaches.csv",
            start,
            "NWM Reaches (*.csv *.txt);;All files (*)",
        )
        if not csv_path_str:
            return None
        csv_path = Path(csv_path_str)
        if not csv_path.is_file():
            self._log(f"File not found: {csv_path}", Qgis.MessageLevel.Warning, push=True)
            return None
        return csv_path

    def _on_load_nwm_reaches(self) -> None:
        if self._gr3_path is None:
            self._log("Load a SCHISM mesh first.", Qgis.MessageLevel.Warning, push=True)
            return

        csv_path = self._prompt_for_reaches_csv()
        if csv_path is None:
            return

        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            blocks = self._read_nwm_reaches(csv_path)
            needed = {elem_id for _, entries in blocks for elem_id, _ in entries}
            centroids = self._read_gr3_centroids_for(self._gr3_path, needed)
        except (OSError, ValueError, IndexError, KeyError) as exc:
            self._log(
                f"Failed to load NWM reaches: {exc}",
                Qgis.MessageLevel.Critical,
                push=True,
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not any(entries for _, entries in blocks):
            self._log("No reaches found in CSV.", Qgis.MessageLevel.Warning, push=True)
            return

        # Use ``_is_layer_alive`` rather than a plain ``is not None`` check —
        # the user may have removed the mesh from the Layers panel, leaving
        # a dangling Python proxy whose ``.crs()`` would raise RuntimeError.
        if self._is_layer_alive(self._mesh_layer) and self._mesh_layer.crs().isValid():
            crs_authid = self._mesh_layer.crs().authid()
        else:
            crs_authid = "EPSG:4326"
        uri = f"Point?crs={crs_authid}&field=type:string&field=elem_id:integer&field=comid:integer"
        layer = QgsVectorLayer(uri, "nwm_reaches", "memory")

        feats, counts, missing = self._build_reach_features(layer, blocks, centroids)
        layer.dataProvider().addFeatures(feats)
        self._style_nwm_reaches(layer)

        project = QgsProject.instance()
        if self._is_layer_alive(self._reaches_layer):
            project.removeMapLayer(self._reaches_layer.id())
        self._reaches_layer = layer
        project.addMapLayer(layer)

        summary = ", ".join(f"{n} {label}s" for label, n in counts.items())
        suffix = f" ({missing} element id(s) not in mesh, skipped)" if missing else ""
        self._log(
            f"NWM reaches loaded: {summary}{suffix}.",
            Qgis.MessageLevel.Success,
            push=True,
        )

    # ------------------------------------------------------------------
    # Buttons 10-12: Draw / Edit / Save Split Line
    # ------------------------------------------------------------------

    def _on_draw_line(self) -> None:
        project = QgsProject.instance()

        if self._is_layer_alive(self._line_layer):
            project.removeMapLayer(self._line_layer.id())
        self._line_layer = None

        crs = project.crs()
        crs_str = crs.authid() if crs.isValid() else "EPSG:4326"

        self._line_layer = QgsVectorLayer(f"LineString?crs={crs_str}", "split_line", "memory")
        if not self._line_layer.isValid():
            self._log(
                "Failed to create scratch line layer.",
                Qgis.MessageLevel.Critical,
                push=True,
            )
            return

        symbol = QgsLineSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        line = QgsSimpleLineSymbolLayer()
        line.setColor(QColor(255, 0, 0))
        line.setWidth(0.8)
        symbol.appendSymbolLayer(line)
        self._line_layer.renderer().setSymbol(symbol)

        project.addMapLayer(self._line_layer)
        self._line_layer.startEditing()
        self.iface.setActiveLayer(self._line_layer)
        self.iface.actionAddFeature().trigger()

        self._log("Draw a split line on the map. Right-click to finish.", push=True)

    def _on_edit_line(self) -> None:
        if self._line_layer is None:
            self._log(
                "No split line to edit. Use 'Draw Split Line' first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        if not self._is_layer_alive(self._line_layer):
            self._log(
                "Split-line layer was removed. Use 'Draw Split Line' to create a new one.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            self._line_layer = None
            return

        if not self._line_layer.isEditable():
            self._line_layer.startEditing()

        self.iface.setActiveLayer(self._line_layer)
        self.iface.actionVertexTool().trigger()
        self._log("Vertex editing activated. Click vertices to move them.", push=True)

    # ------------------------------------------------------------------
    # Shared helper: save memory layer to GeoJSON
    # ------------------------------------------------------------------

    def _save_layer_to_geojson(
        self,
        layer: QgsVectorLayer | None,
        dialog_title: str,
        feature_label: str,
        *,
        reproject_to_wgs84: bool,
    ) -> None:
        """Save *layer* to GeoJSON, optionally reprojected to WGS84."""
        project = QgsProject.instance()
        if not self._is_layer_alive(layer):
            self._log(
                f"No {feature_label} to save.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        if layer.isEditable():
            layer.commitChanges()
        if layer.featureCount() == 0:
            self._log(
                f"{feature_label.capitalize()} layer is empty.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self.iface.mainWindow(),
            dialog_title,
            "",
            "GeoJSON (*.geojson)",
        )
        if not save_path:
            return
        if not save_path.endswith(".geojson"):
            save_path += ".geojson"

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GeoJSON"
        options.fileEncoding = "UTF-8"
        if reproject_to_wgs84:
            wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
            if layer.crs().isValid() and layer.crs() != wgs84:
                options.ct = QgsCoordinateTransform(layer.crs(), wgs84, project.transformContext())

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        error_code, error_msg, *_ = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer,
            save_path,
            project.transformContext(),
            options,
        )

        if error_code == QgsVectorFileWriter.NoError:
            self._log(
                f"{feature_label.capitalize()} saved to {save_path}",
                Qgis.MessageLevel.Success,
                push=True,
            )
        else:
            self._log(
                f"Error saving {feature_label}: {error_msg}",
                Qgis.MessageLevel.Critical,
                push=True,
            )

    def _on_save_line(self) -> None:
        self._save_layer_to_geojson(
            self._line_layer,
            "Save Split Line as GeoJSON",
            "split line",
            reproject_to_wgs84=True,
        )

    # ------------------------------------------------------------------
    # Buttons 13-15: Add / Edit / Save Observation Points
    # ------------------------------------------------------------------

    def _on_add_obs(self) -> None:
        project = QgsProject.instance()

        # Reuse layer if it still exists, else create a fresh one
        if not self._is_layer_alive(self._obs_layer):
            crs = project.crs()
            crs_str = crs.authid() if crs.isValid() else "EPSG:4326"
            self._obs_layer = QgsVectorLayer(f"Point?crs={crs_str}", "obs_points", "memory")
            self._obs_layer.dataProvider().addAttributes([QgsField("id", QMetaType.Type.QString)])
            self._obs_layer.updateFields()

            symbol = QgsMarkerSymbol.createSimple({})
            symbol.deleteSymbolLayer(0)
            marker = QgsSimpleMarkerSymbolLayer(Qgis.MarkerShape.Triangle)
            marker.setColor(QColor("yellow"))
            marker.setStrokeColor(QColor("black"))
            marker.setSize(5.0)
            symbol.appendSymbolLayer(marker)
            self._obs_layer.renderer().setSymbol(symbol)
            self._enable_labels(self._obs_layer, "id")

            project.addMapLayer(self._obs_layer)

        if not self._obs_layer.isEditable():
            self._obs_layer.startEditing()
        self.iface.setActiveLayer(self._obs_layer)
        self.iface.actionAddFeature().trigger()

        self._log(
            "Click on the map to add observation points. "
            "Set the 'id' field to give each point a name.",
            push=True,
        )

    def _on_edit_obs(self) -> None:
        if self._obs_layer is None:
            self._log(
                "No observation-points layer to edit. Use 'Add Observation Points' first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        if not self._is_layer_alive(self._obs_layer):
            self._log(
                "Observation-points layer was removed. "
                "Use 'Add Observation Points' to create a new one.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            self._obs_layer = None
            return

        if not self._obs_layer.isEditable():
            self._obs_layer.startEditing()
        self.iface.setActiveLayer(self._obs_layer)
        self.iface.actionVertexTool().trigger()
        self._log("Vertex tool active — drag points to move them.", push=True)

    def _on_save_obs(self) -> None:
        self._save_layer_to_geojson(
            self._obs_layer,
            "Save Observation Points as GeoJSON",
            "observation points",
            reproject_to_wgs84=True,
        )

    def _on_clear_map(self) -> None:
        """Remove every layer added by the plugin except the basemap."""
        project = QgsProject.instance()
        cleared = 0
        for attr in self._MANAGED_LAYER_ATTRS:
            layer = getattr(self, attr, None)
            if self._is_layer_alive(layer):
                with contextlib.suppress(RuntimeError):
                    if hasattr(layer, "isEditable") and layer.isEditable():
                        layer.rollBack()
                project.removeMapLayer(layer.id())
                cleared += 1
            setattr(self, attr, None)

        for attr in ("_temp_2dm_path", "_temp_dat_path"):
            old = getattr(self, attr, None)
            if old is not None and old.exists():
                with contextlib.suppress(OSError):
                    old.unlink()
            setattr(self, attr, None)
        self._gr3_path = None

        self.iface.mapCanvas().refresh()
        self._log(
            f"Cleared {cleared} layer(s); basemap preserved.",
            Qgis.MessageLevel.Success,
            push=True,
        )

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def _add_spacer(self, width: int) -> None:
        """Add an invisible fixed-width widget so toolbar groups have breathing room."""
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer.setFixedWidth(width)
        self._actions.append(self._toolbar.addWidget(spacer))

    def _add_separator(self) -> None:
        """Add a visible group separator (8-px gap, line, 8-px gap)."""
        self._add_spacer(8)
        self._actions.append(self._toolbar.addSeparator())
        self._add_spacer(8)

    def initGui(self) -> None:  # noqa: N802
        """Create the toolbar and register actions."""
        self._toolbar = self.iface.addToolBar(PLUGIN_NAME)
        self._toolbar.setObjectName("NWMCoastalToolbar")

        # Group: Basemap (XYZ tile add — OSM is loaded as XYZ)
        self._add_action("mActionAddXyzLayer.svg", "Add Basemap", self._on_add_basemap)
        self._add_separator()

        # Group: SCHISM mesh
        self._add_action("mActionAddMeshLayer.svg", "Load SCHISM Mesh", self._on_load_mesh)
        self._add_action(
            "mActionAddPolygon.svg", "Extract Mesh Boundary", self._on_extract_boundary
        )
        self._add_action(
            "mActionAddDelimitedTextLayer.svg",
            "Load NWM Reaches",
            self._on_load_nwm_reaches,
        )
        self._add_separator()

        # Group: Polygon (SFINCS workflow)
        self._add_action("mActionCapturePolygon.svg", "Draw Polygon", self._on_draw_polygon)
        self._add_action("mActionVertexToolActiveLayer.svg", "Edit Polygon", self._on_edit_polygon)
        self._add_separator()

        # Group: Union — works on any selected polygon (e.g. the one
        # produced by 'Extract Mesh Boundary'), with the sketcher as a
        # fallback when nothing is selected.
        self._add_action(
            "mActionMergeFeatures.svg",
            "Union with NHF Divides",
            self._on_union_divides,
        )
        self._add_action("mActionFileSaveAs.svg", "Save Polygon", self._on_save_polygon)
        self._add_separator()

        # Group: Split line (SCHISM workflow)
        self._add_action("mActionCaptureLine.svg", "Draw Split Line", self._on_draw_line)
        self._add_action("mActionVertexToolActiveLayer.svg", "Edit Split Line", self._on_edit_line)
        self._add_action("mActionFileSave.svg", "Save Split Line", self._on_save_line)
        self._add_separator()

        # Group: Observation points
        self._add_action("mActionCapturePoint.svg", "Add Observation Points", self._on_add_obs)
        self._add_action(
            "mActionVertexToolActiveLayer.svg", "Edit Observation Points", self._on_edit_obs
        )
        self._add_action("mActionSaveAllEdits.svg", "Save Observation Points", self._on_save_obs)
        self._add_separator()

        # Group: Flowpaths export
        self._add_action(
            "mActionExport.svg",
            "Export Selected Flowpaths",
            self._on_export_flowpaths,
        )
        self._add_separator()

        # Group: Clear Map
        self._add_action("mActionRemoveLayer.svg", "Clear Map", self._on_clear_map)

    def unload(self) -> None:
        """Clean up toolbar, actions, and any active edits."""
        for attr in self._MANAGED_LAYER_ATTRS:
            layer = getattr(self, attr, None)
            if not self._is_layer_alive(layer):
                continue
            with contextlib.suppress(RuntimeError):
                if hasattr(layer, "isEditable") and layer.isEditable():
                    layer.commitChanges()

        for action in self._actions:
            self._toolbar.removeAction(action)
        self._actions.clear()

        if self._toolbar is not None:
            self.iface.mainWindow().removeToolBar(self._toolbar)
            self._toolbar = None

        for attr in ("_temp_2dm_path", "_temp_dat_path"):
            old = getattr(self, attr, None)
            if old is not None and old.exists():
                with contextlib.suppress(OSError):
                    old.unlink()
            setattr(self, attr, None)
