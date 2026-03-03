"""Main plugin module for NWM Coastal."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateTransform,
    QgsFeatureRequest,
    QgsFillSymbol,
    QgsGeometry,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsSimpleFillSymbolLayer,
    QgsSimpleLineSymbolLayer,
    QgsSimpleMarkerSymbolLayer,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QAction, QFileDialog

if TYPE_CHECKING:
    from qgis.gui import QgisInterface

PLUGIN_NAME = "NWM Coastal"


class NWMCoastalPlugin:
    """Main QGIS plugin class for NWM Coastal."""

    def __init__(self, iface: QgisInterface) -> None:
        self.iface = iface
        self._toolbar = None
        self._actions: list[QAction] = []
        self._sketcher_layer: QgsVectorLayer | None = None
        self._merged_layer: QgsVectorLayer | None = None

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def initGui(self) -> None:  # noqa: N802
        """Create the toolbar and register actions."""
        self._toolbar = self.iface.addToolBar(PLUGIN_NAME)
        self._toolbar.setObjectName("NWMCoastalToolbar")

        self._add_action(
            "mActionAddOgrLayer.svg", "Add Basemap", self._on_add_basemap
        )
        self._add_action(
            "mActionCapturePolygon.svg", "Draw Polygon", self._on_draw_polygon
        )
        self._add_action(
            "mActionNodeTool.svg", "Edit Polygon", self._on_edit_polygon
        )
        self._add_action(
            "mActionMergeFeatures.svg",
            "Union with NHF Divides",
            self._on_union_divides,
        )
        self._add_action(
            "mActionFileSaveAs.svg", "Save Polygon", self._on_save_polygon
        )

    def unload(self) -> None:
        """Clean up toolbar, actions, and any active edits."""
        if self._sketcher_layer is not None and self._sketcher_layer.isEditable():
            self._sketcher_layer.commitChanges()

        for action in self._actions:
            self._toolbar.removeAction(action)
        self._actions.clear()

        if self._toolbar is not None:
            del self._toolbar
            self._toolbar = None

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

    # ------------------------------------------------------------------
    # Button 1: Add Basemap
    # ------------------------------------------------------------------

    def _on_add_basemap(self) -> None:
        from .gui.dlg_basemap import BasemapDialog

        dlg = BasemapDialog(self.iface.mainWindow())
        if dlg.exec() != BasemapDialog.Accepted:
            return

        gpkg_path = dlg.gpkg_path
        parquet_path = dlg.parquet_path
        min_order = dlg.min_stream_order
        project = QgsProject.instance()
        root = project.layerTreeRoot()

        # 1. OpenStreetMap (bottom layer)
        osm_uri = (
            "type=xyz"
            "&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            "&zmax=19&zmin=0"
        )
        osm = QgsRasterLayer(osm_uri, "OpenStreetMap", "wms")
        if osm.isValid():
            project.addMapLayer(osm, False)
            root.insertLayer(-1, osm)
        else:
            self._log("Failed to load OpenStreetMap.", Qgis.MessageLevel.Warning, push=True)

        # 2. divides (polygon: transparent fill, black edge 0.1)
        self._add_gpkg_layer(gpkg_path, "divides", project, self._style_divides)

        # 3. flowpaths (linestring: blue, width 1, filtered)
        self._add_gpkg_layer(
            gpkg_path,
            "flowpaths",
            project,
            lambda lyr: self._style_flowpaths(lyr, min_order),
        )

        # 4. gages (point: red, size 3) — optional
        self._add_gpkg_layer(gpkg_path, "gages", project, self._style_gages, optional=True)

        # 5. nexus (point: green, size 2) — optional
        self._add_gpkg_layer(gpkg_path, "nexus", project, self._style_nexus, optional=True)

        # 6. CO-OPS stations (point: orange star, size 7)
        self._add_coops_layer(parquet_path, project)

        self.iface.mapCanvas().refresh()
        self._log("Basemap layers loaded.", Qgis.MessageLevel.Success, push=True)

    def _add_gpkg_layer(
        self,
        gpkg_path: Path,
        layer_name: str,
        project: QgsProject,
        style_fn,
        *,
        optional: bool = False,
    ) -> None:
        uri = f"{gpkg_path}|layername={layer_name}"
        layer = QgsVectorLayer(uri, layer_name, "ogr")
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

    def _add_coops_layer(self, parquet_path: Path, project: QgsProject) -> None:
        layer = QgsVectorLayer(str(parquet_path), "coops", "ogr")
        if not layer.isValid():
            self._log(
                f"Failed to load CO-OPS stations from {parquet_path}",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        symbol = QgsMarkerSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        marker = QgsSimpleMarkerSymbolLayer(Qgis.MarkerShape.Star)
        marker.setColor(QColor("orange"))
        marker.setSize(7.0)
        symbol.appendSymbolLayer(marker)
        layer.renderer().setSymbol(symbol)

        project.addMapLayer(layer)

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
    def _style_flowpaths(layer: QgsVectorLayer, min_stream_order: int) -> None:
        layer.setSubsetString(f'"stream_order" >= {min_stream_order}')
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

    @staticmethod
    def _style_nexus(layer: QgsVectorLayer) -> None:
        symbol = QgsMarkerSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        marker = QgsSimpleMarkerSymbolLayer()
        marker.setColor(QColor("green"))
        marker.setSize(2.0)
        symbol.appendSymbolLayer(marker)
        layer.renderer().setSymbol(symbol)

    # ------------------------------------------------------------------
    # Button 2: Draw Polygon
    # ------------------------------------------------------------------

    def _on_draw_polygon(self) -> None:
        project = QgsProject.instance()

        # Remove previous merged layer if it exists
        if self._merged_layer is not None:
            if project.mapLayer(self._merged_layer.id()) is not None:
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

        if QgsProject.instance().mapLayer(self._sketcher_layer.id()) is None:
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

    def _on_union_divides(self) -> None:
        """Union the sketcher polygon with all intersecting NHF divides.

        The original sketcher layer is kept untouched. The union result is
        stored in a separate merged layer so the user can edit the original
        and re-run the union at any time.
        """
        if self._sketcher_layer is None:
            self._log(
                "No polygon layer. Use 'Draw Polygon' first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        project = QgsProject.instance()

        if project.mapLayer(self._sketcher_layer.id()) is None:
            self._log(
                "Sketcher layer was removed. Use 'Draw Polygon' to create a new one.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            self._sketcher_layer = None
            return

        if self._sketcher_layer.isEditable():
            self._sketcher_layer.commitChanges()

        if self._sketcher_layer.featureCount() == 0:
            self._log(
                "Sketcher layer has no features. Draw a polygon first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return

        # Find the divides layer
        divides_layers = project.mapLayersByName("divides")
        if not divides_layers:
            self._log(
                "No 'divides' layer found. Load basemap first.",
                Qgis.MessageLevel.Warning,
                push=True,
            )
            return
        divides_layer = divides_layers[0]

        # Get the sketcher polygon geometry
        sketcher_feat = next(self._sketcher_layer.getFeatures())
        sketcher_geom = sketcher_feat.geometry()

        # Transform sketcher geometry to divides CRS if they differ
        sketcher_crs = self._sketcher_layer.crs()
        divides_crs = divides_layer.crs()

        if sketcher_crs != divides_crs:
            to_divides = QgsCoordinateTransform(sketcher_crs, divides_crs, project)
            sketcher_geom_div = QgsGeometry(sketcher_geom)
            sketcher_geom_div.transform(to_divides)
        else:
            sketcher_geom_div = sketcher_geom

        # Spatial query: find divides that intersect with the sketcher polygon
        request = QgsFeatureRequest().setFilterRect(sketcher_geom_div.boundingBox())
        geometries = [sketcher_geom_div]
        n_intersecting = 0
        for feat in divides_layer.getFeatures(request):
            feat_geom = feat.geometry()
            if sketcher_geom_div.intersects(feat_geom):
                geometries.append(feat_geom)
                n_intersecting += 1

        if n_intersecting == 0:
            self._log(
                "No divides intersect the drawn polygon.",
                Qgis.MessageLevel.Info,
                push=True,
            )
            return

        # Union all geometries using native GEOS operation (in divides CRS)
        union_geom = QgsGeometry.unaryUnion(geometries)

        # If result is MultiPolygon, merge into single polygon
        if union_geom.isMultipart():
            parts = union_geom.asGeometryCollection()
            union_geom = QgsGeometry.unaryUnion(parts)

        # Remove interior rings (holes) by keeping only the exterior ring
        if union_geom.isMultipart():
            largest = max(union_geom.asGeometryCollection(), key=lambda g: g.area())
            exterior = QgsGeometry.fromPolygonXY([largest.asPolygon()[0]])
        else:
            exterior = QgsGeometry.fromPolygonXY([union_geom.asPolygon()[0]])
        union_geom = exterior

        # Transform union result back to sketcher CRS if needed
        if sketcher_crs != divides_crs:
            to_sketcher = QgsCoordinateTransform(divides_crs, sketcher_crs, project)
            union_geom.transform(to_sketcher)

        # Remove previous merged layer if it exists
        if self._merged_layer is not None:
            if project.mapLayer(self._merged_layer.id()) is not None:
                project.removeMapLayer(self._merged_layer.id())
            self._merged_layer = None

        # Create a new merged layer with the union result
        crs_str = sketcher_crs.authid()
        self._merged_layer = QgsVectorLayer(
            f"Polygon?crs={crs_str}", "merged_polygon", "memory"
        )

        # Style: semi-transparent green fill to distinguish from sketcher
        symbol = QgsFillSymbol.createSimple({})
        symbol.deleteSymbolLayer(0)
        fill = QgsSimpleFillSymbolLayer()
        fill.setColor(QColor(50, 205, 50, 80))  # lime green, ~30% opacity
        fill.setStrokeColor(QColor(0, 128, 0))
        fill.setStrokeWidth(0.5)
        symbol.appendSymbolLayer(fill)
        self._merged_layer.renderer().setSymbol(symbol)

        # Add the union feature
        from qgis.core import QgsFeature

        self._merged_layer.startEditing()
        feat = QgsFeature()
        feat.setGeometry(union_geom)
        self._merged_layer.addFeature(feat)
        self._merged_layer.commitChanges()

        project.addMapLayer(self._merged_layer)

        self.iface.mapCanvas().refresh()
        self._log(
            f"Union complete: merged with {n_intersecting} divides.",
            Qgis.MessageLevel.Success,
            push=True,
        )

    # ------------------------------------------------------------------
    # Button 5: Save Polygon
    # ------------------------------------------------------------------

    def _on_save_polygon(self) -> None:
        # Prefer merged layer if it exists, otherwise fall back to sketcher
        project = QgsProject.instance()
        save_layer = None

        if (
            self._merged_layer is not None
            and project.mapLayer(self._merged_layer.id()) is not None
            and self._merged_layer.featureCount() > 0
        ):
            save_layer = self._merged_layer
        elif self._sketcher_layer is not None:
            if self._sketcher_layer.isEditable():
                self._sketcher_layer.commitChanges()
            if self._sketcher_layer.featureCount() > 0:
                save_layer = self._sketcher_layer

        if save_layer is None:
            self._log(
                "No polygon to save. Use 'Draw Polygon' first.",
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

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        error_code, error_msg, *_ = QgsVectorFileWriter.writeAsVectorFormatV3(
            save_layer,
            save_path,
            QgsProject.instance().transformContext(),
            options,
        )

        if error_code == QgsVectorFileWriter.NoError:
            self._log(f"Polygon saved to {save_path}", Qgis.MessageLevel.Success, push=True)
        else:
            self._log(f"Error saving polygon: {error_msg}", Qgis.MessageLevel.Critical, push=True)
