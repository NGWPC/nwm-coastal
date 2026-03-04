"""Basemap input dialog for the NWM Coastal plugin."""

from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

REQUIRED_GPKG_LAYERS = ("flowpaths", "divides")


class BasemapDialog(QDialog):
    """Dialog for selecting NHF GeoPackage, CO-OPS parquet, and stream order filter."""

    def _browse_gpkg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select NHF GeoPackage", "", "GeoPackage (*.gpkg)"
        )
        if path:
            self.gpkg_edit.setText(path)

    def _browse_parquet(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CO-OPS Stations Parquet", "", "Parquet (*.parquet)"
        )
        if path:
            self.parquet_edit.setText(path)

    def _validate_and_accept(self) -> None:
        """Validate inputs before accepting the dialog."""
        gpkg = self.gpkg_path
        parquet = self.parquet_path

        if not gpkg.is_file():
            self.validation_label.setText(f"GeoPackage not found: {gpkg}")
            return
        if gpkg.suffix != ".gpkg":
            self.validation_label.setText("GeoPackage file must have .gpkg extension.")
            return
        if not parquet.is_file():
            self.validation_label.setText(f"Parquet file not found: {parquet}")
            return

        # Validate gpkg contains required layers
        from osgeo import ogr

        ds = ogr.Open(str(gpkg))
        if ds is None:
            self.validation_label.setText(f"Cannot open GeoPackage: {gpkg}")
            return

        available = {ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())}
        ds = None

        missing = set(REQUIRED_GPKG_LAYERS) - available
        if missing:
            self.validation_label.setText(
                f"GeoPackage missing required layers: {', '.join(sorted(missing))}"
            )
            return

        self.validation_label.setText("")
        self.accept()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Basemap")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # NHF GeoPackage path
        self.gpkg_edit = QLineEdit()
        self.gpkg_edit.setPlaceholderText("Path to National HydroFabric .gpkg")
        gpkg_browse = QPushButton("Browse...")
        gpkg_browse.clicked.connect(self._browse_gpkg)
        gpkg_row = QHBoxLayout()
        gpkg_row.addWidget(self.gpkg_edit)
        gpkg_row.addWidget(gpkg_browse)
        form.addRow("NHF GeoPackage:", gpkg_row)

        # CO-OPS Parquet path
        self.parquet_edit = QLineEdit()
        self.parquet_edit.setPlaceholderText("Path to CO-OPS stations .parquet")
        parquet_browse = QPushButton("Browse...")
        parquet_browse.clicked.connect(self._browse_parquet)
        parquet_row = QHBoxLayout()
        parquet_row.addWidget(self.parquet_edit)
        parquet_row.addWidget(parquet_browse)
        form.addRow("CO-OPS Stations:", parquet_row)

        # Min stream order
        self.stream_order_spin = QSpinBox()
        self.stream_order_spin.setMinimum(1)
        self.stream_order_spin.setMaximum(10)
        self.stream_order_spin.setValue(3)
        form.addRow("Min Stream Order:", self.stream_order_spin)

        layout.addLayout(form)

        # Validation label
        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: red;")
        layout.addWidget(self.validation_label)

        # OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def gpkg_path(self) -> Path:
        """Return the NHF GeoPackage file path."""
        return Path(self.gpkg_edit.text())

    @property
    def parquet_path(self) -> Path:
        """Return the CO-OPS stations parquet file path."""
        return Path(self.parquet_edit.text())

    @property
    def min_stream_order(self) -> int:
        """Return the minimum stream order filter value."""
        return self.stream_order_spin.value()
