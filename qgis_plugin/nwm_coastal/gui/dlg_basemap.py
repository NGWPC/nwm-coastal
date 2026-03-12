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

REQUIRED_GPKG_LAYERS_ALL = ("flowpaths", "divides")
REQUIRED_GPKG_LAYERS_NO_FLOWPATHS = ("divides",)


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

    def _browse_nwm_flowlines(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select NWM Flowlines GeoPackage", "", "GeoPackage (*.gpkg)"
        )
        if path:
            self.nwm_flowlines_edit.setText(path)

    @staticmethod
    def _validate_gpkg(path: Path, label: str) -> str | None:
        """Return an error if *path* is not a valid GeoPackage, else ``None``."""
        if not path.is_file():
            return f"{label} not found: {path}"
        if path.suffix != ".gpkg":
            return f"{label} must have .gpkg extension."
        return None

    def _check_paths(self) -> str | None:
        """Return an error message if file paths are invalid, else ``None``."""
        if not self.gpkg_edit.text().strip():
            return "Please select a GeoPackage file."
        if not self.parquet_edit.text().strip():
            return "Please select a CO-OPS Parquet file."

        error = self._validate_gpkg(self.gpkg_path, "GeoPackage")
        if error:
            return error
        if not self.parquet_path.is_file():
            return f"Parquet file not found: {self.parquet_path}"

        nwm = self.nwm_flowlines_edit.text().strip()
        if nwm:
            error = self._validate_gpkg(Path(nwm), "NWM Flowlines")
            if error:
                return error
        return None

    @staticmethod
    def _get_stream_order_range(
        gpkg_path: str, layer_name: str, col_name: str
    ) -> tuple[int, int] | str:
        """Query min/max of *col_name* from *layer_name* via SQLite.

        Returns ``(min_val, max_val)`` on success or an error string.
        GeoPackage is SQLite-backed, so we query it directly.
        """
        import sqlite3

        try:
            conn = sqlite3.connect(gpkg_path)
            # Verify the column exists first; SQLite silently treats
            # unmatched double-quoted names as string literals.
            cur = conn.execute(f'PRAGMA table_info("{layer_name}")')
            columns = {row[1] for row in cur.fetchall()}
            if col_name not in columns:
                conn.close()
                return (
                    f"Column '{col_name}' not found in '{layer_name}'. "
                    f"Available columns: {', '.join(sorted(columns))}"
                )
            # col_name and layer_name are validated above against the schema,
            # so this is safe from injection.
            cur = conn.execute(
                f"SELECT MIN({col_name}), MAX({col_name}) FROM [{layer_name}]"  # noqa: S608
            )
            row = cur.fetchone()
            conn.close()
        except sqlite3.OperationalError as exc:
            return str(exc)

        if row is None or row[0] is None or row[1] is None:
            return f"Column '{col_name}' contains only NULL values in '{layer_name}'."
        return (int(row[0]), int(row[1]))

    @staticmethod
    def _check_gpkg_layers(gpkg_path: str, required: set[str]) -> str | None:
        """Return an error if *gpkg_path* is missing any *required* layers."""
        from osgeo import ogr

        ds = ogr.Open(gpkg_path)
        if ds is None:
            return f"Cannot open GeoPackage: {gpkg_path}"
        available = {ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())}
        ds = None
        missing = required - available
        if missing:
            return f"GeoPackage missing required layers: {', '.join(sorted(missing))}"
        return None

    def _validate_and_accept(self) -> None:
        """Validate inputs before accepting the dialog."""
        error = self._check_paths()
        if error:
            self.validation_label.setText(error)
            return

        has_nwm = bool(self.nwm_flowlines_edit.text().strip())
        required = REQUIRED_GPKG_LAYERS_NO_FLOWPATHS if has_nwm else REQUIRED_GPKG_LAYERS_ALL

        error = self._check_gpkg_layers(str(self.gpkg_path), set(required))
        if error:
            self.validation_label.setText(error)
            return

        # Determine which gpkg/layer to use for flowpaths
        stream_order_col = self.stream_order_column
        if has_nwm:
            fp_gpkg = str(self.nwm_flowlines_path)
            fp_layer = self.nwm_layer_name
            error = self._check_gpkg_layers(fp_gpkg, {fp_layer})
            if error:
                self.validation_label.setText(error)
                return
        else:
            fp_gpkg = str(self.gpkg_path)
            fp_layer = "flowpaths"

        # Validate stream order value against actual data range
        result = self._get_stream_order_range(fp_gpkg, fp_layer, stream_order_col)
        if isinstance(result, str):
            self.validation_label.setText(result)
            return

        so_min, so_max = result
        user_val = self.min_stream_order
        if user_val < so_min or user_val > so_max:
            self.validation_label.setText(
                f"Min Stream Order must be between {so_min} and {so_max} "
                f"(range found in '{fp_layer}')."
            )
            return

        self.validation_label.setText("")
        self.accept()

    def _init_nwm_section(self, form: QFormLayout) -> None:
        """Build the optional NWM Flowlines Override section."""
        separator = QLabel("<b>NWM Flowlines Override (optional)</b>")
        form.addRow(separator)

        self.nwm_flowlines_edit = QLineEdit()
        self.nwm_flowlines_edit.setPlaceholderText(
            "Path to NWM flowlines .gpkg (replaces NHF flowpaths)"
        )
        nwm_browse = QPushButton("Browse...")
        nwm_browse.clicked.connect(self._browse_nwm_flowlines)
        nwm_row = QHBoxLayout()
        nwm_row.addWidget(self.nwm_flowlines_edit)
        nwm_row.addWidget(nwm_browse)
        form.addRow("NWM Flowlines:", nwm_row)

        self.nwm_layer_name_edit = QLineEdit()
        self.nwm_layer_name_edit.setText("flowpaths")
        self.nwm_layer_name_edit.setPlaceholderText("Layer name in NWM flowlines gpkg")
        form.addRow("Flowpaths Layer:", self.nwm_layer_name_edit)

        self.stream_order_col_edit = QLineEdit()
        self.stream_order_col_edit.setText("stream_order")
        self.stream_order_col_edit.setPlaceholderText("Column name for stream order")
        form.addRow("Stream Order Column:", self.stream_order_col_edit)

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

        # Optional NWM Flowlines Override
        self._init_nwm_section(form)

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

    @property
    def nwm_flowlines_path(self) -> Path | None:
        """Return NWM flowlines GeoPackage path, or ``None`` if not set."""
        text = self.nwm_flowlines_edit.text().strip()
        return Path(text) if text else None

    @property
    def nwm_layer_name(self) -> str:
        """Return the layer name within the NWM flowlines gpkg."""
        return self.nwm_layer_name_edit.text().strip() or "flowpaths"

    @property
    def stream_order_column(self) -> str:
        """Return the column name for stream order values."""
        return self.stream_order_col_edit.text().strip() or "stream_order"
