"""Basemap input dialog for the NWM Coastal plugin."""

from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class BasemapDialog(QDialog):
    """Dialog for selecting an NHF dataset and the layers to load.

    Both ``.gpkg`` GeoPackage files and ``.gdb`` File Geodatabase folders
    are accepted; OGR opens both transparently.

    Layout (top to bottom):

    1. NHF dataset path with one browse button that auto-detects format.
    2. NHF layers to load — one editable field per category (divides,
       flowpaths, gages, nexus) gated by a checkbox. At least one must
       remain checked.
    3. Optional Flowpaths Override — path to a separate ``.gpkg``/``.gdb``
       that replaces the NHF flowpaths, with the override layer name
       directly beneath.
    4. CO-OPS auto-download checkbox.
    """

    _BROWSE_TOOLTIP = (
        "Pick a .gpkg file directly. For a File Geodatabase, navigate into the "
        ".gdb folder and pick any file inside it — the .gdb path will be detected "
        "automatically."
    )

    def _browse_dataset_into(self, edit: QLineEdit, title: str) -> None:
        """Single browse dialog accepting .gpkg files and .gdb directories.

        Native file pickers don't let you select a ``.gdb`` directory itself
        (it shows up as a folder, not a file). Workaround: navigate *into*
        the ``.gdb`` and pick any internal table file (``*.gdbtable``,
        ``*.gdbtablx``, …); we detect the parent suffix and store the
        ``.gdb`` path itself.
        """
        # Filter must include .gdb internals so they aren't grayed out when
        # the user navigates into a .gdb directory. ``*.gdbtable`` is the
        # safest pick because every File Geodatabase has at least one.
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "GeoPackage / FileGDB (*.gpkg *.gdbtable *.gdbtablx);;"
            "GeoPackage (*.gpkg);;"
            "All files (*)",
        )
        if not path:
            return
        p = Path(path)
        # If the user picked a file inside a .gdb folder, store the .gdb folder.
        if p.is_file() and p.parent.suffix.lower() == ".gdb":
            edit.setText(str(p.parent))
        else:
            edit.setText(str(p))

    @staticmethod
    def _validate_dataset(path: Path, label: str) -> str | None:
        """Return an error if *path* is not a .gpkg file or .gdb directory."""
        suffix = path.suffix.lower()
        if suffix == ".gpkg":
            if not path.is_file():
                return f"{label} not found: {path}"
            return None
        if suffix == ".gdb":
            if not path.is_dir():
                return f"{label} (.gdb) must be an existing directory: {path}"
            return None
        return f"{label} must be a .gpkg file or a .gdb directory: {path}"

    @staticmethod
    def _list_layers(dataset_path: str) -> set[str] | str:
        """Return the set of layer names in *dataset_path* (.gpkg or .gdb)."""
        from osgeo import ogr

        ds = ogr.Open(dataset_path)
        if ds is None:
            return f"Cannot open dataset: {dataset_path}"
        layers = {ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())}
        ds = None
        return layers

    def _check_paths(self) -> str | None:
        """Return an error message if file paths are invalid, else ``None``."""
        if not self.gpkg_edit.text().strip():
            return "Please select an NHF dataset (.gpkg file or .gdb folder)."

        error = self._validate_dataset(self.gpkg_path, "NHF dataset")
        if error:
            return error

        override = self.flowpaths_override_edit.text().strip()
        if override:
            error = self._validate_dataset(Path(override), "Flowpaths Override")
            if error:
                return error
        return None

    @staticmethod
    def _check_layer_name(label: str, name: str, available: set[str]) -> str | None:
        """Return an error if *name* is empty or missing from *available*."""
        if not name:
            return f"{label} is enabled but the layer name is empty."
        if name not in available:
            available_str = ", ".join(sorted(available)) or "(none)"
            return f"{label} layer '{name}' not found in dataset. Available layers: {available_str}"
        return None

    def _validate_layer_names(self) -> str | None:
        """Confirm at least one layer is selected and every name resolves."""
        rows = [
            ("Divides", self.divides_check, self.divides_edit),
            ("Flowpaths", self.flowpaths_check, self.flowpaths_edit),
            ("Gages", self.gages_check, self.gages_edit),
            ("Nexus", self.nexus_check, self.nexus_edit),
        ]

        if not any(check.isChecked() for _, check, _ in rows):
            return "Select at least one NHF layer to load."

        nhf_layers = self._list_layers(str(self.gpkg_path))
        if isinstance(nhf_layers, str):
            return nhf_layers

        # Validate every checked NHF row. Skip flowpaths when override is set —
        # the override dataset is validated separately below.
        override_path = self.flowpaths_override_path
        for label, check, edit in rows:
            if not check.isChecked() or (label == "Flowpaths" and override_path is not None):
                continue
            error = self._check_layer_name(label, edit.text().strip(), nhf_layers)
            if error:
                return error

        if override_path is not None and self.flowpaths_check.isChecked():
            override_layers = self._list_layers(str(override_path))
            if isinstance(override_layers, str):
                return override_layers
            return self._check_layer_name(
                "Flowpaths Override", self.flowpaths_override_layer, override_layers
            )

        return None

    def _validate_and_accept(self) -> None:
        """Validate inputs before accepting the dialog."""
        error = self._check_paths()
        if error:
            self.validation_label.setText(error)
            return

        error = self._validate_layer_names()
        if error:
            self.validation_label.setText(error)
            return

        self.validation_label.setText("")
        self.accept()

    @staticmethod
    def _make_layer_row(label: str, default: str) -> tuple[QCheckBox, QLineEdit]:
        """Build a (checkbox-as-label, line-edit) pair for one NHF layer."""
        check = QCheckBox(label)
        check.setChecked(True)
        edit = QLineEdit(default)
        edit.setPlaceholderText("Layer name in dataset")
        check.toggled.connect(edit.setEnabled)
        return check, edit

    def _init_layers_section(self, form: QFormLayout) -> None:
        """Build the editable, checkbox-toggled list of NHF layer names."""
        form.addRow(QLabel("<b>Layers to Load (NHF)</b>"))

        self.divides_check, self.divides_edit = self._make_layer_row("Divides", "divides")
        form.addRow(self.divides_check, self.divides_edit)

        self.flowpaths_check, self.flowpaths_edit = self._make_layer_row("Flowpaths", "flowpaths")
        form.addRow(self.flowpaths_check, self.flowpaths_edit)

        self.gages_check, self.gages_edit = self._make_layer_row("Gages", "gages")
        form.addRow(self.gages_check, self.gages_edit)

        self.nexus_check, self.nexus_edit = self._make_layer_row("Nexus", "nexus")
        form.addRow(self.nexus_check, self.nexus_edit)

    def _make_browse_button(self, edit: QLineEdit, title: str) -> QPushButton:
        """Build a single browse button that handles both .gpkg and .gdb."""
        btn = QPushButton("Browse...")
        btn.setToolTip(self._BROWSE_TOOLTIP)
        btn.clicked.connect(lambda: self._browse_dataset_into(edit, title))
        return btn

    def _init_override_section(self, form: QFormLayout) -> None:
        """Build the optional Flowpaths Override section."""
        form.addRow(QLabel("<b>Flowpaths Override (optional)</b>"))

        self.flowpaths_override_edit = QLineEdit()
        self.flowpaths_override_edit.setPlaceholderText(
            "Path to alternate flowpaths .gpkg or .gdb (replaces NHF flowpaths)"
        )
        self._override_browse = self._make_browse_button(
            self.flowpaths_override_edit, "Select Flowpaths Override"
        )
        override_row = QHBoxLayout()
        override_row.addWidget(self.flowpaths_override_edit)
        override_row.addWidget(self._override_browse)
        form.addRow("Flowpaths Override:", override_row)

        self.flowpaths_override_layer_edit = QLineEdit("flowpaths")
        self.flowpaths_override_layer_edit.setPlaceholderText(
            "Layer name within the override dataset"
        )
        form.addRow("Flowpath Layer Name:", self.flowpaths_override_layer_edit)

        # Override only applies when Flowpaths is enabled — disable when it is not.
        for widget in (
            self.flowpaths_override_edit,
            self.flowpaths_override_layer_edit,
            self._override_browse,
        ):
            self.flowpaths_check.toggled.connect(widget.setEnabled)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Basemap")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # NHF dataset (GeoPackage file or File Geodatabase folder)
        self.gpkg_edit = QLineEdit()
        self.gpkg_edit.setPlaceholderText("Path to National HydroFabric .gpkg file or .gdb folder")
        gpkg_browse = self._make_browse_button(self.gpkg_edit, "Select NHF Dataset")
        gpkg_row = QHBoxLayout()
        gpkg_row.addWidget(self.gpkg_edit)
        gpkg_row.addWidget(gpkg_browse)
        form.addRow("NHF Dataset:", gpkg_row)

        self._init_layers_section(form)
        self._init_override_section(form)

        # CO-OPS auto-download
        self.coops_check = QCheckBox("Include CO-OPS stations (auto-download)")
        self.coops_check.setChecked(True)
        self.coops_check.setToolTip(
            "Fetch the NOAA CO-OPS water-level station list and cache it under the "
            "QGIS profile so subsequent sessions load instantly."
        )
        form.addRow("CO-OPS Stations:", self.coops_check)

        layout.addLayout(form)

        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: red;")
        self.validation_label.setWordWrap(True)
        layout.addWidget(self.validation_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def gpkg_path(self) -> Path:
        """Return the NHF GeoPackage file path."""
        return Path(self.gpkg_edit.text().strip())

    @property
    def divides_layer(self) -> str:
        """Return the divides layer name; empty when the checkbox is off."""
        if not self.divides_check.isChecked():
            return ""
        return self.divides_edit.text().strip()

    @property
    def flowpaths_layer(self) -> str:
        """Return the NHF flowpaths layer name; empty when the checkbox is off."""
        if not self.flowpaths_check.isChecked():
            return ""
        return self.flowpaths_edit.text().strip()

    @property
    def gages_layer(self) -> str:
        """Return the gages layer name; empty when the checkbox is off."""
        if not self.gages_check.isChecked():
            return ""
        return self.gages_edit.text().strip()

    @property
    def nexus_layer(self) -> str:
        """Return the nexus layer name; empty when the checkbox is off."""
        if not self.nexus_check.isChecked():
            return ""
        return self.nexus_edit.text().strip()

    @property
    def flowpaths_override_path(self) -> Path | None:
        """Return the override gpkg path; ``None`` when Flowpaths is off."""
        if not self.flowpaths_check.isChecked():
            return None
        text = self.flowpaths_override_edit.text().strip()
        return Path(text) if text else None

    @property
    def flowpaths_override_layer(self) -> str:
        """Return the layer name to use within the override gpkg."""
        return self.flowpaths_override_layer_edit.text().strip() or "flowpaths"

    @property
    def include_coops(self) -> bool:
        """Return ``True`` if CO-OPS stations should be downloaded and added."""
        return self.coops_check.isChecked()
