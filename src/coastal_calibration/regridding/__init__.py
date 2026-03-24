"""ESMF-based regridding utilities for coastal model coupling.

This package provides clean, reusable regridding functionality for:
- ESTOFS water level data to SCHISM open boundary nodes
- WRF-Hydro forcing data to regular lat-lon grids and SCHISM mesh elements

Built on ESMF/ESMPy with MPI parallelism, inspired by xESMF's design patterns.
"""
