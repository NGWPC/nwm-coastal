###############################################################################
#  Module name: SCHISNcMGrid                                                  #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:  12/12/2025                                          #
#                                                                             #
#  Last modification date:                                                    # 
#                                                                             #
#  Description: Manages a SCHISM hgrid.nc file                                #
#                                                                             #
###############################################################################
import numpy as np
import xarray as xr
from pyproj import Transformer

class SCHISMNcGrid:
        """
           SCHISM hgrid.nc file
        """        

        def __init__(self, hgridncfile ): 
           self.source = hgridncfile

           #self.elems = dict()
           #self.centerCoords = dict()
           #self.nodeCoords = dict()
           #self.numElementCounts = []
           grid = xr.open_dataset(hgridncfile)
            
           self._elems = grid['elementConn'].data
           self._centerCoords = grid['centerCoords'].data
           self._nodeCoords = grid['nodeCoords'].data
           self._numElementCounts = grid['numElementConn']

           grid.close()


        @property
        def source(self):
            return self._source

        @source.setter
        def source(self, s):
            self._source = s

        @property
        def elems(self):
            return self._elems

        @elems.setter
        def elems(self, e):
            self._elems = e

        @property
        def centerCoords(self):
            return self._centerCoords

        @centerCoords.setter
        def centerCoords(self, c):
            self._centerCoords = c

        @property
        def nodeCoords(self):
            return self._nodeCoords

        @nodeCoords.setter
        def nodeCoords(self, n):
            self._nodeCoords = n

        @property
        def numElementCounts(self):
            return self._numElementCounts

        @numElementCounts.setter
        def numElementCounts(self, n):
            self._numElementCounts = n

        def getElemLat(self, e ):
            return self.centerCoords[e, 1]

        def getElemLon(self, e ):
            return self.centerCoords[e, 0]

        def getElemCenterCoordsInCrs(self, e, crs_to ):
            # EPSG:4326 is WGS84 (latitude/longitude)
            transformer = Transformer.from_crs("EPSG:4326", crs_to, always_xy=True)

            # Perform the transformation on a specific coordinate
            # Input order is typically longitude, latitude (x, y) for geographic CRS
            # Return order is the same as the input order

            return transformer.transform(self.getElemLon(e), self.getElemLat(e))



