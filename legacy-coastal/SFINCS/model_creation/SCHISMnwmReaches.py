###############################################################################
#  Module name: SCHISMnwmReaches                                              #
#                                                                             #
#  Author     : Zhengtao Cui (Zhengtao.Cui@rtx.com)                           #
#                                                                             #
#  Initial version date:  12/12/2025                                          #
#                                                                             #
#  Last modification date:                                                    # 
#                                                                             #
#  Description: Manages a SCHISM corsswalk file, nwmReaches.csv               #
#                                                                             #
###############################################################################
import os
from os import path

class SCHISMnwmReaches:
        """
           SCHISM nwmReaches.csv file
        """        

        def __init__(self, reachfile ): 
           self.source = reachfile

           self._soelem_ids = dict()
           self._sielem_ids = dict()
           with open(path.join(reachfile)) as f:
             nso = int(f.readline())
             for i in range(nso):
                line = f.readline()
                self._soelem_ids[int(line.split()[0]) -1] = line.split()[1]
             next(f)
             nsi = int(f.readline())
             for i in range(nsi):
                line = f.readline()
                self._sielem_ids[int(line.split()[0]) -1 ] = line.split()[1]

        @property
        def source(self):
            return self._source

        @source.setter
        def source(self, s):
            self._source = s

        @property
        def soelem_ids(self):
            return self._soelem_ids

        @soelem_ids.setter
        def soelem_ids(self, e):
            self._soelem_ids = e

        @property
        def sielem_ids(self):
            return self._sielem_ids

        @sielem_ids.setter
        def sielem_ids(self, c):
            self._sielem_ids = c
