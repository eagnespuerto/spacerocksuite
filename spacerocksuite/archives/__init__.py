"""Plate and image archives that can be searched along a track."""
from .base import PlateHit, Archive, write_csv
from .ssois import SSOIS
from .dasch import DASCH
from .supercosmos import SuperCOSMOS
from .dss import DSS

ALL = {"SSOIS": SSOIS, "DASCH": DASCH, "SuperCOSMOS": SuperCOSMOS, "DSS": DSS}
