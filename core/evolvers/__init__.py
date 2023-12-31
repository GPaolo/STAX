# Created by Giuseppe Paolo 
# Date: 27/07/2020

from core.evolvers.base_evolver import BaseEvolver
from core.evolvers.emitter_evolver import EmitterEvolver
from core.evolvers.ns import NoveltySearch
from core.evolvers.cma_es import CMAES
from core.evolvers.nsga_ii import NSGAII
from core.evolvers.serene import SERENE
from core.evolvers.map_elites import MAPElites
from core.evolvers.cma_me import CMAME
from core.evolvers.random import RandomSearch
from core.evolvers.surprise import SurpriseSearch
from core.evolvers.taxons import TaxonsSearch, TaxonsPareto
from core.evolvers.serene_taxons import SereneTaxons, STAXPareto, STAX_NT
from core.evolvers.nslc import NSLC