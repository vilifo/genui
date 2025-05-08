"""

Created by: Martin Sicho
On: 16-01-20, 11:08
"""

import importlib
from . import bases


class Fingerprint(bases.EmbeddingCalculator):
    name = "Fingerprint"
    _module = importlib.import_module("qsprpred.data.descriptors.fingerprints")
    abstract = True


class MorganFP(Fingerprint):
    name = "MorganFP"
    abstract = False


class RDKitMACCSFP(Fingerprint):
    name = "RDKitMACCSFP"
    abstract = False


class MaccsFP(Fingerprint):
    name = "MaccsFP"
    abstract = False


class AvalonFP(Fingerprint):
    name = "AvalonFP"
    abstract = False


class TopologicalFP(Fingerprint):
    name = "TopologicalFP"
    abstract = False


class AtomPairFP(Fingerprint):
    name = "AtomPairFP"
    abstract = False

class RDKitFP(Fingerprint):
    name = "RDKitFP"
    abstract = False


class PatternFP(Fingerprint):
    name = "PatternFP"
    abstract = False


class LayeredFP(Fingerprint):
    name = "LayeredFP"
    abstract = False


class DescriptorSet(bases.EmbeddingCalculator):
    name = "DescriptorSet"
    _module = importlib.import_module("qsprpred.data.descriptors.sets")
    abstract = True

    def get_descriptors_names(self):
        return getattr(self._module, self.name)().descriptors


class DrugExPhyschem(DescriptorSet):
    name = "DrugExPhyschem"
    abstract = False


class RDKitDescs(DescriptorSet):
    name = "RDKitDescs"
    abstract = False
