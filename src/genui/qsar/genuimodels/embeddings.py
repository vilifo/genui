"""

Created by: Martin Sicho
On: 16-01-20, 11:08
"""

import importlib
from . import bases


class Fingerprint(bases.EmbeddingCalculator):
    name = "Fingerprint"
    _module = importlib.import_module("qsprpred.data.descriptors.fingerprints")


class MorganFP(Fingerprint):
    name = "MorganFP"


class RDKitMACCSFP(Fingerprint):
    name = "RDKitMACCSFP"


class MaccsFP(Fingerprint):
    name = "MaccsFP"


class AvalonFP(Fingerprint):
    name = "AvalonFP"


class TopologicalFP(Fingerprint):
    name = "TopologicalFP"


class AtomPairFP(Fingerprint):
    name = "AtomPairFP"


class RDKitFP(Fingerprint):
    name = "RDKitFP"


class PatternFP(Fingerprint):
    name = "PatternFP"


class LayeredFP(Fingerprint):
    name = "LayeredFP"


class DescriptorSet(bases.EmbeddingCalculator):
    name = "DescriptorSet"
    _module = importlib.import_module("qsprpred.data.descriptors.sets")

    def get_descriptors_names(self):
        return getattr(self._module, self.name)().descriptors


class DrugExPhyschem(DescriptorSet):
    name = "DrugExPhyschem"


class RDKitDescs(DescriptorSet):
    name = "RDKitDescs"
