import importlib
from . import bases


class Fingerprint(bases.EmbeddingCalculator):
    name = "Fingerprint"
    module = importlib.import_module("qsprpred.data.descriptors.fingerprints")
    abstract = True


class DescriptorSet(bases.EmbeddingCalculator):
    name = "DescriptorSet"
    module = importlib.import_module("qsprpred.data.descriptors.sets")
    abstract = True

    def get_default_parameters(self):
        return {"descriptors": getattr(self.module, self.name)().descriptors}


def create_fingerprint_class(name):
    return type(name, (Fingerprint,), {
        'name': name,
        'abstract': False
    })


def create_descriptor_set_class(name):
    return type(name, (DescriptorSet,), {
        'name': name,
        'abstract': False
    })


fingerprint_types = [
    name for name in dir(Fingerprint.module)
    if name.endswith('FP') and not name.startswith('_')
]

globals().update({
    name: create_fingerprint_class(name)
    for name in fingerprint_types
})

descriptor_set_types = ['DrugExPhyschem', 'RDKitDescs',]

globals().update({
    name: create_descriptor_set_class(name)
    for name in descriptor_set_types
})
