"""
Created by: Martin Sicho
On: 14-01-20, 10:16
"""

from abc import ABC
import importlib
from genui.utils.inspection import findSubclassByID, importFromPackage
from genui.qsar import models
from genui.utils.inspection import get_default_params


class EmbeddingCalculator(ABC):
    name = None
    _module = None
    _model = models.EmbeddingCalculator

    def __init__(self, builder):
        self.builder = builder
        self.instance = None

    def __call__(self, **kwargs):
        self.instance = getattr(self._module, self.name)(**kwargs)
        return self.instance

    def get_default_parameters(self):
        return get_default_params(self.name, self._module.__name__)

    @classmethod
    def getDjangoModel(cls, corePackage, update=False):
        if not cls.name:
            raise Exception('You have to specify a name for the embedding group in its class "name" property')

        ret, ret_created = cls._model.objects.get_or_create(name=cls.name)

        # just return if we are not setting up a new instance
        if not ret_created and not update:
            return ret

        if corePackage:
            ret.corePackage = corePackage
            ret.save()

        return ret


class ScaffoldCalculator(EmbeddingCalculator):
    name = None
    _module = importlib.import_module("qsprpred.data.chem.scaffolds")
    _model = models.ScaffoldCalculator


class EmbeddingBuilderMixIn:

    @staticmethod
    def findEmbeddingClass(name, corePackage, subtype="embeddings"):
        module = importFromPackage(corePackage, subtype)
        return findSubclassByID(EmbeddingCalculator, module, "name", name)

    def __init__(self, instance: models.Model, progress=None, onFitCall=None):
        super().__init__(instance, progress, onFitCall)
        self.molsets = [self.instance.molset] if hasattr(self.instance, "molset") else self.instance.molsets.all()
        self.embeddingCalculators = [self._init_embedding_calculator(x) for x in self.training.embeddings.all()]

    def _init_embedding_calculator(self, django_model, subtype="embeddings"):
        class_ = self.findEmbeddingClass(django_model.name, django_model.corePackage, subtype)
        return class_(self)(**django_model.arguments) if class_ else None
