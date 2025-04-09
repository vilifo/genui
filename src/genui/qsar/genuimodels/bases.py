"""
Created by: Martin Sicho
On: 14-01-20, 10:16
"""

from abc import ABC
from genui.utils.inspection import findSubclassByID, importFromPackage
from genui.qsar import models
from genui.utils.inspection import get_default_params


class EmbeddingCalculator(ABC):
    name = None
    _module = None

    def __init__(self, builder):
        self.builder = builder
        self.instance = None

    def __call__(self, **kwargs):
        self.instance = getattr(self._module, self.name)(**kwargs)
        return self.instance

    def get_default_parameters(self):
        return get_default_params(self._module, self.name)

    @classmethod
    def getDjangoModel(cls, corePackage, update=False) -> models.EmbeddingCalculator:
        if not cls.name:
            raise Exception('You have to specify a name for the embedding group in its class "group_name" property')

        ret, ret_created = models.EmbeddingCalculator.objects.get_or_create(name=cls.name)

        # just return if we are not setting up a new instance
        if not ret_created and not update:
            return ret

        if corePackage:
            ret.corePackage = corePackage
            ret.save()

        return ret


class EmbeddingBuilderMixIn:

    @staticmethod
    def findEmbeddingClass(name, corePackage):
        module = importFromPackage(corePackage, "embeddings")
        return findSubclassByID(EmbeddingCalculator, module, "name", name)

    def __init__(self, instance: models.Model, progress=None, onFitCall=None):
        super().__init__(instance, progress, onFitCall)
        self.molsets = [self.instance.molset] if hasattr(self.instance, "molset") else self.instance.molsets.all()
        classes = {self.findEmbeddingClass(x.name, x.corePackage): x.arguments for x in self.training.embeddings.all()}
        self.embeddingCalculators = [class_(self)(**kwargs) for class_, kwargs in classes.items()]
