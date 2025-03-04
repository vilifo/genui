import json

from django.db import models
from polymorphic.models import PolymorphicModel

from genui.compounds.models import MolSet, ActivitySet, Activity, ActivityTypes, ActivityUnits, Molecule
from genui.models.models import Model, TrainingStrategy, DataSplit

from genui.models.models import ImportableModelComponent
from genui.utils.inspection import get_non_abstract_classes_from_module

QSPRPredScaffoldAlgorithms = [(fun, fun) for fun in get_non_abstract_classes_from_module('qsprpred.data.chem.scaffolds')]


class DescriptorGroup(ImportableModelComponent):
    name = models.CharField(max_length=128, blank=False)
    corePackage = models.CharField(blank=False, null=False, default='genui.qsar.genuimodels', max_length=1024)
    arguments = models.JSONField(blank=False, null=False, default=dict)

    def __str__(self):
        return '%s object (%s)' % (self.__class__.__name__, self.name)

class QSARTrainingStrategy(TrainingStrategy):
    descriptors = models.ManyToManyField(DescriptorGroup)
    activityThreshold = models.FloatField(null=True)
    activitySet = models.ForeignKey(ActivitySet, null=True, on_delete=models.CASCADE)
    activityType = models.ForeignKey(ActivityTypes, on_delete=models.CASCADE, null=True)

class QSARModel(Model):
    molset = models.ForeignKey(MolSet, null=True, on_delete=models.CASCADE, related_name="models")
    predictionsType = models.ForeignKey(ActivityTypes, on_delete=models.CASCADE, null=True)
    predictionsUnits = models.ForeignKey(ActivityUnits, on_delete=models.CASCADE, null=True)

class ModelActivitySet(ActivitySet):
    model = models.ForeignKey(QSARModel, null=False, on_delete=models.CASCADE, related_name="predictions")

class ModelActivity(Activity):
    pass

class MoleculeClusters(PolymorphicModel):
    pass

class RandomClusters(MoleculeClusters):
    seed = models.IntegerField(blank=False, default=42)
    nClusters = models.IntegerField(blank=True)
    IDProp = models.CharField(max_length=128, blank=True)


class ScaffoldClusters(MoleculeClusters):
    scaffold = models.CharField(max_length=128, blank=False, default="BemisMurckoRDKit", choices=QSPRPredScaffoldAlgorithms)
    IDProp = models.CharField(max_length=128, blank=True)

class FPSimilarityClusters(MoleculeClusters):
    FPCalculator = models.ForeignKey(DescriptorGroup, null=False, on_delete=models.CASCADE)
    IDProp = models.CharField(max_length=128, blank=True)

class FPSimilarityMaxMinClusters(FPSimilarityClusters):
    nClusters = models.IntegerField(blank=True)
    seed = models.IntegerField(blank=True, default=42)
    # initialCentroids = models.ManyToManyField(Molecule, blank=True)  # TODO: implement this - list[str]

class FPSimilarityLeaderPickerClusters(FPSimilarityClusters):
    similarityThreshold = models.FloatField(blank=False)

class BootstrapSplit(DataSplit):
    split = models.ForeignKey(DataSplit, null=False, on_delete=models.CASCADE, related_name="bootstrappedSplits")
    nBootstraps = models.IntegerField(blank=False)
    seed = models.IntegerField(blank=True, default=42)


class TemporalSplit(DataSplit):
    timeSplit = models.FloatField(blank=False)
    timeProp = models.CharField(max_length=128, blank=False)

class ClusterSplit(DataSplit):
    testFraction = models.FloatField(blank=False)
    nFolds = models.IntegerField(blank=False)
    customTestList = models.ManyToManyField(Molecule, blank=True)
    seed = models.IntegerField(blank=True, default=42)
    clustering = models.ForeignKey(MoleculeClusters, null=False, on_delete=models.CASCADE)


class GBMTDataSplit(DataSplit):
    clustering = models.ForeignKey(MoleculeClusters, null=False, on_delete=models.CASCADE)
    testFraction = models.FloatField(blank=False)
    nFolds = models.IntegerField(blank=False)
    customTestList = models.ManyToManyField(Molecule, blank=True)


class GBMTRandomSplit(GBMTDataSplit):
    seed = models.IntegerField(blank=True, default=42)
    nInitialClusters = models.IntegerField(blank=True)


class ScaffoldSplit(DataSplit):
    scaffold = models.CharField(max_length=128, blank=False, default="BemisMurckoRDKit", choices=QSPRPredScaffoldAlgorithms)
    testFraction = models.FloatField(blank=False)
    nFolds = models.IntegerField(blank=False)
    customTestList = models.ManyToManyField(Molecule, blank=True)

class QSPRPredSklearnModel(models.Model):
    name = models.CharField()
    type = models.CharField()
    params = models.JSONField()
