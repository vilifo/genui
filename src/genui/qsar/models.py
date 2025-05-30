from django.db import models
from polymorphic.models import PolymorphicModel

from genui.compounds.models import MolSet, ActivitySet, Activity, ActivityTypes, ActivityUnits
from genui.models.models import Model, TrainingStrategy, DataSplit, ImportableModelComponent


class EmbeddingCalculator(ImportableModelComponent):
    name = models.CharField(max_length=128, blank=False)
    corePackage = models.CharField(blank=False, null=False, default='genui.qsar.genuimodels', max_length=1024)
    arguments = models.JSONField(blank=False, null=False, default=dict)

    def __str__(self):
        return '%s object (%s)' % (self.__class__.__name__, self.name)


class ScaffoldCalculator(EmbeddingCalculator):
    pass


class QSARTrainingStrategy(TrainingStrategy):
    embeddings = models.ManyToManyField(EmbeddingCalculator)
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
    seed = models.IntegerField(blank=True, default=42)
    nClusters = models.IntegerField(blank=True)


class ScaffoldClusters(MoleculeClusters):
    scaffold = models.ForeignKey(ScaffoldCalculator, null=False, on_delete=models.CASCADE)


class FPSimilarityClusters(MoleculeClusters):
    FPCalculator = models.ForeignKey(EmbeddingCalculator, null=False, on_delete=models.CASCADE)


class FPSimilarityMaxMinClusters(FPSimilarityClusters):
    nClusters = models.IntegerField(blank=True)
    seed = models.IntegerField(blank=True, default=42)


class FPSimilarityLeaderPickerClusters(FPSimilarityClusters):
    similarityThreshold = models.FloatField(blank=False)


class TemporalSplit(DataSplit):
    timeSplit = models.FloatField(blank=False)
    timeProp = models.CharField(max_length=128, blank=False)


class GBMTDataSplit(DataSplit):
    clustering = models.ForeignKey(MoleculeClusters, null=False, on_delete=models.CASCADE)
    testFraction = models.FloatField(blank=True, null=True, default=0.8)  # mutually exclusive with nFolds
    # nFolds = models.IntegerField(blank=True, null=True)  # mutually exclusive with testFraction


class GBMTRandomSplit(GBMTDataSplit):
    seed = models.IntegerField(blank=True, default=42)
    nInitialClusters = models.IntegerField(blank=True, null=True, default=2)

    def save(self, *args, **kwargs):
        self.clustering = RandomClusters.objects.create(seed=self.seed, nClusters=self.nInitialClusters)
        super().save(*args, **kwargs)


class ScaffoldSplit(GBMTDataSplit):
    scaffold = models.ForeignKey(ScaffoldCalculator, null=False, on_delete=models.CASCADE)

    def save(self, *args, **kwargs):
        self.clustering = ScaffoldClusters.objects.create(scaffold=self.scaffold)
        super().save(*args, **kwargs)


class ClusterSplit(GBMTDataSplit):
    seed = models.IntegerField(blank=True, default=42)


class QSPRPredSklearnModel(models.Model):
    name = models.CharField()
    type = models.CharField()
    params = models.JSONField()
