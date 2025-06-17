"""
serializers

Created by: Martin Sicho
On: 13-01-20, 11:07
"""
import json

from rest_framework import serializers

from genui.compounds.models import ActivityTypes, ActivitySet, ActivityUnits
from genui.compounds.serializers import MolSetSerializer, ActivitySetSerializer, ActivityTypeSerializer, \
    ActivityUnitsSerializer
from genui.models.serializers import TrainingStrategySerializer, ModelSerializer, \
    TrainingStrategyInitSerializer, BasicValidationStrategy, \
    DataSplitSerializer
from genui.models import models as genui_models
from . import models
from .models import EmbeddingCalculator
from ..utils.inspection import get_default_params, SKLEARN_MODELS


class EmbeddingCalculatorSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.EmbeddingCalculator
        fields = ('id', 'name', 'arguments')


class ScaffoldCalculatorSerializer(EmbeddingCalculatorSerializer):
    pass


class QSARTrainingStrategySerializer(TrainingStrategySerializer):
    embeddings = EmbeddingCalculatorSerializer(many=True, required=False)
    activityType = ActivityTypeSerializer(many=False)
    activitySet = ActivitySetSerializer(many=False)

    class Meta:
        model = models.QSARTrainingStrategy
        fields = TrainingStrategySerializer.Meta.fields + (
            'embeddings', 'activityThreshold', 'activityType', 'activitySet')


class QSARTrainingStrategyInitSerializer(TrainingStrategyInitSerializer):
    embeddings = serializers.PrimaryKeyRelatedField(many=True, queryset=models.EmbeddingCalculator.objects.all(),
                                                    allow_empty=True, required=False)
    activityType = serializers.PrimaryKeyRelatedField(many=False, queryset=ActivityTypes.objects.all(), required=False)
    activitySet = serializers.PrimaryKeyRelatedField(many=False, queryset=ActivitySet.objects.all(), required=False)

    class Meta:
        model = models.QSARTrainingStrategy
        fields = TrainingStrategyInitSerializer.Meta.fields + (
            'embeddings', 'activityThreshold', 'activityType', 'activitySet'
        )


class QSARModelSerializer(ModelSerializer):
    trainingStrategy = QSARTrainingStrategySerializer(many=False)
    molset = MolSetSerializer(many=False, required=False)
    predictions = serializers.PrimaryKeyRelatedField(many=True, queryset=models.ActivitySet.objects.all())
    predictionsType = ActivityTypeSerializer(many=False)
    predictionsUnits = ActivityUnitsSerializer(many=False)

    class Meta:
        model = models.QSARModel
        fields = ModelSerializer.Meta.fields + ('molset', 'predictions', 'predictionsType', 'predictionsUnits')
        read_only_fields = ModelSerializer.Meta.read_only_fields + ('predictions',)


class QSARModelInitSerializer(QSARModelSerializer):
    trainingStrategy = QSARTrainingStrategyInitSerializer(many=False)
    molset = serializers.PrimaryKeyRelatedField(many=False, queryset=models.MolSet.objects.all(), required=False)
    predictionsType = serializers.CharField(required=False, max_length=128, allow_null=False)
    predictionsUnits = serializers.CharField(required=False, max_length=128, allow_null=True)

    class Meta:
        model = models.QSARModel
        fields = [x for x in QSARModelSerializer.Meta.fields if x not in ('predictions',)]
        read_only_fields = QSARModelSerializer.Meta.read_only_fields

    def is_valid(self, raise_exception=True):
        initial_data = self.initial_data
        if "trainingStrategy" in initial_data and "embeddings" in initial_data["trainingStrategy"]:
            embeddings = []
            for emb in initial_data["trainingStrategy"]["embeddings"]:
                if isinstance(emb, dict):
                    eid, _ = EmbeddingCalculator.objects.get_or_create(**emb)
                    embeddings.append(eid.id)
                else:
                    embeddings.append(emb)
            initial_data["trainingStrategy"]["embeddings"] = embeddings

        if "trainingStrategy" in initial_data and "validationStrategies" in initial_data["trainingStrategy"]:
            validation_strategies = initial_data["trainingStrategy"]["validationStrategies"]
            for vs in validation_strategies:
                ds = vs["dataSplit"]
                if isinstance(ds, dict):
                    name = ds.pop("name")
                    try:
                        cls = getattr(models, name)
                    except AttributeError:
                        try:
                            cls = getattr(genui_models, name)
                        except AttributeError:
                            raise serializers.ValidationError(f"Data split {name} not found.")
                    if "scaffold" in ds:
                        sid, _ = models.ScaffoldCalculator.objects.get_or_create(**ds["scaffold"])
                        ds["scaffold"] = sid
                    did, _ = cls.objects.get_or_create(**ds)
                    vs["dataSplit"] = did.id
                else:
                    vs["dataSplit"] = ds
            self.initial_data["trainingStrategy"]["validationStrategies"] = validation_strategies

        ret = super().is_valid(raise_exception=raise_exception)
        data = self.validated_data
        tr_strat_data = data['trainingStrategy']

        if data['build'] and tr_strat_data['mode'].name == "classification" and (
                'activityThreshold' not in tr_strat_data or tr_strat_data['activityThreshold'] is None):
            raise serializers.ValidationError("You must specify an activity threshold for a classification model.")

        if data['build'] and ('activityType' not in tr_strat_data or tr_strat_data['activityType'] is None):
            raise serializers.ValidationError(
                "You have to specify the activity type of the training data. Use the 'activityType' parameter in 'trainingStrategy'.")

        if data['build'] and ('activitySet' not in tr_strat_data or tr_strat_data['activitySet'] is None):
            raise serializers.ValidationError(
                "You have to specify the activity set that contains the true activities for training. Use the 'activitySet' parameter in 'trainingStrategy'.")

        if not data["build"] and (
                "predictionsType" not in data or "predictionsUnits" not in data or not data["predictionsType"]):
            raise serializers.ValidationError(
                "You have to specify the type and units of the predicted values if you are not building the model from existing data. Both 'predictionsType' and 'predictionsUnits' must be specified. You can set 'predictionsUnits' to 'null' if the model output variable has no dimension.")

        if tr_strat_data["algorithm"].name == 'QSPRPredScikitModel' and "parameters" in tr_strat_data.keys():
            params = tr_strat_data['parameters']
            alg = params['alg']
            if "Classifier" in alg and tr_strat_data['mode'].name == "regression":
                raise serializers.ValidationError("You cannot use a classifier algorithm for a regression model.")
            if "Regressor" in alg and tr_strat_data['mode'].name == "classification":
                raise serializers.ValidationError("You cannot use a regressor algorithm for a classification model.")
            parameters = json.loads(params['parameters'])
            alg_parameters = get_default_params(None, SKLEARN_MODELS[alg])
            for param in parameters:
                if not param in alg_parameters:
                    raise serializers.ValidationError(
                        f"Parameter {param} is not valid for the selected algorithm {alg}.")

        if ("hyperParamOptStrategies" in tr_strat_data and
                len(tr_strat_data["hyperParamOptStrategies"]) > 0
                and "validationStrategies" in tr_strat_data
                and len(tr_strat_data["validationStrategies"]) > 1):
            raise serializers.ValidationError(
                "You cannot use more than one validation strategy with hyperparameter optimization.")
        return ret

    def create(self, validated_data, **kwargs):
        validation_strategies_data = validated_data['trainingStrategy'].pop('validationStrategies', [])
        hypo_data = validated_data['trainingStrategy'].pop('hyperParamOptStrategies', None)
        instance = super().create(
            validated_data
            , molset=validated_data['molset'] if 'molset' in validated_data else None
            , **kwargs
        )

        strat_data = validated_data['trainingStrategy']
        trainingStrategy = models.QSARTrainingStrategy(
            modelInstance=instance,
            algorithm=strat_data['algorithm'],
            mode=strat_data['mode'],
            activityThreshold=strat_data['activityThreshold'] if 'activityThreshold' in strat_data else None,
            activitySet=strat_data['activitySet'] if 'activitySet' in strat_data else None,
            activityType=strat_data['activityType'] if 'activityType' in strat_data else None
        )
        trainingStrategy.save()
        if 'embeddings' in strat_data and strat_data['embeddings']:
            trainingStrategy.embeddings.set(strat_data['embeddings'])
        trainingStrategy.save()

        self.saveParameters(trainingStrategy, strat_data)

        for vs_data in validation_strategies_data:
            validationStrategy = BasicValidationStrategy.objects.create(
                trainingStrategy=trainingStrategy,
                cvFolds=vs_data['cvFolds'],
                dataSplit=vs_data['dataSplit'],
            )
            validationStrategy.metrics.set(vs_data['metrics'])
            validationStrategy.save()

        if hypo_data:
            hypo_data = hypo_data[0]
            hyperparam_opt_strategy_class = getattr(genui_models, hypo_data["resourcetype"])
            hypo_data.pop("resourcetype")
            hyperparam_opt_strategy = hyperparam_opt_strategy_class.objects.create(
                trainingStrategy=trainingStrategy,
                **hypo_data
            )
            hyperparam_opt_strategy.save()

        if "predictionsType" in validated_data:
            instance.predictionsType = ActivityTypes.objects.get_or_create(
                value=validated_data["predictionsType"]
            )[0]

        if "predictionsUnits" in validated_data and validated_data["predictionsUnits"]:
            instance.predictionsUnits = ActivityUnits.objects.get_or_create(
                value=validated_data["predictionsUnits"]
            )[0]

        instance.save()
        return instance


class ModelActivitySetSerializer(ActivitySetSerializer):
    model = serializers.PrimaryKeyRelatedField(many=False, queryset=models.QSARModel.objects.all())
    taskID = serializers.UUIDField(read_only=True, required=False)

    class Meta:
        model = models.ModelActivitySet
        fields = ActivitySetSerializer.Meta.fields + ('model', 'taskID')
        read_only_fields = ActivitySetSerializer.Meta.read_only_fields + ('taskID', 'model', 'project')


class TemporalSplitSerializer(DataSplitSerializer):
    class Meta:
        model = models.TemporalSplit
        fields = DataSplitSerializer.Meta.fields + ('timeSplit', 'timeProp')

    def validate_timeSplit(self, value):
        if value < 0.0 or value > 1.0:
            raise serializers.ValidationError("Time split must be between 0 and 1.")
        return value


class GBMTDataSplitSerializer(DataSplitSerializer):
    clustering = serializers.PrimaryKeyRelatedField(required=False, queryset=models.MoleculeClusters.objects.all())
    testFraction = serializers.FloatField(required=False)  # mutually exclusive with nFolds

    # nFolds = serializers.IntegerField(required=False)

    class Meta:
        model = models.GBMTDataSplit
        fields = DataSplitSerializer.Meta.fields + ('clustering', 'testFraction')

    def validate_testFraction(self, value):
        if value < 0.0 or value > 1.0:
            raise serializers.ValidationError("Test fraction must be between 0 and 1.")
        return value


class GBMTRandomSplitSerializer(GBMTDataSplitSerializer):
    seed = serializers.IntegerField(required=False, default=42)
    nInitialClusters = serializers.IntegerField(required=False, min_value=10)

    class Meta:
        model = models.GBMTRandomSplit
        fields = GBMTDataSplitSerializer.Meta.fields + ('seed', 'nInitialClusters')

    def validate_seed(self, value):
        if value < 0 or value > 2**32 - 1:
            raise serializers.ValidationError("Seed must be between 0 and 2**32 - 1.")
        return value


class ScaffoldSplitSerializer(GBMTDataSplitSerializer):
    scaffold = serializers.PrimaryKeyRelatedField(many=False, queryset=models.ScaffoldCalculator.objects.all())

    class Meta:
        model = models.ScaffoldSplit
        fields = GBMTDataSplitSerializer.Meta.fields + ('scaffold',)


class ClusterSerializer(GBMTDataSplitSerializer):
    seed = serializers.IntegerField(required=False, default=42)

    class Meta:
        model = models.ClusterSplit
        fields = GBMTDataSplitSerializer.Meta.fields + ('seed',)

    def validate_seed(self, value):
        if value < 0 or value > 2**32 - 1:
            raise serializers.ValidationError("Seed must be between 0 and 2**32 - 1.")
        return value


class MoleculeClustersSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = models.MoleculeClusters
        fields = ('id',)


class RandomClustersSerializer(MoleculeClustersSerializer):
    seed = serializers.IntegerField(required=False, default=42)
    nClusters = serializers.IntegerField(required=False, min_value=10)

    class Meta:
        model = models.RandomClusters
        fields = MoleculeClustersSerializer.Meta.fields + ('seed', 'nClusters')

    def validate_seed(self, value):
        if value < 0 or value > 2 ** 32 - 1:
            raise serializers.ValidationError("Seed must be between 0 and 2**32 - 1.")
        return value


class ScaffoldClustersSerializer(MoleculeClustersSerializer):
    scaffold = serializers.PrimaryKeyRelatedField(required=True, many=False,
                                                  queryset=models.ScaffoldCalculator.objects.all())

    class Meta:
        model = models.ScaffoldClusters
        fields = MoleculeClustersSerializer.Meta.fields + ('scaffold',)


class FPSimilarityClustersSerializer(MoleculeClustersSerializer):
    FPCalculator = serializers.PrimaryKeyRelatedField(required=True, many=False,
                                                      queryset=models.EmbeddingCalculator.objects.all())

    class Meta:
        model = models.FPSimilarityClusters
        fields = MoleculeClustersSerializer.Meta.fields + ('FPCalculator',)


class FPSimilarityMaxMinClustersSerializer(FPSimilarityClustersSerializer):
    seed = serializers.IntegerField(required=False)
    nClusters = serializers.IntegerField(required=False, min_value=10)

    class Meta:
        model = models.FPSimilarityMaxMinClusters
        fields = FPSimilarityClustersSerializer.Meta.fields + ('seed', 'nClusters')

    def validate_seed(self, value):
        if value < 0 or value > 2**32 - 1:
            raise serializers.ValidationError("Seed must be between 0 and 2**32 - 1.")
        return value

class FPSimilarityLeaderPickerClustersSerializer(FPSimilarityClustersSerializer):
    similarityThreshold = serializers.FloatField(required=True)

    class Meta:
        model = models.FPSimilarityLeaderPickerClusters
        fields = FPSimilarityClustersSerializer.Meta.fields + ('similarityThreshold',)

    def validate_similarityThreshold(self, value):
        if value < 0.0 or value > 1.0:
            raise serializers.ValidationError("Similarity threshold must be between 0 and 1.")
        return value


class QSPRPredSklearnModelSerializer(serializers.Serializer):
    name = serializers.CharField()
    type = serializers.CharField()
    params = serializers.DictField(child=serializers.CharField(), required=False)
