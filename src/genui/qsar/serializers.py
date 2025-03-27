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
from ..utils.inspection import get_model_params


class DescriptorGroupSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = models.DescriptorGroup
        fields = ('id', 'name', 'arguments')

class QSARTrainingStrategySerializer(TrainingStrategySerializer):
    descriptors = DescriptorGroupSerializer(many=True)
    activityType = ActivityTypeSerializer(many=False)
    activitySet = ActivitySetSerializer(many=False)    

    class Meta:
        model = models.QSARTrainingStrategy
        fields = TrainingStrategySerializer.Meta.fields + ('descriptors', 'activityThreshold', 'activityType', 'activitySet')

class QSARTrainingStrategyInitSerializer(TrainingStrategyInitSerializer):
    descriptors = serializers.PrimaryKeyRelatedField(many=True, queryset=models.DescriptorGroup.objects.all(), allow_empty=False)
    activityType = serializers.PrimaryKeyRelatedField(many=False, queryset=ActivityTypes.objects.all(), required=False)
    activitySet = serializers.PrimaryKeyRelatedField(many=False, queryset=ActivitySet.objects.all(), required=False)

    class Meta:
        model = models.QSARTrainingStrategy
        fields = TrainingStrategyInitSerializer.Meta.fields + (
            'descriptors', 'activityThreshold', 'activityType', 'activitySet'
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
        ret = super().is_valid(raise_exception=raise_exception)
        data = self.validated_data
        tr_strat_data = data['trainingStrategy']

        if data['build'] and tr_strat_data['mode'].name == "classification" and ('activityThreshold' not in  tr_strat_data or tr_strat_data['activityThreshold'] is None):
            raise serializers.ValidationError("You must specify an activity threshold for a classification model.")

        if data['build'] and ('activityType' not in  tr_strat_data or tr_strat_data['activityType'] is None):
            raise serializers.ValidationError("You have to specify the activity type of the training data. Use the 'activityType' parameter in 'trainingStrategy'.")

        if data['build'] and ('activitySet' not in  tr_strat_data or tr_strat_data['activitySet'] is None):
            raise serializers.ValidationError("You have to specify the activity set that contains the true activities for training. Use the 'activitySet' parameter in 'trainingStrategy'.")

        if not data["build"] and ("predictionsType" not in data or "predictionsUnits" not in data or not data["predictionsType"]):
            raise serializers.ValidationError("You have to specify the type and units of the predicted values if you are not building the model from existing data. Both 'predictionsType' and 'predictionsUnits' must be specified. You can set 'predictionsUnits' to 'null' if the model output variable has no dimension.")

        if tr_strat_data["algorithm"].name == 'QSPRPredScikitModel' and "parameters" in tr_strat_data.keys():
            params = tr_strat_data['parameters']
            alg = params['alg']
            if "Classifier" in alg and tr_strat_data['mode'].name == "regression":
                raise serializers.ValidationError("You cannot use a classifier algorithm for a regression model.")
            if "Regressor" in alg and tr_strat_data['mode'].name == "classification":
                raise serializers.ValidationError("You cannot use a regressor algorithm for a classification model.")
            parameters = json.loads(params['parameters'])
            alg_parameters = get_model_params(alg)
            for param in parameters:
                if not param in alg_parameters:
                    raise serializers.ValidationError(f"Parameter {param} is not valid for the selected algorithm {alg}.")

        hypo_data = tr_strat_data["hyperParamOptStrategies"]
        val_data = tr_strat_data["validationStrategies"]
        if hypo_data and len(val_data) > 1:
            raise serializers.ValidationError("You cannot use more than one validation strategy with hyperparameter optimization.")
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
            modelInstance = instance,
            algorithm = strat_data['algorithm'],
            mode = strat_data['mode'],
            activityThreshold = strat_data['activityThreshold'] if 'activityThreshold' in strat_data else None,
            activitySet=strat_data['activitySet'] if 'activitySet' in strat_data else None,
            activityType=strat_data['activityType'] if 'activityType' in strat_data else None
        )
        trainingStrategy.save()
        trainingStrategy.descriptors.set(strat_data['descriptors'])
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

class BootstrapSplitSerializer(DataSplitSerializer):
    split = serializers.PrimaryKeyRelatedField(many=False, queryset=models.DataSplit.objects.all())

    class Meta:
        model = models.BootstrapSplit
        fields = DataSplitSerializer.Meta.fields + ('nBootstraps', 'seed', 'split')

class TemporalSplitSerializer(DataSplitSerializer):

    class Meta:
        model = models.TemporalSplit
        fields = DataSplitSerializer.Meta.fields + ('timeSplit', 'timeProp')

class MoleculeClustersSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = models.MoleculeClusters
        fields = ('id',)

class RandomClustersSerializer(MoleculeClustersSerializer):
    IDProp = serializers.CharField(max_length=128, required=False)
    nClusters = serializers.IntegerField(required=False, min_value=1)

    class Meta:
        model = models.RandomClusters
        fields = MoleculeClustersSerializer.Meta.fields + ('seed', 'nClusters', 'IDProp')

class ScaffoldClustersSerializer(MoleculeClustersSerializer):
    IDProp = serializers.CharField(max_length=128, required=False)

    class Meta:
        model = models.ScaffoldClusters
        fields = MoleculeClustersSerializer.Meta.fields + ('scaffold', 'IDProp')

class FPSimilarityClustersSerializer(MoleculeClustersSerializer):
    FPCalculator = serializers.PrimaryKeyRelatedField(many=False, queryset=models.DescriptorGroup.objects.all())
    IDProp = serializers.CharField(max_length=128, required=False)

    class Meta:
        model = models.FPSimilarityClusters
        fields = MoleculeClustersSerializer.Meta.fields + ('FPCalculator', 'IDProp')

class FPSimilarityMaxMinClustersSerializer(FPSimilarityClustersSerializer):
    seed = serializers.IntegerField(required=False)
    nClusters = serializers.IntegerField(required=False, min_value=1)
    # initialCentroids = serializers.PrimaryKeyRelatedField(many=True, queryset=models.Molecule.objects.all(), required=False)

    class Meta:
        model = models.FPSimilarityMaxMinClusters
        fields = FPSimilarityClustersSerializer.Meta.fields + ('seed', 'nClusters', 'initialCentroids')


class FPSimilarityLeaderPickerClustersSerializer(FPSimilarityClustersSerializer):
    similarityThreshold = serializers.FloatField(required=True)

    class Meta:
        model = models.FPSimilarityLeaderPickerClusters
        fields = FPSimilarityClustersSerializer.Meta.fields + ('similarityThreshold',)
        
class QSPRPredSklearnModelSerializer(serializers.Serializer):
    name = serializers.CharField()
    type = serializers.CharField()
    params = serializers.DictField(child=serializers.CharField(), required=False)