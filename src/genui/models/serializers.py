"""
serializers

Created by: Martin Sicho
On: 24-01-20, 14:44
"""
from rest_framework import serializers
from rest_polymorphic.serializers import PolymorphicSerializer

from genui.utils.serializers import GenericModelSerializerMixIn
from genui.models.models import (ModelFileFormat, ModelBuilder, Model, PARAM_VALUE_CTYPE_TO_MODEL_MAP, ModelParameter, \
                                 Algorithm, TrainingStrategy, ModelFile, BasicValidationStrategy,
                                 ModelPerformanceMetric, ValidationStrategy, \
                                 AlgorithmMode, ModelParameterValue, ModelPerformance, DataSplit, RandomSplit,
                                 ValueAggregationFunction,
                                 HyperparameterOptimizationStrategy, GridSearchOptimization, OptunaOptimization)
from genui.models import models
from genui.projects.models import Project


class ModelFileFormatSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ModelFileFormat
        fields = ('id', 'fileExtension', 'description')


class ModelParameterSerializer(serializers.HyperlinkedModelSerializer):
    defaultValue = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = ModelParameter
        fields = ('id', 'name', 'contentType', 'defaultValue')
        read_only_fields = ('defaultValue',)

    def get_defaultValue(self, obj):
        return obj.defaultValue.value


class AlgorithmModeSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = AlgorithmMode
        fields = ('id', 'name',)


class AlgorithmSerializer(serializers.HyperlinkedModelSerializer):
    fileFormats = ModelFileFormatSerializer(many=True)
    parameters = ModelParameterSerializer(many=True)
    validModes = AlgorithmModeSerializer(many=True)

    class Meta:
        model = Algorithm
        fields = ('id', 'name', 'fileFormats', 'parameters', 'validModes')


class ModelPerformanceMetricSerializer(serializers.HyperlinkedModelSerializer):
    validAlgorithms = serializers.PrimaryKeyRelatedField(many=True, queryset=Algorithm.objects.all())
    validModes = AlgorithmModeSerializer(many=True)

    class Meta:
        model = ModelPerformanceMetric
        fields = ('id', 'name', 'description', 'validAlgorithms', 'validModes')


class ModelPerformanceSerializer(GenericModelSerializerMixIn, serializers.HyperlinkedModelSerializer):
    className = GenericModelSerializerMixIn.className
    extraArgs = GenericModelSerializerMixIn.extraArgs
    metric = ModelPerformanceMetricSerializer(many=False)

    class Meta:
        model = ModelPerformance
        fields = ('id', 'value', 'metric', 'className', 'extraArgs')


class ModelParameterValueSerializer(serializers.HyperlinkedModelSerializer):
    parameter = ModelParameterSerializer(many=False)
    value = serializers.CharField()

    class Meta:
        model = ModelParameterValue
        fields = ('id', 'parameter', 'value')


class ValueAggregationFunctionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ValueAggregationFunction
        fields = ('id', 'name', 'description')


class HyperparameterOptimizationStrategySerializer(serializers.HyperlinkedModelSerializer):
    scoreAggregation = ValueAggregationFunctionSerializer(many=False)
    searchSpace = serializers.JSONField()

    class Meta:
        model = HyperparameterOptimizationStrategy
        fields = ('searchSpace', 'scoreAggregation', 'trainingStrategy')


class HyperparameterOptimizationStrategyInitSerializer(HyperparameterOptimizationStrategySerializer):
    scoreAggregation = serializers.PrimaryKeyRelatedField(many=False, queryset=ValueAggregationFunction.objects.all())
    searchSpace = serializers.JSONField()

    class Meta:
        model = HyperparameterOptimizationStrategy
        fields = tuple(x for x in HyperparameterOptimizationStrategySerializer.Meta.fields if x != 'trainingStrategy')


class GridSearchStrategyInitSerializer(HyperparameterOptimizationStrategyInitSerializer):
    class Meta:
        model = GridSearchOptimization
        fields = HyperparameterOptimizationStrategyInitSerializer.Meta.fields


class GridSearchStrategySerializer(GridSearchStrategyInitSerializer):
    class Meta:
        model = GridSearchOptimization
        fields = GridSearchStrategyInitSerializer.Meta.fields


class OptunaStrategyInitSerializer(HyperparameterOptimizationStrategyInitSerializer):
    nTrials = serializers.IntegerField()

    class Meta:
        model = OptunaOptimization
        fields = HyperparameterOptimizationStrategyInitSerializer.Meta.fields + ('nTrials',)


class OptunaStrategySerializer(OptunaStrategyInitSerializer):
    nTrials = serializers.IntegerField(min_value=1)

    class Meta:
        model = OptunaOptimization
        fields = OptunaStrategyInitSerializer.Meta.fields


class HyperparameterOptimizationStrategyPolymorphicSerializer(PolymorphicSerializer):
    model_serializer_mapping = {
        GridSearchOptimization: GridSearchStrategySerializer,
        OptunaOptimization: OptunaStrategySerializer,
    }


class HyperparameterOptimizationStrategyPolymorphicInitSerializer(PolymorphicSerializer):
    model_serializer_mapping = {
        HyperparameterOptimizationStrategy: HyperparameterOptimizationStrategyInitSerializer,
        GridSearchOptimization: GridSearchStrategyInitSerializer,
        OptunaOptimization: OptunaStrategyInitSerializer,
    }


class ValidationStrategySerializer(serializers.HyperlinkedModelSerializer):
    metrics = ModelPerformanceMetricSerializer(many=True)

    class Meta:
        model = ValidationStrategy
        fields = ("metrics", "trainingStrategy")


class ValidationStrategyInitSerializer(ValidationStrategySerializer):
    metrics = serializers.PrimaryKeyRelatedField(many=True, queryset=ModelPerformanceMetric.objects.all())

    class Meta:
        model = ValidationStrategy
        fields = tuple(x for x in ValidationStrategySerializer.Meta.fields if x != 'trainingStrategy')


class BasicValidationStrategyInitSerializer(ValidationStrategyInitSerializer):
    metrics = serializers.PrimaryKeyRelatedField(many=True, queryset=ModelPerformanceMetric.objects.all())
    cvFolds = serializers.IntegerField(min_value=0)
    dataSplit = serializers.PrimaryKeyRelatedField(many=False, queryset=DataSplit.objects.all())

    # TODO: check if correct metrics are used with the correct algorithm

    class Meta:
        model = BasicValidationStrategy
        fields = ValidationStrategyInitSerializer.Meta.fields + ('cvFolds', 'dataSplit')


class BasicValidationStrategySerializer(BasicValidationStrategyInitSerializer):
    metrics = ModelPerformanceMetricSerializer(many=True)


class ValidationStrategyPolymorphicSerializer(PolymorphicSerializer):
    model_serializer_mapping = {
        BasicValidationStrategy: BasicValidationStrategySerializer,
        # Add other subclasses and their serializers here
    }


class ValidationStrategyPolymorphicInitSerializer(PolymorphicSerializer):
    model_serializer_mapping = {
        ValidationStrategy: ValidationStrategyInitSerializer,
        BasicValidationStrategy: BasicValidationStrategyInitSerializer,
    }


class TrainingStrategySerializer(serializers.HyperlinkedModelSerializer):
    algorithm = AlgorithmSerializer(many=False)
    parameters = ModelParameterValueSerializer(many=True)
    mode = AlgorithmModeSerializer(many=False)
    validationStrategies = ValidationStrategyPolymorphicSerializer(many=True, read_only=True)
    hyperParamOptStrategies = HyperparameterOptimizationStrategyPolymorphicSerializer(many=True, read_only=True)

    class Meta:
        model = TrainingStrategy
        fields = ('id', 'algorithm', 'mode', 'parameters', 'validationStrategies', 'hyperParamOptStrategies')


class TrainingStrategyInitSerializer(TrainingStrategySerializer):
    algorithm = serializers.PrimaryKeyRelatedField(many=False, queryset=Algorithm.objects.all())
    parameters = serializers.DictField(allow_empty=True, child=serializers.CharField(), required=False)
    mode = serializers.PrimaryKeyRelatedField(many=False, queryset=AlgorithmMode.objects.all())
    validationStrategies = ValidationStrategyPolymorphicInitSerializer(many=True, required=False)
    hyperParamOptStrategies = HyperparameterOptimizationStrategyPolymorphicInitSerializer(many=True, required=False)

    class Meta:
        model = TrainingStrategy
        fields = TrainingStrategySerializer.Meta.fields

    def create(self, validated_data, **kwargs):
        instance = super().create(validated_data, **kwargs)

        if 'validationStrategies' in validated_data:
            strat_data = validated_data['validationStrategies']
            for vs_data in strat_data:
                validationStrategy = BasicValidationStrategy.objects.create(
                    trainingStrategy=instance,
                    cvFolds=vs_data['cvFolds'],
                    dataSplit=vs_data['dataSplit'],
                )
                validationStrategy.metrics.set(vs_data['metrics'])
                validationStrategy.save()


        if 'hyperParamOptStrategies' in validated_data:
            hypo_data = validated_data['hyperParamOptStrategies'][0]
            hyperparam_opt_strategy_class = getattr(models, hypo_data["resourcetype"])
            hypo_data.pop("resourcetype")
            hyperparam_opt_strategy = hyperparam_opt_strategy_class.objects.create(
                trainingStrategy=instance,
                **hypo_data
            )
            hyperparam_opt_strategy.save()

        return instance


class ModelFileSerializer(serializers.HyperlinkedModelSerializer):
    model = serializers.PrimaryKeyRelatedField(many=False, required=False, queryset=Model.objects.all())
    format = ModelFileFormatSerializer(many=False, read_only=True)

    class Meta:
        model = ModelFile
        fields = ('id', 'file', 'format', 'kind', 'model', 'note')
        read_only_fields = ('id', 'format',)

    def create(self, validated_data):
        return ModelFile.create(
            validated_data['model'],
            validated_data['file'].name,
            validated_data['file'],
            validated_data['kind'] if 'kind' in validated_data else ModelFile.AUXILIARY,
            validated_data['note'] if 'note' in validated_data else ''
        )


class ModelSerializer(serializers.HyperlinkedModelSerializer):
    project = serializers.PrimaryKeyRelatedField(many=False, queryset=Project.objects.all())
    trainingStrategy = TrainingStrategySerializer(many=False)
    build = serializers.BooleanField(default=True)
    taskID = serializers.UUIDField(required=False, read_only=True, allow_null=True)
    modelFile = ModelFileSerializer(many=False, read_only=True, allow_null=True, required=False)

    def __init__(self, *args, builder_class=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.builder_class = self.instance.builder.name if self.instance and isinstance(self.instance,
                                                                                        Model) else builder_class

    def is_valid(self, *, raise_exception=False):
        ret = super().is_valid(raise_exception=raise_exception)
        if "hyperParamOptStrategies" in self.validated_data["trainingStrategy"]:
            val_data = self.validated_data["trainingStrategy"]["validationStrategies"]
            if len(val_data) > 1:
                raise serializers.ValidationError(
                    "Only one validation strategy is allowed when using hyperparameter optimization.")
        return ret

    # def validate(self, attrs):
    #     if "modelFile" in attrs and "validationStrategy" in attrs:
    #         raise ValidationError("If 'modelFile' is present, 'validationStrategy' field should be empty.")
    #     if "modelFile" not in attrs and "validationStrategy" not in attrs:
    #         raise ValidationError("You have to specify 'modelFile' if you omit 'validationStrategy'.")
    #     return super().validate(attrs)

    @staticmethod
    def saveParameters(strat_instance: TrainingStrategy, strat_data):
        if 'parameters' not in strat_data:
            return

        alg_name = strat_data['algorithm'].name
        alg = Algorithm.objects.get(name=alg_name)

        for param in alg.parameters.all():
            if param.name not in strat_data['parameters']:
                strat_data['parameters'][param.name] = str(param.defaultValue.value)

        for param_name in strat_data['parameters']:
            parameter = ModelParameter.objects.get(
                name=param_name
                , algorithm__name=alg_name
            )
            value_class = PARAM_VALUE_CTYPE_TO_MODEL_MAP[parameter.contentType]
            parameter_value = value_class(
                parameter=parameter
                , strategy=strat_instance
                , value=value_class.parseValue(strat_data['parameters'][param_name]))
            parameter_value.save()

    class Meta:
        model = Model
        fields = (
            'id', 'name', 'description', 'created', 'updated', 'project', 'trainingStrategy', 'modelFile', 'build',
            'taskID')
        read_only_fields = ('id', 'created', 'updated', 'modelFile', 'taskID')

    def useBuilder(self, builder_class):
        self.builder_class = builder_class

    def create(self, validated_data, **kwargs):
        instance = self.Meta.model.objects.create(
            name=validated_data['name'],
            description=validated_data['description'] if 'description' in validated_data else '',
            project=validated_data['project'],
            builder=ModelBuilder.objects.get_or_create(
                name=self.builder_class if type(self.builder_class) == str else self.builder_class.__name__
            )[0],
            **kwargs
        )
        instance.build = validated_data["build"]
        return instance


class DataSplitSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = DataSplit
        fields = ('id',)


class RandomSplitSerializer(DataSplitSerializer):
    class Meta:
        model = RandomSplit
        fields = DataSplitSerializer.Meta.fields + ('testFraction', 'seed')
