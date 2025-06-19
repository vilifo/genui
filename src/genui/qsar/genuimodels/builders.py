"""
builders

Created by: Martin Sicho
On: 15-01-20, 12:55
"""
import json
import importlib
import inspect
import tempfile
import traceback
from abc import ABC

from rdkit import Chem
import numpy as np
import pandas as pd

from qsprpred.data import QSPRDataset
from qsprpred.data.sampling import splits
from qsprpred import models as qsprpred_models
from qsprpred.models import CrossValAssessor, TestSetAssessor
from sklearn.model_selection import KFold, StratifiedKFold

from django.core.exceptions import ImproperlyConfigured
from genui.compounds.models import Molecule
from genui.utils.inspection import snake_to_camel, get_default_params
from .qsprpred_utils import RecordProgressMonitor, MetricsAggregator
from genui.compounds.models import ActivityTypes, ActivitySet
from genui.models import models as core_models
from genui.models.genuimodels.bases import Algorithm, PredictionMixIn, ValidationMixIn, ProgressMixIn, ModelBuilder
from genui.qsar import models as qsar_models
from .bases import EmbeddingBuilderMixIn

clustering_module = importlib.import_module("qsprpred.data.chem.clustering")


class BasicQSARModelBuilder(EmbeddingBuilderMixIn, PredictionMixIn, ValidationMixIn, ProgressMixIn, ModelBuilder, ABC):
    def __init__(self, instance: qsar_models.Model, progress=None, onFitCall=None, validations=None):
        super().__init__(instance, progress, onFitCall)
        self.validations = validations if validations and len(
            validations) > 0 else self.instance.trainingStrategy.validationStrategies.all()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset = None

    def build(self) -> qsar_models.QSARModel:
        if not self.validations:
            raise ImproperlyConfigured("You cannot build a QSAR model without validation strategies.")

        if not self.molsets:
            raise ImproperlyConfigured("You cannot build a QSAR model without an associated molecule set.")

        self.progressStages = [
            "Fetching activities...",
            "Calculating embeddings..."
        ]
        if self.hyper_param_opt:
            self.progressStages.append("Optimizing hyperparameters...")

        for validation in self.validations:
            self.progressStages.extend([f"CV fold {x + 1}" for x in range(validation.cvFolds)])
        self.progressStages.extend(["Fitting model on the training set...", "Validating on test set..."])
        self.progressStages.extend(["Fitting the final model...", "Saving the model..."])

        self.recordProgress()
        dataset = self.saveActivities()

        self.recordProgress()
        dataset.prepareDataset(feature_calculators=self.embeddingCalculators)

        monitor = RecordProgressMonitor(self.recordProgress)
        monitor.on_fold_start = True
        monitor.on_optimization_start = True
        metrics_aggregator = MetricsAggregator(self.metricClasses[0], [m for m in self.metricClasses], self, monitor,
                                               core_models.ModelPerformanceCV)
        self._model = self.algorithmClass(self)

        for validation_strategy_index, validation in enumerate(self.validations):
            monitor.validation_index = validation_strategy_index
            if hasattr(validation, 'dataSplit') and hasattr(validation, 'cvFolds'):
                split_instance = self._init_split(validation)
                dataset.split(split_instance)

                if self.hyper_param_opt:
                    monitor.validation_index = -20
                    optimizer_class = getattr(qsprpred_models, self.hyper_param_opt.__class__.__name__)
                    kwargs = {"param_grid": self.hyper_param_opt.searchSpace,
                              "model_assessor": CrossValAssessor(self.hypo_metric(self)),
                              "score_aggregation": self.hyper_param_aggregator(self)}
                    if "nTrials" in dir(self.hyper_param_opt):
                        kwargs["n_trials"] = self.hyper_param_opt.nTrials
                    optimizer = optimizer_class(**kwargs)
                    params = optimizer.optimize(self.model.model, dataset)
                    old_params = json.loads(self.model.params['parameters'])
                    old_params.update(**params)
                    self.model.params["parameters"] = json.dumps(old_params)
                    monitor.validation_index = validation_strategy_index

                # Perform cross-validation
                is_regression = self.training.mode.name == Algorithm.REGRESSION
                if is_regression:
                    folds = KFold(validation.cvFolds)
                else:
                    folds = StratifiedKFold(validation.cvFolds)
                cva = CrossValAssessor(metrics_aggregator,
                                       folds,
                                       monitor,
                                       use_proba=self.training.mode.name == Algorithm.CLASSIFICATION, )
                cva(self.model.model, dataset, False, )

        monitor.on_assessment_start = True
        monitor.validation_index = -1 # Test set performance
        metrics_aggregator.perfClass = core_models.ModelPerformance
        tsa = TestSetAssessor(metrics_aggregator,
                              monitor,
                              use_proba=self.training.mode.name == Algorithm.CLASSIFICATION, )
        tsa(self.model.model, dataset, False)

        self.recordProgress()
        self.model.model.fitDataset(dataset, save_model=False)

        # Final validation (optional, as it's not truly a validation)
        y_predicted = self.model.predict(dataset.X)
        # for validation in self.validations:
        self.validate(dataset.y, y_predicted, validationIndex=-10)  # Not a validation

        self.recordProgress()
        self.saveFile()
        return self.instance

    def populateActivitySet(self, aset: qsar_models.ModelActivitySet):
        if not self.instance.predictionsType:
            raise Exception("The activity type for QSAR model predictions is not specified.")

        aset.activities.all().delete()
        molecules = aset.molecules.molecules.all()
        predictions = self.predictMolecules(molecules)

        for mol, prediction in zip(molecules, predictions):
            qsar_models.ModelActivity.objects.create(
                value=prediction,
                type=self.instance.predictionsType,
                units=self.instance.predictionsUnits,
                source=aset,
                molecule=mol,
            )

        return aset.activities.all()

    def predictMolecules(self, mols):
        smiles = []
        failed_indices = []
        for idx, mol in enumerate(mols):
            if mol:
                smiles.append(Chem.MolToSmiles(mol) if type(mol) == Chem.Mol else mol)
            else:
                failed_indices.append(idx)

        predictions = [-1] * len(mols)
        if len(failed_indices) == len(mols):
            return np.array(predictions)

        real_predictions = list(self.predictMols(smiles))
        for idx, prediction in enumerate(predictions):
            if idx not in failed_indices:
                predictions[idx] = real_predictions.pop(0)
        assert len(real_predictions) == 0
        return np.array(predictions)

    def _init_split(self, validation):
        split_name = validation.dataSplit.__class__.__name__
        split_instance = getattr(splits, split_name)
        split_params = self._extract_kwargs(validation.dataSplit, split_instance)

        kwargs = get_default_params(split_name, splits.__name__)
        if "dataset" in kwargs:
            split_params["dataset"] = self.dataset
        if "scaffold" in kwargs:
            split_params["scaffold"] = self._init_embedding_calculator(split_params["scaffold"], "scaffolds")
        if "clustering" in kwargs:
            clustering = getattr(clustering_module, split_params["clustering"].__class__.__name__)
            fp_calculator = self._init_embedding_calculator(split_params["clustering"].FPCalculator)
            clustering_kwargs = self._extract_kwargs(split_params["clustering"], clustering)
            clustering_kwargs["fp_calculator"] = fp_calculator
            split_params["clustering"] = clustering(**clustering_kwargs)
        return split_instance(**split_params)

    @staticmethod
    def _extract_kwargs(django_model, qsprpred_class):
        return {param: getattr(django_model, snake_to_camel(param)) for param in
                inspect.signature(qsprpred_class.__init__).parameters if
                hasattr(django_model, snake_to_camel(param))}

    # def fitAndValidate(self, X_train, y_train, X_valid, y_valid, y_predicted=None,
    #                    perfClass=core_models.ModelPerformance, *args, **kwargs):
    #     if not y_predicted:
    #         model = self.algorithmClass(self)
    #         model.fit(X_train, y_train)
    #         y_predicted = model.predict(X_valid)
    #     for validation in self.validations:
    #         self.validate(validation, y_valid, y_predicted, perfClass, *args, **kwargs)

    def validate(self, y_validated, y_predicted, perfClass=core_models.ModelPerformance, validationIndex=-1, *args, **kwargs):
        for metric_class in self.metricClasses:
            try:
                metric_class(self).save(y_validated, y_predicted, validationIndex, perfClass, *args, **kwargs)
            except Exception as exp:
                print("Failed to obtain values for metric: ", metric_class.name)
                self.errors.append(exp)
                traceback.print_exc()
                continue

    def getDataset(self) -> QSPRDataset:
        return self.dataset

    def saveActivities(self):
        if not self.getDataset():
            activity_set = ActivitySet.objects.get(pk=self.training.activitySet.id)
            activity_type = self.training.activityType
            if not activity_set:
                raise Exception("No activity set specified.")
            if not activity_type:
                raise Exception("No activity type specified.")

            compounds, activities, units = activity_set.cleanForModelling(activity_type)
            if not len(compounds) == len(activities):
                raise Exception(
                    f'Number of compounds in a QSAR model ({len(compounds)}) is different from the set of activities assigned to them ({len(activities)}). Something went wrong when the data was cleaned for modeling.')

            if self.training.mode.name == Algorithm.CLASSIFICATION:
                if not self.instance.predictionsType:
                    self.instance.predictionsType = ActivityTypes.objects.get_or_create(
                        value="Active Probability"
                    )[0]

            if not self.instance.predictionsType:
                self.instance.predictionsType = activity_type
            if not self.instance.predictionsUnits:
                self.instance.predictionsUnits = units

            self.instance.save()
            if self.training.mode.name == Algorithm.CLASSIFICATION:
                target_props = {"name": activity_type.value, "task": "SINGLECLASS",
                                "th": [self.training.activityThreshold], }
            elif self.training.mode.name == Algorithm.MULTICLASS:
                target_props = {"name": activity_type.value, "task": "MULTICLASS", "th": self.training.activityThreshold}
            else:
                target_props = {"name": activity_type.value, "task": "REGRESSION"}
            smiles = self.getSmiles(compounds)
            df = pd.DataFrame({"SMILES": smiles, activity_type.value: activities})
            self.dataset = QSPRDataset(self.instance.name, [target_props], df, store_dir=self.temp_dir.name)
            return self.dataset

    def load_model(self):  # Backup, for future endeavours
        self._model = self.model.load_model(self.instance.files.filter(note=self.model.model_name)[0].path)

    def getSmiles(self, mols=None) -> list:
        if mols is not None:
            smiles = [x.canonicalSMILES if isinstance(x, Molecule) else x for x in mols]
        elif hasattr(self, "mols"):
            smiles = [x.canonicalSMILES if isinstance(x, Molecule) else x for x in self.mols]
        elif hasattr(self, "molsets"):
            smiles = []
            for molset in self.molsets:
                for mol in molset.molecules.all():
                    smiles.append(mol.canonicalSMILES)
        else:
            raise Exception("No molecules to calculate embeddings from.")

        return smiles

    def predictMols(self, smiles=None):
        smiles = self.getSmiles(smiles)
        return self.model.predictMols(smiles)
