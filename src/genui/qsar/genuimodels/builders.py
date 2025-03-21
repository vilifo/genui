"""
builders

Created by: Martin Sicho
On: 15-01-20, 12:55
"""
import traceback
import inspect

import numpy as np
import re
from abc import ABC
import pandas as pd
from qsprpred.data.descriptors.sets import DataFrameDescriptorSet
from rdkit import Chem
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, StratifiedKFold

from django.core.exceptions import ImproperlyConfigured
from django.core.files.base import ContentFile

from genui.models.models import ModelFile
from genui.compounds.models import ActivityTypes, ActivitySet
from genui.models.genuimodels.bases import Algorithm, PredictionMixIn, ValidationMixIn, ProgressMixIn, ModelBuilder
from genui.models import models as core_models
from genui.qsar import models as qsar_models
from .bases import DescriptorBuilderMixIn

from qsprpred.data.sampling import splits
from qsprpred.data import QSPRDataset

def camel_to_snake(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

def snake_to_camel(name):
    words = name.split('_')
    return words[0] + ''.join(word.capitalize() for word in words[1:])


class BasicQSARModelBuilder(DescriptorBuilderMixIn, PredictionMixIn, ValidationMixIn, ProgressMixIn, ModelBuilder, ABC):
    def __init__(self, instance: qsar_models.Model, progress=None, onFitCall=None, validations=None):
        super().__init__(instance, progress, onFitCall)
        self.validations = validations if validations and len(validations) > 0 else self.instance.trainingStrategy.validationStrategies.all()

    def build(self) -> qsar_models.QSARModel:
        if not self.validations:
            raise ImproperlyConfigured("You cannot build a QSAR model without validation strategies.")

        if not self.molsets:
            raise ImproperlyConfigured("You cannot build a QSAR model without an associated molecule set.")

        self.progressStages = [
            "Fetching activities...",
            "Calculating descriptors..."
        ]

        for validation in self.validations:
            self.progressStages.extend([f"CV fold {x+1}" for x in range(validation.cvFolds)])
        self.progressStages.extend(["Fitting model on the training set...", "Validating on test set..."])
        self.progressStages.extend(["Fitting the final model...", "Saving the model..."])

        self.recordProgress()
        mols = self.saveActivities()[1]

        self.recordProgress()
        self.calculateDescriptors(mols)
        mols = pd.DataFrame([x.canonicalSMILES for x in mols], columns=["SMILES"])
        X = self.X
        y = pd.DataFrame(self.y, columns=["activity"])
        if self.training.mode.name == Algorithm.CLASSIFICATION:
            target_props = {"name": "activity", "task": "SINGLECLASS", "th": [self.training.activityThreshold],}
        else:
            target_props = {"name": "activity", "task": "REGRESSION"}
        dataset = QSPRDataset("Dataset", [target_props],
                              df=pd.concat([mols, y], axis=1))
        dataset.prepareDataset(feature_calculators=[DataFrameDescriptorSet(df=pd.concat([X, mols], axis=1), joining_cols=["SMILES"])])

        for validation in self.validations:
            if hasattr(validation, 'dataSplit') and hasattr(validation, 'cvFolds'):
                split_name = validation.dataSplit.__class__.__name__
                split_instance = getattr(splits, split_name)
                split_params = {param: getattr(validation.dataSplit, snake_to_camel(param)) for param in
                                inspect.signature(split_instance.__init__).parameters if
                                hasattr(validation.dataSplit, snake_to_camel(param))}
                split_instance = split_instance(**split_params)

                dataset.split(split_instance)
                X_train, y_train = dataset.X, dataset.y["activity"]
                X_valid, y_valid = dataset.X_ind, dataset.y_ind["activity"]

                # Perform cross-validation
                is_regression = self.training.mode.name == Algorithm.REGRESSION
                if is_regression:
                    folds = KFold(validation.cvFolds).split(X_train)
                else:
                    folds = StratifiedKFold(validation.cvFolds).split(X_train, y_train)

                for i, (train_index, test_index) in enumerate(folds):
                    self.recordProgress()
                    self.fitAndValidate(
                        X_train.iloc[train_index], y_train.iloc[train_index],
                        X_train.iloc[test_index], y_train.iloc[test_index],
                        perfClass=core_models.ModelPerformanceCV, 
                        fold=i + 1
                    )

                self.fitAndValidate(X_train, y_train, X_valid, y_valid)

        # Final model fitting on all data
        final_model = self.algorithmClass(self)
        final_model.fit(dataset.X, dataset.y["activity"])
        self._model = final_model

        # Final validation (optional, as it's not truly a validation)
        y_predicted = final_model.predict(dataset.X)
        for validation in self.validations:
            self.validate(validation, dataset.y["activity"], y_predicted)

        self.recordProgress()
        # self.save_model()
        self.saveFile()
        return self.instance

    def save_model(self): # Backup, for future endeavours
        file = ModelFile.create(
            self.instance,
            f'{self.model.model_name}.tar.gz',
            ContentFile('placeholder'),
            kind=ModelFile.AUXILIARY,
            note=f'{self.model.model_name}'
        )
        path = file.path
        self.model.save_model(path)
        self.instance.save()

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

        self.calculateDescriptors(smiles)
        real_predictions = list(self.predict(self.getX()))
        for idx,prediction in enumerate(predictions):
            if idx not in failed_indices:
                predictions[idx] = real_predictions.pop(0)
        assert len(real_predictions) == 0
        return np.array(predictions)
    
    def populateActivitySet(self, aset : qsar_models.ModelActivitySet):
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
    
    def fitAndValidate(self, X_train, y_train, X_valid, y_valid, y_predicted=None, perfClass=core_models.ModelPerformance, *args, **kwargs):
        if not y_predicted:
            model = self.algorithmClass(self)
            model.fit(X_train, y_train)
            y_predicted = model.predict(X_valid)
        for validation in self.validations:
            self.validate(validation, y_valid, y_predicted, perfClass, *args, **kwargs)
            
    def validate(self, validation_strategy, y_validated, y_predicted, perfClass=core_models.ModelPerformance, *args, **kwargs):
        if not validation_strategy:
            raise ImproperlyConfigured(f"No validation strategy is set for model: {repr(self.instance)}")
        for metric_class in self.metricClasses:
            try:
                metric_class(self).save(y_validated, y_predicted, perfClass, *args, **kwargs)
            except Exception as exp:
                print("Failed to obtain values for metric: ", metric_class.name)
                self.errors.append(exp)
                traceback.print_exc()
                continue

    def getX(self) -> DataFrame:
        return self.X

    def getY(self) -> Series:
        return self.y

    def saveActivities(self):
        if not self.getY():
            activity_set = ActivitySet.objects.get(pk=self.training.activitySet.id)
            activity_type = self.training.activityType
            if not activity_set:
                raise Exception("No activity set specified.")
            if not activity_type:
                raise Exception("No activity type specified.")

            compounds, activities, units = activity_set.cleanForModelling(activity_type)
            if not len(compounds) == len(activities):
                raise Exception(f'Number of compounds in a QSAR model ({len(compounds)}) is different from the set of activities assigned to them ({len(activities)}). Something went wrong when the data was cleaned for modeling.')
            activities = Series(activities)

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
            self.y = activities
            return self.y, compounds

    def load_model(self): # Backup, for future endeavours
        self._model = self.model.load_model(self.instance.files.filter(note=self.model.model_name)[0].path)
