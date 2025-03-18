import ml2json
from pandas import DataFrame, Series
from qsprpred.models import SklearnModel
import tempfile
import tarfile
import os
import importlib
import json

from genui.models.models import ModelParameter
from genui.models.genuimodels.bases import Algorithm, ModelNotFittedException
from genui.utils.inspection import SKLEARN_MODELS

class QSPRPredScikitModel(Algorithm): # TODO: testy upload modelu ...
    name = "QSPRPredScikitModel"
    parameters = {
        "alg": {
            "type": ModelParameter.STRING,
            "defaultValue": "DummyClassifier"
        },
        "parameters": {
            "type": ModelParameter.STRING,
            "defaultValue": "{}"
        }
    }

    def __init__(self, builder, callback=None):
        super().__init__(builder, callback)
        self.alg = SklearnModel
        self.temp_dir = tempfile.TemporaryDirectory()

    @property
    def model(self):
        return self._model

    @property
    def model_name(self):
        return f"{self.name}_{self.params["alg"]}"

    def fit(self, X: DataFrame, y: Series):
        alg_instance = self.alg(
            base_dir=self.temp_dir.name,
            alg=self.import_sklearn_model(self.params['alg']),
            name=self.model_name,
            parameters=json.loads(self.params['parameters']),
        )
        self._model = alg_instance
        self._model.estimator.fit(X, y)
        if self.callback:
            self.callback(self)

    def predict(self, X: DataFrame) -> Series:
        is_regression = self.mode.name == self.REGRESSION
        if self.model:
            if is_regression:
                return self.model.estimator.predict(X)
            else:
                return self.model.estimator.predict_proba(X)[:, 0]
        else:
            raise ModelNotFittedException("You have to fit the model first.")


    def load_model(self, path):
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(self.temp_dir.name)
        base_folder = os.path.join(self.temp_dir.name, self.model_name)
        self._model = self._model.fromFile(f"{os.path.join(base_folder, self.model_name)}_meta.json")
        self.model.estimator = ml2json.from_json(f"{os.path.join(base_folder, self.model_name)}.json")

    def save_model(self, path):
        self._model.save(True)
        with tarfile.open(path, "w:gz") as tar:
            tar.add(self.temp_dir.name, arcname=self.model_name)

    @staticmethod
    def import_sklearn_model(model_name):
            """Dynamically imports a model."""
            if model_name not in SKLEARN_MODELS:
                raise ValueError(f"Model '{model_name}' not found in algorithms dictionary.")
            full_path = SKLEARN_MODELS[model_name]
            module_path, class_name = full_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return model_class
