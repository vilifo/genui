from pandas import DataFrame, Series
from qsprpred.models import SklearnModel
import importlib

from genui.models.models import ModelParameter
from genui.models.genuimodels.bases import Algorithm, ModelNotFittedException
from genui.utils.inspection import get_sklearn_models

R, C = get_sklearn_models()
MODELS = R | C

class QSPRPredScikitModel(Algorithm): # TODO: testy upload modelu ...
    name = "QSPRPredScikitModel"
    parameters = {
        "base_dir": { # TODO:interni server logika - složka models, metafile + id, zip adresáře mainfile
            "type": ModelParameter.STRING,
            "defaultValue": "models"
        },
        "alg": {
            "type": ModelParameter.STRING,
            "defaultValue": "DummyClassifier"
        },
        "name": {
            "type": ModelParameter.STRING,
            "defaultValue": "DummyName"
        },
        "random_state": {
            "type": ModelParameter.INTEGER,
            "defaultValue": 42
        },
        "parameters": { # TODO: parameter check v serializaci
            "type": ModelParameter.STRING,
            "defaultValue": "{}"
        }
    }

    def __init__(self, builder, callback=None):
        super().__init__(builder, callback)
        self.alg = SklearnModel

    @property
    def model(self):
        return self._model

    def fit(self, X: DataFrame, y: Series):
        alg_instance = self.alg(
            base_dir=self.params['base_dir'],
            alg=self.load_model(self.params['alg']),
            name=self.params['name'],
            # parameters=self.params['parameters'],
        )
        self._model = alg_instance.estimator
        self._model.fit(X, y)
        if self.callback:
            self.callback(self)

    def predict(self, X: DataFrame) -> Series:
        is_regression = self.mode.name == self.REGRESSION
        if self.model:
            if is_regression:
                return self.model.predict(X)
            else:
                return self.model.predict_proba(X)[:, 0]
        else:
            raise ModelNotFittedException("You have to fit the model first.")

    @staticmethod
    def load_model(model_name):
            """Dynamically imports a model."""
            if model_name not in MODELS:
                raise ValueError(f"Model '{model_name}' not found in algorithms dictionary.")
            full_path = MODELS[model_name]
            module_path, class_name = full_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return model_class
