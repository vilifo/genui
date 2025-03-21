import ml2json
from pandas import DataFrame, Series
from qsprpred.models import SklearnModel
import tempfile
import tarfile
import zipfile
import os
import importlib
import json

from genui.models.models import ModelParameter
from genui.models.genuimodels.bases import Algorithm, ModelNotFittedException
from genui.models.models import ModelFileFormat
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
        self.sklearn_class = None

    @property
    def model(self):
        return self._model

    @property
    def model_name(self):
        return f"{self.name}_{self.params["alg"]}"

    def fit(self, X: DataFrame, y: Series):
        if self.sklearn_class is None:
            self.sklearn_class = self.import_sklearn_model(self.params['alg'])
        alg_instance = self.alg(
            base_dir=self.temp_dir.name,
            alg=self.sklearn_class,
            name=self.model_name,
            parameters=json.loads(self.params['parameters']),
        )
        self._model = alg_instance
        self._model.estimator.fit(X.values, y.values)
        if self.callback:
            self.callback(self)

    def predict(self, X: DataFrame) -> Series:
        is_regression = self.mode.name == self.REGRESSION
        if self.model:
            if is_regression:
                return self.model.estimator.predict(X.values)
            else:
                return self.model.estimator.predict_proba(X.values)[:, 0]
        else:
            raise ModelNotFittedException("You have to fit the model first.")


    def load_model(self, path):
        if path.endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir.name)
                base_name = zip_ref.namelist()[0]
        elif path.endswith(('.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar')):
            with tarfile.open(path, 'r:*') as tar_ref:
                tar_ref.extractall(self.temp_dir.name)
                base_name = tar_ref.getmembers()[0].name
        else:
            raise ValueError("Unsupported archive format: {}".format(path))

        base_folder = os.path.join(self.temp_dir.name, base_name)
        self._model = self.alg.fromFile(f"{os.path.join(base_folder, base_name)}_meta.json")
        self.params['alg'] = self.model.estimator.__class__.__name__
        return self._model

    def save_model(self, path):
        self._model.save(True)
        with tarfile.open(path, "w:gz") as tar:
            tar.add(os.path.join(self.temp_dir.name, self.model_name), arcname=self.model_name)

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

    def getDeserializer(self):
        return self.load_model

    def getSerializer(self):
        return self.save_model

    @classmethod
    def getFileFormats(cls, attach_to=None):
        formats = [ModelFileFormat.objects.get_or_create(
                       fileExtension=".tar.gz",
                       description="A tar archive file."
                    )[0],
                   ]
        if attach_to:
            cls.attachToInstance(attach_to, formats, attach_to.fileFormats)
        return formats