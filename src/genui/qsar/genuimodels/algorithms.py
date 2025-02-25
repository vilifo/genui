from pandas import DataFrame, Series
from qsprpred.models import SklearnModel

from genui.models.genuimodels.bases import Algorithm, ModelNotFittedException


class QSPRPredScikitModel(Algorithm):
    name = "QSPRPredScikitModel"
    parameters = {}

    def __init__(self, builder, callback):
        super().__init__(builder, callback)
        self.alg = SklearnModel

    @property
    def model(self):
        return self._model

    def fit(self, X: DataFrame, y: Series):
        self._model = self.alg(**self.params)
        self._model.fit(X, y)
        if self.callback:
            self.callback(self)

    def predict(self, X: DataFrame) -> Series:
        is_regression = self.mode.name == self.REGRESSION
        if self.model:
            if is_regression:
                return self.model.predict(X)
            else:
                return self.model.predict_proba(X)[:, 1]
        else:
            raise ModelNotFittedException("You have to fit the model first.")
