"""
metrics

Created by: Martin Sicho
On: 24-01-20, 15:04
"""
from pandas import Series
from sklearn import metrics

from . import bases
from .. import models


# -------------------------- CLASSIFICATION METRICS ------------------------------

class ROC(bases.ValidationMetric):
    name = "ROC"
    description = "The area under ROC curve."
    modes = [bases.Algorithm.CLASSIFICATION]

    def __call__(self, true_vals: Series, predicted_vals: Series):
        fpr, tpr, thrs = self.getCurve(true_vals, predicted_vals)
        return metrics.auc(fpr, tpr)

    def save(
            self,
            true_vals: Series,
            predicted_vals: Series,
            perfClass=models.ModelPerformance,
            **kwargs
    ):
        roc_auc = super().save(true_vals, predicted_vals, perfClass, **kwargs)
        for fpr, tpr, trh in zip(*self.getCurve(true_vals, predicted_vals)):
            models.ROCCurvePoint.objects.create(
                metric=models.ModelPerformanceMetric.objects.get(name=self.name),
                model=self.builder.instance,
                value=tpr,
                fpr=fpr,
                auc=roc_auc,
            )
        return roc_auc

    def getCurve(self, true_vals, scores):
        scores = scores if not isinstance(scores, list) else scores[0]
        return metrics.roc_curve(true_vals, scores[:, 0], pos_label=0)


class MCC(bases.ValidationMetric):
    name = "MCC"
    description = "Matthew's Correlation Coefficient"
    modes = [bases.Algorithm.CLASSIFICATION]

    def __call__(self, true_vals: Series, predicted_vals: Series):
        return metrics.matthews_corrcoef(true_vals, self.probasToClasses(predicted_vals)[0])


class Accuracy(bases.ValidationMetric):
    name = "Accuracy"
    description = "Accuracy classification score. As implemented in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score."
    modes = [bases.Algorithm.CLASSIFICATION]

    def __call__(self, true_vals: Series, predicted_vals: Series):
        return metrics.accuracy_score(true_vals, self.probasToClasses(predicted_vals)[0])


class F1(bases.ValidationMetric):
    name = "F1"
    description = "F1 score. As implemented in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score."
    modes = [bases.Algorithm.CLASSIFICATION]

    def __call__(self, true_vals: Series, predicted_vals: Series):
        return metrics.f1_score(true_vals, self.probasToClasses(predicted_vals)[0])


class Precision(bases.ValidationMetric):
    name = "Precision"
    description = "Precision classification score. As implemented in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score."
    modes = [bases.Algorithm.CLASSIFICATION]

    def __call__(self, true_vals: Series, predicted_vals: Series):
        return metrics.precision_score(true_vals, self.probasToClasses(predicted_vals)[0])


class Recall(bases.ValidationMetric):
    name = "Recall"
    description = "Recall classification score. As implemented in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score."
    modes = [bases.Algorithm.CLASSIFICATION]

    def __call__(self, true_vals: Series, predicted_vals: Series):
        return metrics.recall_score(true_vals, self.probasToClasses(predicted_vals)[0])


# -------------------------- REGRESSION METRICS ------------------------------
class R2(bases.ValidationMetric):
    name = "R2"
    description = "R^2 (coefficient of determination) regression score function. As implemented in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score."
    modes = [bases.Algorithm.REGRESSION]

    def __call__(self, true_vals: Series, predicted_vals: Series):
        return metrics.r2_score(true_vals, predicted_vals)


class MSE(bases.ValidationMetric):
    name = "MSE"
    description = "Mean squared error regression loss. As implemented in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error."
    modes = [bases.Algorithm.REGRESSION]

    def __call__(self, true_vals: Series, predicted_vals: Series):
        return metrics.mean_squared_error(true_vals, predicted_vals)


class MAE(bases.ValidationMetric):
    name = "MAE"
    description = "Mean absolute error regression loss. As implemented in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error."
    modes = [bases.Algorithm.REGRESSION]

    def __call__(self, true_vals: Series, predicted_vals: Series):
        return metrics.mean_absolute_error(true_vals, predicted_vals)


class RMSE(bases.ValidationMetric):
    name = "RMSE"
    description = "Root mean squared error regression loss. As implemented in scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error."
    modes = [bases.Algorithm.REGRESSION]

    def __call__(self, true_vals: Series, predicted_vals: Series):
        return metrics.root_mean_squared_error(true_vals, predicted_vals)
