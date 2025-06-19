from qsprpred.models.monitors import BaseMonitor
from genui.utils.inspection import camel_to_snake, snake_to_camel
import traceback
from genui.models import models
from genui.models.genuimodels.bases import Algorithm
import re


def private_state(state):
    return f"_{state}"


class RecordProgressMonitor(BaseMonitor):
    @property
    def validation_index(self):
        return self._validation_index

    @validation_index.setter
    def validation_index(self, value):
        self._validation_index = value

    def __init__(self, progress_callback=None):
        super().__init__()
        self._validation_index = 0
        states = [camel_to_snake(name) for name in dir(self.__class__) if re.match(r"^on[A-Z]", name)]
        for state in states:
            setattr(self, snake_to_camel(state), self._update_callback_method(state))
            setattr(self, private_state(state), False)
            setattr(self.__class__, state, property(
                fget=lambda instance, s=state: getattr(instance, private_state(s)),
                fset=lambda instance, value, s=state: setattr(instance, private_state(s), value)))
        self.progress_callback = progress_callback if progress_callback else lambda: print("Progress updated")

    def _update_callback_method(self, state):
        func = getattr(self, snake_to_camel(state))

        def method(*args, **kwargs):
            if getattr(self, private_state(state)):
                self.progress_callback()
            return func(*args, **kwargs)

        return method


class MetricsAggregator:
    @property
    def perfClass(self):
        return self._perfClass

    @perfClass.setter
    def perfClass(self, value):
        self._perfClass = value

    def __init__(self, validation_metric, metrics, builder, monitor, perfClass=models.ModelPerformance):
        self.validation_metric = validation_metric
        self.metricClasses = metrics
        self._perfClass = perfClass
        self.monitor = monitor
        self.builder = builder

    def __call__(self, y_true, y_pred):
        kwargs = {}
        if self.perfClass == models.ModelPerformanceCV:
            kwargs["fold"] = self.monitor.currentFold + 1

        for metric_class in self.metricClasses:
            try:
                metric_class(self.builder).save(y_true, y_pred, self.monitor.validation_index,
                                                self.perfClass, **kwargs)
            except Exception as exp:
                print("Failed to obtain values for metric: ", metric_class.name)
                self.builder.errors.append(exp)
                traceback.print_exc()
                continue
        return self.validation_metric(self.builder)(y_true, y_pred)
