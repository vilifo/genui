from genui.models.genuimodels import bases
import numpy as np

class Mean(bases.ValueAggregationFunction):
    name = "Mean"
    description = "Mean value of the input values."

    def __call__(self, values):
        return np.mean(values)


class Median(bases.ValueAggregationFunction):
    name = "Median"
    description = "Median value of the input values."

    def __call__(self, values):
        return np.median(values)
