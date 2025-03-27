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

class Min(bases.ValueAggregationFunction):
    name = "Min"
    description = "Minimum value of the input values."

    def __call__(self, values):
        return np.min(values)

class Max(bases.ValueAggregationFunction):
    name = "Max"
    description = "Maximum value of the input values."

    def __call__(self, values):
        return np.max(values)
