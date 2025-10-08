from genui.qsar.genuimodels.bases import ScaffoldCalculator


def create_scaffold_class(name):
    return type(name, (ScaffoldCalculator,), {
        'name': name,
        'abstract': False
    })

scaffold_types = [name for name in dir(ScaffoldCalculator.module) if not name.startswith('_')]

globals().update({
    name: create_scaffold_class(name)
    for name in scaffold_types
})
