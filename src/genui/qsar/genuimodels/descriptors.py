"""
descriptors

Created by: Martin Sicho
On: 16-01-20, 11:08
"""
import traceback

from rdkit.Chem.Scaffolds import MurckoScaffold
from qsprpred.data.descriptors import fingerprints as fp_module
from qsprpred.data.descriptors import sets as descriptors_set_module

from . import bases
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd

from genui.utils.inspection import get_non_abstract_classes_from_module


class MorganFPCalculator(bases.DescriptorCalculator):
    group_name = "MORGANFP"

    def __call__(self, smiles, radius=3, bit_len=4096, scaffold=0, **kwargs):
        fps = np.zeros((len(smiles), bit_len))
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            arr = np.zeros((1,))
            try:
                if scaffold == 1:
                    mol = MurckoScaffold.GetScaffoldForMol(mol)
                elif scaffold == 2:
                    mol = MurckoScaffold.MakeScaffoldGeneric(mol)
                if not mol:
                    raise Exception(f'Failed to calculate Morgan fingerprint (creating RDKit instance from smiles failed: {smile})')
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps[i, :] = arr
            except Exception as exp:
                # TODO: use a more specific exception related to descriptor errors
                # traceback.print_exc()
                self.builder.errors.append(exp)
                fps[i, :] = [0] * bit_len
        return pd.DataFrame(fps)

class QSPRPredFingerprintCalculator(bases.DescriptorCalculator):
    group_name = "QSPRPRED_FINGERPRINT"

    def __init__(self, builder):
        super().__init__(builder)
        self.fingerprint_instance = None

    def __call__(self, smiles, **kwargs):
        if not self.fingerprint_instance:
            fp_name = kwargs["fingerprint"]
            kwargs.pop("fingerprint")
            self.fingerprint_instance = getattr(fp_module, fp_name)(**kwargs)
        mols = self.smilesToMol(smiles)
        mols = self.fingerprint_instance.prepMols(mols)
        return self.fingerprint_instance.getDescriptors(mols, props={})

    @staticmethod
    def get_endpoints():
        return get_non_abstract_classes_from_module(fp_module)


class QSPRPredDescriptorSetCalculator(bases.DescriptorCalculator):
    group_name = "QSPRPRED_DESCRIPTOR_SET"

    def __init__(self, builder, descriptor_set):
        super().__init__(builder)
        self.descriptor_set_instance = None

    def __call__(self, smiles, **kwargs):
        if not self.descriptor_set_instance:
            set_name = kwargs["descriptor_set"]
            kwargs.pop("descriptor_set")
            self.descriptor_set_instance = getattr(descriptors_set_module, set_name)(**kwargs)
        return self.descriptor_set_instance(smiles)

    @staticmethod
    def get_endpoints():
        return get_non_abstract_classes_from_module(descriptors_set_module)
