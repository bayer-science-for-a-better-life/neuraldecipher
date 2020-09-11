import rdkit.Chem as Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
import argparse
import numpy as np
from multiprocessing import Pool
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
import datetime
import itertools
import os

from utils import str_to_bool


def get_ecfp_count_vector(smiles: str, radius: int, nbits: int) -> np.ndarray:
    """
    Returns the count ECFP representation as numpy array
    :param smiles: Smiles string
    :param radius: Radius for the ECFP algorithm. (eq. to number of iterations per atom)
    :param nbits: Length of final ECFP representation
    :return: ECFP as numpy array
    """
    m = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetHashedMorganFingerprint(m, radius, nbits)
    ecfp_count = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ecfp_count)
    return ecfp_count


def convert2bit(x: np.ndarray) -> np.ndarray:
    """
    Returns the ECFP bit representation of a ECFP count representation.
    :param x: ECFP count array
    :return: ECFP bit array
    """
    ecfp_bit = np.array(x>0, dtype=np.int8)
    return ecfp_bit


def create_fp_and_save(smiles_list: list, radius: int, nbits: int,
                       smiles_temporal_list: list, nworkers: int):
    """
    Helper function to create ECFP count and bit representation and saving the results.
    For example: ´../data/dfFold{nbits}/{2*radius}/_train_c.npy´ for the count ECFP.
    :param radius: Radius for the ECFP algorithm.
    :param nbits: Length of the folded ECFP
    :param smiles_list: List of SMILES where the ECFP representation should be computed
    :param smiles_temporal_list: List of SMILES where the ECFP representation should be computed for the temporal testset
    :param nworkers: Number of (parallel) workers to use for computing the ECFPs
    :return:
    """

    start = datetime.datetime.now().replace(microsecond=0)
    save_dir = f"../data/dfFold{nbits}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Processing Chembl25 for training and validation")
    count_save_path = f"{save_dir}/ecfp{int(2*radius)}_train_c.npy"
    bit_save_path = f"{save_dir}/ecfp{int(2*radius)}_train_b.npy"

    # Compute ECFP count representation
    if nworkers > 1:
        with Pool(processes=nworkers) as pool:
            ecfp_count = pool.starmap(get_ecfp_count_vector, iterable=zip(smiles_list,
                                                                          itertools.repeat(radius, len(smiles_list)),
                                                                          itertools.repeat(nbits, len(smiles_list)))
                                      )
    else:
        ecfp_count = [get_ecfp_count_vector(smi, radius=radius, nbits=nbits) for smi in smiles_list]

    # Get ECFP bit representation
    ecfp_bit = [convert2bit(x) for x in ecfp_count]

    #small assert
    assert np.max(ecfp_count) > 1.0, print("The count ECFP was not correctly computed")
    assert np.max(ecfp_bit) == 1, print(f"Max val for ECFP bit: {np.max(ecfp_bit)}. Error.")

    np.save(bit_save_path, ecfp_bit)
    np.save(count_save_path, ecfp_count)
    del ecfp_count, ecfp_bit

    # temporal split
    print("Processing Chembl26 temporal split for testing")
    count_save_path = f"{save_dir}/ecfp{int(2 * radius)}_temporal_c.npy"
    bit_save_path = f"{save_dir}/ecfp{int(2 * radius)}_temporal_b.npy"

    # Compute ECFP count representation
    if nworkers > 1:
        with Pool(processes=nworkers) as pool:
            ecfp_count = pool.starmap(get_ecfp_count_vector, iterable=zip(smiles_temporal_list,
                                                                          itertools.repeat(radius,
                                                                                           len(smiles_temporal_list)),
                                                                          itertools.repeat(nbits,
                                                                                           len(smiles_temporal_list)))
                                      )
    else:
        ecfp_count = [get_ecfp_count_vector(smi, radius=radius, nbits=nbits) for smi in smiles_temporal_list]

    # Get ECFP bit representation
    ecfp_bit = [convert2bit(x) for x in ecfp_count]

    # small assert
    assert np.max(ecfp_count) > 1.0, print("The count ECFP was not correctly computed")
    assert np.max(ecfp_bit) == 1, print(f"Max val for ECFP bit: {np.max(ecfp_bit)}. Error.")

    np.save(bit_save_path, ecfp_bit)
    np.save(count_save_path, ecfp_count)

    end = datetime.datetime.now().replace(microsecond=0)
    difference = end - start
    print(f"Execution time for radius {radius} and length {nbits}: {difference}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocessing script to compute ECFP fingerprints \
                                                  from SMILES representations')
    parser.add_argument('--all', action='store', dest='all',
                        default="False",
                        help="Whether or not all ECFP configurations from the paper should be computed.\
                             Defaults to False. In the False setting, the ECFP6_1024 count and bit will be computed.")

    parser.add_argument('--nworkers', action='store', dest='nworkers',
                        default=1, type=int,
                        help="Number of workers for multiprocessing. Defaults to 1 and therefore no parallelisation")

    args = parser.parse_args()
    all_flag = str_to_bool(args.all)

    if args.nworkers == 1:
        print("Script running without parallelisation. You might want to use parallelism. Check --nworkers flag")
    else:
        print(f"Using {args.nworkers} processes in parallel.")

    import os
    source_path = os.path.abspath(__file__)
    base_path = os.path.dirname(os.path.dirname(source_path))

    start_time = datetime.datetime.now().replace(microsecond=0)

    smiles = np.load(os.path.join(base_path, "data/smiles.npy"), allow_pickle=True).tolist()
    smiles_temporal = np.load(os.path.join(base_path, "data/smiles_temporal.npy"), allow_pickle=True).tolist()

    print(f"Filtered Chembl25 SMILES dataset consists of {len(smiles)} unique samples.")
    print(f"Temporal Chembl26 SMILES dataset consists of {len(smiles_temporal)} unique samples.")

    if all_flag:
        SETTINGS = {"1024": {"r": [3]},
                    "2048": {"r": [3]},
                    "4096": {"r": [2, 3, 4, 5]},
                    "8192": {"r": [3]},
                    "16384": {"r": [3]},
                    "32768": {"r": [3]}}
    else:
        SETTINGS = {"1024": {"r": [3]}}

    for nbits, nested in SETTINGS.items():
        for r in nested["r"]:
            print(f"Computing counts and bits for length {nbits} and radius {r}...")
            create_fp_and_save(smiles_list=smiles, radius=int(r), nbits=int(nbits),
                               smiles_temporal_list=smiles_temporal,
                               nworkers=args.nworkers)

    print("Finished.")
    end_time = datetime.datetime.now().replace(microsecond=0)
    time_difference = end_time - start_time
    print(f"Script execution time: {time_difference}.")