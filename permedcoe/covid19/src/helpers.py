import os
import glob
from pathlib import Path


def get_genefiles(prefix, genes):
    """
    Create a list of genes files for the given patient.

    :param prefix: prefix
    :param genes: KO genes
    :return: List of names to be processed
    """
    genefiles = []
    for gene in genes:
        if gene != "":
            name = prefix + "_personalized__" + gene + "_ko"
        else:
            name = prefix + "_personalized"
        genefiles.append(name)
    return genefiles
