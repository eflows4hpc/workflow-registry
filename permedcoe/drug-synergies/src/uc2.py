#!/usr/bin/python3

import os
import csv

# To set building block debug mode
from permedcoe import set_debug
# To set the default PyCOMPSs TMPDIR
from permedcoe import TMPDIR
# Import building block tasks
from build_model_from_species_BB import build_model_from_species
from personalize_patient_BB import personalize_patient_cellline
from MaBoSS_BB import MaBoSS_analysis
from MaBoSS_BB import MaBoSS_sensitivity_analysis
from print_drug_results_BB import print_drug_results_parallelized
# Import utils
from utils import parse_input_parameters
from helpers import get_genefiles

# PyCOMPSs imports
from pycompss.api.api import compss_wait_on_directory
from pycompss.api.api import compss_wait_on_file
from pycompss.api.api import compss_barrier


def get_cell_lines(rnaseq_data, limit=0):
    """ Retrieves the cell lines from the given rnaseq_data file.

    Example:
    cell_lines = ["SIDM00003", "SIDM00023", "SIDM00040", "SIDM00041"]

    :param rnaseq_data: Rnaseq csv file
    :type rnaseq_data: string
    :param limit: Maximum number of cell lines to retrieve (0 is all)
    :type limit: int
    :return: List of strings with the cell identifiers
    :rtype: [string]
    """
    with open(rnaseq_data, "r") as rnaseq_file:
        rnaseq_file_reader = csv.DictReader(rnaseq_file, delimiter=',')
        header = rnaseq_file_reader.fieldnames
    cell_lines = header[2:]  # removes the two first fields ('model_id', '',...)
    if limit == 0:
        return cell_lines
    else:
        return cell_lines[:limit]

def main():
    """
    MAIN CODE
    """
    set_debug(True)

    print("---------------------------")
    print("|   Use Case 2 Workflow   |")
    print("---------------------------")

    # GET INPUT PARAMETERS
    args = parse_input_parameters()

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder, exist_ok=True)

    # 1st STEP: Build model from species
    build_model_folder = os.path.join(args.results_folder, "build_model")
    os.makedirs(build_model_folder, exist_ok=True)
    model_bnd_path = os.path.join(build_model_folder, "model.bnd")
    model_cfg_path = os.path.join(build_model_folder, "model.cfg")

    build_model_from_species(
        tmpdir=TMPDIR,
        output_bnd_file=model_bnd_path,
        output_cfg_file=model_cfg_path,
        input_file=args.list_genes
    )

    # Get cell lines from rnaseq csv file
    cell_lines = get_cell_lines(args.rnaseq_data, limit=2)

    personalize_patient_folder = os.path.join(args.results_folder, "personalize_patient")
    os.makedirs(personalize_patient_folder, exist_ok=True)
    mutant_results_folder = os.path.join(args.results_folder, "mutant_results")
    os.makedirs(mutant_results_folder, exist_ok=True)
    results_files = []
    for cell_line in cell_lines:

        # 2nd STEP: Personalize patients
        personalize_patient_folder_cell = os.path.join(personalize_patient_folder, cell_line)
        os.makedirs(personalize_patient_folder_cell, exist_ok=True)
        personalize_patient_cellline(
            tmpdir=TMPDIR,
            expression_data=args.rnaseq_data,
            cnv_data=args.cn_data,
            mutation_data=args.mutation_data,
            model_bnd=model_bnd_path,
            model_cfg=model_cfg_path,
            t=cell_line,
            model_output_dir=personalize_patient_folder_cell
        )

        # 3rd STEP: MaBoSS
        mutant_results_folder_cell = os.path.join(mutant_results_folder, cell_line)
        os.makedirs(mutant_results_folder_cell, exist_ok=True)
        mutant_results_file = os.path.join(mutant_results_folder_cell, "sensitivity.json")
        MaBoSS_sensitivity_analysis(
            tmpdir=TMPDIR,
            model_folder=personalize_patient_folder_cell,
            genes_druggable=args.genes_drugs,
            genes_target=args.state_objective,
            result_file=mutant_results_file
        )
        results_files.append(mutant_results_file)

    report_folder = os.path.join(args.results_folder, "report")
    os.makedirs(report_folder, exist_ok=True)
    print_drug_results_parallelized(cell_lines,
                                    results_files,
                                    report_folder)

    compss_barrier()


if __name__ == "__main__":
    main()
