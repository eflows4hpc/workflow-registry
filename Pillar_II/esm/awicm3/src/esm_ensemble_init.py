import configparser
import os
import random
import shutil
from typing import Dict

from pycompss.api.parameter import IN  # type: ignore
# COMPSs/PyCOMPSs imports
from pycompss.api.task import task  # type: ignore


def esm_ensemble_setup_config(config_file_data: str, value_dict: Dict[str, str]) -> str:
    for key, value in value_dict.items():
        config_file_data = config_file_data.replace(key, value)

    return config_file_data


def esm_ensemble_process_namelist(namelist_template: str, outpath: str, namelist_value_dict: Dict[str, str]) -> None:
    # config namelists
    fin = open(os.path.join(os.path.dirname(__file__), 'config/awicm3', namelist_template + '.tmpl'), "rt")
    # read file contents to string
    data = fin.read()
    # replace all occurrences of the required string
    data = esm_ensemble_setup_config(data, namelist_value_dict)

    file = open(outpath + "/" + namelist_template, "w")
    file.write(data)
    file.close()


def esm_ensemble_generate_namelists(exp_id: str, outpath: str, start_year: str,
                                    esm_config: configparser.ConfigParser) -> None:
    try:
        # namelist.config
        mapdict = {'{START_YEAR}': start_year, '{MESH_PATH}': esm_config['fesom2']['mesh_file_path'],
                   '{CLIMATOLOGY_PATH}': esm_config['fesom2']['climatology_path'],
                   '{OUTPUT_PATH}': esm_config['common']['output_dir'] + start_year + "/"}
        esm_ensemble_process_namelist('namelist.config', outpath, mapdict)

        # namcoupled
        mapdict = {}
        esm_ensemble_process_namelist('namcouple', outpath, mapdict)

        # namelist.forcing
        mapdict = {'{FORCING_SET_PATH}': esm_config['fesom2']['forcing_files_path']}
        esm_ensemble_process_namelist('namelist.forcing', outpath, mapdict)

        # namelist.ice
        mapdict = {}
        esm_ensemble_process_namelist('namelist.ice', outpath, mapdict)

        # namelist.icepack
        ##mapdict = {}
        ##esm_ensemble_process_namelist('namelist.icepack', outpath, mapdict)

        # namelist.io
        mapdict = {}
        esm_ensemble_process_namelist('namelist.io', outpath, mapdict)

        # namelist.ice
        mapdict = {}
        esm_ensemble_process_namelist('namelist.oce', outpath, mapdict)

        # FESOM - runoffmapper config file
        mapdict = {}
        esm_ensemble_process_namelist('namelist.runoffmapper', outpath, mapdict)

        # IFS master config file
        mapdict = {'{START_YEAR}': start_year, '{EXP_ID}': 'awi3'}
        esm_ensemble_process_namelist('fort.4', outpath, mapdict)


    except OSError as exc:  # Python ≥ 2.5
        print("Config files Placeholder processing failed :" + exc.strerror)


@task(exp_id=IN)
def esm_ensemble_init(exp_id: str, setup_working_env: bool = True):
    # 1 - load ensemble config file
    print("Initialization for experiment " + str(exp_id))
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config', 'esm_ensemble.conf'))
    # 2 - create folders for the topdir and each ensemble member runtime folder
    if setup_working_env:
        # 1 - load ensemble config file
        print("Initialization for experiment " + str(exp_id))
        if config.has_option('common', 'top_working_dir'):
            top_dir = config['common']['top_working_dir'] + "/" + str(exp_id)
            output_dir = top_dir  # config['common']['output_dir']
            # define the access rights - readable and accessible by all users, and write access by only the owner
            access_rights = 0o755
            try:
                # top dir
                if not os.path.exists(top_dir):
                    os.makedirs(top_dir, access_rights)

                # output dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, access_rights)

                # get list of members
                sdates_list = (config['common']['ensemble_start_dates']).split()
                print("Creating directories for the following list of dates: " + str(sdates_list))
                for sdate in sdates_list:
                    print(sdate)
                    member_working_dir = top_dir + "/" + sdate
                    if not os.path.exists(member_working_dir):
                        os.mkdir(member_working_dir, access_rights)

                    member_output_dir = output_dir + "/" + sdate
                    if not os.path.exists(member_output_dir):
                        os.mkdir(member_output_dir, access_rights)

                    # copy experiment base files, ifs and fesom auxiliary files
                    # /home/bsc32/bsc32044/awi-cm3/tests/coupled/TCO95L91-CORE2-BASEFILES/
                    exp_base_files = '/home/bsc32/bsc32044/awi-cm3/tests/coupled/TCO95L91-CORE2-BASEFILES'
                    for item in os.listdir(exp_base_files):
                        os.symlink(os.path.join(exp_base_files, item), os.path.join(member_output_dir, item))

                    awicm3_binaries = '/home/bsc32/bsc32044/awi-cm3/tests/coupled/AWICM3_BIN'
                    for item in os.listdir(awicm3_binaries):
                        os.symlink(os.path.join(awicm3_binaries, item), os.path.join(member_output_dir, item))

                    fclock = open(member_output_dir + "/awi3fesom.clock", "w")
                    fclock.write("0 1 " + sdate + "\r\n")
                    fclock.write("0 1 " + sdate)
                    fclock.close()

                    fclock = open(member_output_dir + "/fesom.clock", "w")
                    fclock.write("0 1 " + sdate + "\r\n")
                    fclock.write("0 1 " + sdate)
                    fclock.close()

                    # generate the config files for each year/member
                    esm_ensemble_generate_namelists(exp_id, member_working_dir, sdate, config)
                    print("############ creating clock file  ################")
                    source_directory_datamodels = config['fesom2']['fesom_hecuba_datamodel']
                    for source_filename in os.listdir(source_directory_datamodels):
                        if source_filename.endswith(".yaml"):
                            source_file_path = os.path.join(source_directory_datamodels, source_filename)
                            shutil.copy(source_file_path, str(member_working_dir) + '/')

            except OSError as exc:  # Python ≥ 2.5
                print("Initialization failed :" + exc.strerror)
                raise
        else:
            print("ERROR - topdir undefined path not found")
    return config


# for testing purposes
if __name__ == "__main__":
    print("Running AWICM3 INIT")
    # first parameter: number of ensemble members
    # 0 - create a ramdon nr for the experiment id
    exp_id = random.randint(100000, 999999)
    esm_ensemble_init(exp_id, True)
    print("############ All experiment folders created successfully ################")
