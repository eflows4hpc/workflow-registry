from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import Callable

import pytest


@pytest.fixture(scope='function')
def prepare_esm_simulation(request: pytest.FixtureRequest) -> Callable:
    tmp_dir = TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)

    def _create_simulation_configuration() -> Path:
        root_dir = tmp_path
        top_working_dir = root_dir / 'top_working_dir'
        output_dir = root_dir / 'output_dir'

        top_working_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        fesom_x = top_working_dir / 'fesom.x'
        fesom_x.touch(mode=0o774)

        with open(fesom_x, 'w+') as fesom_exe:
            fesom_exe.write('#!/bin/bash')
            fesom_exe.write('true')
            fesom_exe.flush()

        template_configuration = Path(__file__).parent.resolve() / 'esm_ensemble.conf'
        simulation_configuration = top_working_dir / 'esm_ensemble.conf'

        with open(template_configuration, 'r') as template, open(simulation_configuration, 'w+') as new_template:
            original = template.read()
            modified = original.format(**{
                'FESOM_EXE': str(fesom_x),
                'TEST_TEMP_DIR': str(root_dir)
            })
            new_template.write(modified)

        return simulation_configuration

    def finalizer() -> None:
        rmtree(tmp_path)

    request.addfinalizer(finalizer)

    return _create_simulation_configuration
