from pytest import raises

from esm_simulation import main


def test_esm_init(mocker, prepare_esm_simulation):
    with raises(FileNotFoundError) as e:
        esm_ensemble_conf = prepare_esm_simulation()
        mocker.patch('sys.argv', [
            'esm_simulation',
            '--debug',
            '--expid', '123456',
            '--model=fesom2',
            '--config', str(esm_ensemble_conf)
        ])
        main()

    assert "No such file or directory: 'srun'" in str(e.value)


def test_main_prints_help(capsys):
    with raises(SystemExit):
        main()

    assert 'usage: esm_simulation' in str(capsys.readouterr())
