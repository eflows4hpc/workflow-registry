from string import ascii_letters

from pytest import raises

from esm_simulation import main, _create_expid


def test_esm_init(mocker, prepare_esm_simulation):
    with raises(FileNotFoundError) as e:
        esm_ensemble_conf = prepare_esm_simulation()
        mocker.patch('sys.argv', [
            'esm_simulation',
            '--debug',
            '--expid', '123456',
            '--start_dates', '1948',
            '--model=fesom2',
            '--processes', '2',
            '--processes_per_node', '48',
            '--config', str(esm_ensemble_conf)
        ])
        main()

    assert "No such file or directory:" in str(e.value)


def test_main_prints_help(capsys):
    with raises(SystemExit):
        main()

    assert 'usage: esm_simulation' in str(capsys.readouterr())


def test_create_expid():
    expid = _create_expid()

    # 1 char, 5 digits
    assert len(expid) == 6

    assert expid[0] in ascii_letters
    number = int(expid[1:])
    assert number > 0
