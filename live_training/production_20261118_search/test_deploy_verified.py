"""Exercise actual partial switch recovery without running Docker."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('orbit24_deploy', Path(__file__).with_name('deploy_verified.py'))
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)

OLD_ID = 'a' * 64
NEW_ID = 'b' * 64
NEW_IMAGE = 'sha256:' + 'c' * 64
COMPOSE = '''services:
  caddy:
    image: caddy:latest
  pesten-server:
    image: pesten-server:teacher4387-orbit24-20260926
    restart: unless-stopped
    environment:
      KEEP: preserved
  unrelated:
    image: unrelated:original
networks:
  default:
    name: caddy_default
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    old = {'Id': OLD_ID, 'Image': d.OLD_IMAGE, 'Name': '/' + d.LIVE,
           'State': {'Running': True, 'Health': {'Status': 'healthy'}},
           'Config': {'Env': ['SECRET=never print this'], 'User': 'app', 'WorkingDir': '/app',
                      'Cmd': ['python', 'web_server.py'], 'Entrypoint': None, 'Labels': None},
           'HostConfig': {'RestartPolicy': {'Name': 'unless-stopped', 'MaximumRetryCount': 0},
                          'PortBindings': {}, 'Memory': 0},
           'Mounts': [], 'NetworkSettings': {'Networks': {'caddy_default': {}}}}
    containers = {OLD_ID: copy.deepcopy(old)}
    calls = []
    fault = {'at': None, 'fired': False}
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    monkeypatch.setattr(d, 'COMPOSE', tmp_path / 'compose.yml')
    d.COMPOSE.write_text(COMPOSE)
    monkeypatch.setattr(d.signal, 'signal', lambda *args: None)

    def inspect(name):
        for value in containers.values():
            if name in (value['Id'], value['Name'].removeprefix('/')):
                return copy.deepcopy(value)
        raise d.DeploymentError('Missing container')

    def optional(name):
        try:
            return inspect(name)
        except d.DeploymentError:
            return None

    def docker(*args, **kwargs):
        calls.append(args)
        action = args[0]
        if action == 'image':
            return NEW_IMAGE
        if action == 'stop':
            containers[args[-1]]['State']['Running'] = False
        elif action == 'rename':
            require_empty = optional(args[-1])
            assert require_empty is None
            containers[args[1]]['Name'] = '/' + args[2]
        elif action == 'update':
            containers[args[-1]]['HostConfig']['RestartPolicy']['Name'] = args[1].split('=', 1)[1]
        elif action == 'start':
            containers[args[1]]['State']['Running'] = True
        elif action == 'rm':
            del containers[args[-1]]
        else:
            raise AssertionError(f'Unexpected fake action {action}')
        # Simulate daemon applying an operation before CLI reports failure.
        if args == fault['at'] and not fault['fired']:
            fault['fired'] = True
            raise d.DeploymentError('Injected failure')
        return ''

    def create(*args):
        candidate = copy.deepcopy(old)
        candidate.update(Id=NEW_ID, Image=NEW_IMAGE, Name='/' + d.READY)
        candidate['State']['Running'] = False
        containers[NEW_ID] = candidate
        if fault['at'] == 'create':
            fault['fired'] = True
            raise d.DeploymentError('Create succeeded but response lost')
        return NEW_ID

    config = {'image_id': NEW_IMAGE, 'staging_container_id': 'stage', 'image_files': {'web_server.py': 'f' * 64}}
    monkeypatch.setattr(d, 'inspect', inspect)
    monkeypatch.setattr(d, 'optional_inspect', optional)
    monkeypatch.setattr(d, 'docker', docker)
    monkeypatch.setattr(d, 'create_replacement', create)
    monkeypatch.setattr(d, 'checked_health', lambda *args, **kwargs: dict(d.HEALTH))
    monkeypatch.setattr(d, 'checked_files', lambda *args: None)
    monkeypatch.setattr(d, 'wait_healthy', lambda name: d.require(inspect(name)['State']['Running'], 'Not running'))
    monkeypatch.setattr(d, 'preflight', lambda _: (copy.deepcopy(old), COMPOSE, *d.replace_pesten_image(COMPOSE, d.IMAGE), dict(d.HEALTH)))
    return containers, calls, fault, config, old


@pytest.mark.parametrize('at', [
    'create',
    ('stop', '--time', '15', OLD_ID),
    ('rename', OLD_ID, d.ROLLBACK),
    ('update', '--restart=no', OLD_ID),
    ('rename', NEW_ID, d.LIVE),
    ('start', NEW_ID),
])
def test_daemon_applied_mutation_then_error_restores_old(world, at):
    containers, calls, fault, config, old = world
    fault['at'] = at
    with pytest.raises(d.DeploymentError, match='previous container restored'):
        d.deploy(config)
    assert fault['fired']
    assert containers == {OLD_ID: old}
    assert d.COMPOSE.read_text() == COMPOSE


def test_health_failure_rolls_back_and_keeps_unrelated_containers(world, monkeypatch):
    containers, calls, fault, config, old = world
    unrelated = copy.deepcopy(old)
    unrelated.update(Id='unrelated', Name='/pesten-server-rollback-20260907')
    containers['unrelated'] = unrelated
    def health(url, **kwargs):
        if kwargs.get('origin') and kwargs.get('expected_fields') is None:
            raise d.DeploymentError('Wrong profile')
        return d.HEALTH
    monkeypatch.setattr(d, 'checked_health', health)
    with pytest.raises(d.DeploymentError, match='previous container restored'):
        d.deploy(config)
    assert containers == {OLD_ID: old, 'unrelated': unrelated}


def test_success_and_explicit_rollback_preserve_other_compose_edits(world):
    containers, calls, fault, config, old = world
    assert d.deploy(config)['status'] == 'deployed'
    assert containers[OLD_ID]['Name'] == '/' + d.ROLLBACK
    assert not containers[OLD_ID]['State']['Running']
    assert containers[OLD_ID]['HostConfig']['RestartPolicy']['Name'] == 'no'
    assert containers[NEW_ID]['Name'] == '/' + d.LIVE
    updated = d.COMPOSE.read_text().replace('unrelated:original', 'unrelated:subsequent')
    d.COMPOSE.write_text(updated)
    journal = json.loads((d.ROOT / 'switch.private.json').read_text())
    assert d.rollback(journal)['status'] == 'rolled_back'
    assert containers == {OLD_ID: old}
    assert d.COMPOSE.read_text() == COMPOSE.replace('unrelated:original', 'unrelated:subsequent')


def test_compose_write_applied_then_error_rolls_back(world, monkeypatch):
    containers, calls, fault, config, old = world
    original = d.atomic_compose
    failed = False
    def atomic(text):
        nonlocal failed
        original(text)
        if not failed:
            failed = True
            raise d.DeploymentError('Applied write before failure')
    monkeypatch.setattr(d, 'atomic_compose', atomic)
    with pytest.raises(d.DeploymentError, match='previous container restored'):
        d.deploy(config)
    assert containers == {OLD_ID: old}
    assert d.COMPOSE.read_text() == COMPOSE


def test_concurrent_pre_switch_change_aborts_without_touching_old(world, monkeypatch):
    containers, calls, fault, config, old = world
    def changed(*args):
        containers[OLD_ID]['HostConfig']['Memory'] = 123
    monkeypatch.setattr(d, 'assert_preserved', changed)
    with pytest.raises(d.DeploymentError, match='previous container restored'):
        d.deploy(config)
    assert containers[OLD_ID]['HostConfig']['Memory'] == 123
    assert containers[OLD_ID]['State']['Running']
    assert all(args[0] not in ('stop', 'start', 'rename', 'update') for args in calls)
    assert NEW_ID not in containers


def test_foreign_live_name_is_never_removed(world):
    containers, calls, fault, config, old = world
    containers[OLD_ID]['Name'] = '/' + d.ROLLBACK
    foreign = copy.deepcopy(old)
    foreign.update(Id='foreign', Name='/' + d.LIVE)
    containers['foreign'] = foreign
    with pytest.raises(d.DeploymentError, match='another container'):
        d.rollback({'old_id': OLD_ID, 'new_id': NEW_ID, 'new_image_id': NEW_IMAGE,
                    'switch_started': True, 'old_compose_image': 'old'})
    assert not calls
    assert containers['foreign'] == foreign


def test_health_requires_exact_types(monkeypatch):
    import io
    wrong = dict(d.HEALTH, search_samples=128.0)
    monkeypatch.setattr(d.urllib.request, 'urlopen', lambda *args, **kwargs: io.StringIO(json.dumps(wrong)))
    with pytest.raises(d.DeploymentError, match='search_samples'):
        d.checked_health('http://example.test')


def test_subprocess_errors_do_not_disclose_command_or_stderr(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(d.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout='', stderr='private credential'))
    with pytest.raises(d.DeploymentError) as result:
        d.run(['docker', 'create', '--env', 'PASSWORD=secret'])
    assert str(result.value) == 'docker operation failed'


def test_compose_replacement_only_changes_image():
    updated, old = d.replace_pesten_image(COMPOSE, d.IMAGE)
    assert old == 'pesten-server:teacher4387-orbit24-20260926'
    assert updated == COMPOSE.replace(old, d.IMAGE)
    with pytest.raises(d.DeploymentError):
        d.replace_pesten_image(COMPOSE + '  pesten-server:\n    image: another\n', d.IMAGE)


def test_interrupted_journal_write_preserves_previous_json(tmp_path, monkeypatch):
    path = tmp_path / 'switch.private.json'
    d.private_json(path, {'status': 'preparing', 'old_id': OLD_ID})
    before = path.read_bytes()
    def failed_fsync(descriptor):
        raise OSError('simulated flush failure')
    monkeypatch.setattr(d.os, 'fsync', failed_fsync)
    with pytest.raises(OSError):
        d.private_json(path, {'status': 'deployed', 'old_id': OLD_ID, 'new_id': NEW_ID})
    assert path.read_bytes() == before
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [path]


def test_rollback_health_failure_still_restores_compose(world, monkeypatch):
    containers, calls, fault, config, old = world
    d.deploy(config)
    journal = json.loads((d.ROOT / 'switch.private.json').read_text())
    def failed_health(*args, **kwargs):
        raise d.DeploymentError('origin not ready')
    monkeypatch.setattr(d, 'checked_health', failed_health)
    with pytest.raises(d.DeploymentError, match='origin not ready'):
        d.rollback(journal)
    assert containers == {OLD_ID: old}
    assert d.COMPOSE.read_text() == COMPOSE


def test_default_dns_ulimit_oom_serializations_are_equivalent(world):
    _, _, _, _, old = world
    new = copy.deepcopy(old)
    old['HostConfig'].update(Dns=[], Ulimits=[], OomKillDisable=None)
    new['HostConfig'].update(Dns=None, Ulimits=None, OomKillDisable=False)
    d.assert_preserved(old, new)
    d.assert_preserved(new, old)


@pytest.mark.parametrize('key,custom', [
    ('Dns', ['192.0.2.53']),
    ('Ulimits', [{'Name': 'nofile', 'Soft': 1024, 'Hard': 1024}]),
    ('OomKillDisable', True),
    ('Memory', 768 * 1024 * 1024),
])
def test_real_host_setting_difference_is_rejected_and_only_key_is_reported(world, key, custom):
    _, _, _, _, old = world
    old['HostConfig'].update(Dns=[], Ulimits=[], OomKillDisable=None)
    new = copy.deepcopy(old)
    new['HostConfig'][key] = custom
    with pytest.raises(d.DeploymentError) as error:
        d.assert_preserved(old, new)
    assert str(error.value) == 'Replacement host settings differ: ' + key


def test_default_normalization_does_not_hide_missing_fields(world):
    _, _, _, _, old = world
    new = copy.deepcopy(old)
    new['HostConfig']['Dns'] = None
    with pytest.raises(d.DeploymentError, match='Dns'):
        d.assert_preserved(old, new)


def test_sanitized_failure_reason_survives_rollback(world):
    _, _, fault, config, _ = world
    fault['at'] = ('rename', NEW_ID, d.LIVE)
    with pytest.raises(d.DeploymentError):
        d.deploy(config)
    journal = json.loads((d.ROOT / 'switch.private.json').read_text())
    assert journal['status'] == 'rolled_back'
    assert journal['failure'] == {'exception_type': 'DeploymentError', 'reason': 'Injected failure'}


def test_unexpected_failure_message_is_not_saved(world, monkeypatch):
    _, _, _, config, _ = world
    def fail(*args):
        raise ValueError('SECRET=never record this')
    monkeypatch.setattr(d, 'assert_preserved', fail)
    with pytest.raises(d.DeploymentError):
        d.deploy(config)
    journal_text = (d.ROOT / 'switch.private.json').read_text()
    journal = json.loads(journal_text)
    assert journal['failure'] == {'exception_type': 'ValueError', 'reason': 'Unexpected ValueError'}
    assert 'SECRET' not in journal_text
