"""One-time, ID-guarded deployment on the Pesten origin; no SSH operations.

Run preflight first, then deploy. rollback restores the saved container and only
the Pesten Compose image entry. Configuration and backups stay root-only.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path('/root/pesten-deployments/20261118-search')
COMPOSE = Path('/root/caddy/compose.yml')
IMAGE = 'pesten-server:learned-search-20261118'
OLD_IMAGE = 'sha256:4b620f1e8710f2021476e0fe8eba96e2ace94b560a02a28ebe177c7e787457c6'
LIVE = 'pesten-server'
STAGING = 'pesten-search-candidate-20261118'
READY = 'pesten-search-ready-20261118'
ROLLBACK = 'pesten-server-rollback-20261118'
HEALTH = {
    'status': 'ok',
    'source_checkpoint_sha256': 'f8f0aece2d759eadc60b3cbccca30c321c01da13322f7cae511a73c8da3a9f93',
    'runtime_artifact_sha256': '14cb8f5b7f552c923ad4b6af6b333da1c97d444e09d4e8df56f97d65f2dc2afc',
    'profile': 'teacher4387_orbit24_learned_search',
    'profile_sha256': '03c8c111927fdfe6d3355fe47b6bc1d12e260ea31514622392e5de8aced08426',
    'search_table_sizes': [2, 3, 4],
    'search_samples': 128,
    'search_horizon': 72,
    'search_trust': 72,
    'orbit_size': 24,
    'aggregation': 'mean_logits',
    'deployment_authorization': 'user_requested',
}


class DeploymentError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise DeploymentError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(command, *, data=None):
    result = subprocess.run(command, input=data, capture_output=True, text=True)
    # Never include command arguments, stderr, inspect data, or environment in
    # exceptions: deployment metadata includes inherited application secrets.
    require(result.returncode == 0, f'{Path(command[0]).name} operation failed')
    return result.stdout.strip()


def docker(*arguments, data=None):
    return run(['docker', *arguments], data=data)


def inspect(name):
    return json.loads(docker('inspect', name))[0]


def optional_inspect(name):
    # A failed inspect is only absence if the daemon's container list agrees.
    ids = docker('ps', '-a', '--no-trunc', '--format', '{{.ID}} {{.Names}}')
    matches = [line.split()[0] for line in ids.splitlines()
               if len(line.split()) == 2 and name in line.split()]
    require(len(matches) <= 1, 'Ambiguous container identity')
    return inspect(matches[0]) if matches else None


def private_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(json.dumps(value, indent=2) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replace_pesten_image(text, image):
    # Inspect only the two-space Pesten service block; unrelated services and
    # formatting are copied byte-for-byte. Ambiguous layouts fail closed.
    blocks = list(re.finditer(r'^  pesten-server:[ \t]*\n(?:(?!^  \S)[\s\S])*', text, re.MULTILINE))
    require(len(blocks) == 1, 'Cannot identify unique Pesten Compose service')
    block = blocks[0]
    entries = list(re.finditer(r'^(    image: )([^\r\n]+)', block[0], re.MULTILINE))
    require(len(entries) == 1, 'Cannot identify unique Pesten Compose image')
    entry = entries[0]
    start, end = block.start() + entry.start(2), block.start() + entry.end(2)
    previous = text[start:end]
    require(re.fullmatch(r'[A-Za-z0-9_./:@-]+', previous) is not None,
            'Pesten image entry uses unexpected syntax')
    return text[:start] + image + text[end:], previous


def atomic_compose(text):
    mode = COMPOSE.stat().st_mode & 0o777
    temporary = COMPOSE.with_name('compose.search-20261118.tmp')
    require(not temporary.exists(), 'Compose temporary file already exists')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        temporary.replace(COMPOSE)
    finally:
        temporary.unlink(missing_ok=True)


def checked_health(url, *, origin=False, expected_fields=HEALTH):
    if origin:
        value = json.loads(run([
            'curl', '--fail', '--silent', '--show-error', '--max-time', '8',
            '--resolve', 'pesten.tommiekalb.net:443:127.0.0.1', url,
        ]))
    else:
        with urllib.request.urlopen(url, timeout=8) as response:
            value = json.load(response)
    for key, expected in expected_fields.items():
        require(key in value and type(value[key]) is type(expected) and value[key] == expected,
                f'Health field mismatch: {key}')
    return {key: value[key] for key in expected_fields}


def checked_files(container, expected):
    require(bool(expected), 'Image source/artifact hash list is empty')
    for name, sha in expected.items():
        require(isinstance(name, str) and name and not Path(name).is_absolute()
                and '..' not in Path(name).parts and re.fullmatch(r'[a-f0-9]{64}', sha),
                'Invalid image file hash entry')
    code = ('import hashlib,json,pathlib,sys; names=json.load(sys.stdin); '
            'print(json.dumps({n:hashlib.sha256((pathlib.Path("/app")/n).read_bytes()).hexdigest() '
            'for n in names},sort_keys=True))')
    actual = json.loads(docker('exec', '-i', container, 'python', '-c', code,
                               data=json.dumps(sorted(expected))))
    require(actual == expected, 'Image source/artifact hash mismatch')


def wait_healthy(container):
    for _ in range(60):
        state = inspect(container)['State']
        require(state['Running'], 'Replacement stopped during readiness check')
        if state.get('Health', {}).get('Status') == 'healthy':
            return
        time.sleep(1)
    raise DeploymentError('Replacement did not become healthy')


def stable_container(value):
    return {key: value[key] for key in ('Id', 'Image', 'Name', 'Config', 'HostConfig', 'Mounts')} | {
        'networks': sorted(value['NetworkSettings']['Networks']),
    }


def assert_old(old):
    require(old['Image'] == OLD_IMAGE, 'Live image changed; re-audit required')
    require(old['State']['Running'], 'Existing service is not running')
    require(not old['Mounts'] and not old['HostConfig']['PortBindings'], 'Unexpected mount or port mapping')
    require(sorted(old['NetworkSettings']['Networks']) == ['caddy_default'], 'Unexpected live network')
    require(not old['Config'].get('Labels'), 'Unexpected live labels')
    require(not old['Config'].get('Entrypoint') and old['Config']['Cmd'] == ['python', 'web_server.py'],
            'Unexpected live startup command')
    require(old['Config']['User'] == 'app' and old['Config']['WorkingDir'] == '/app',
            'Unexpected live user or working directory')
    require(old['HostConfig']['RestartPolicy'] == {'Name': 'unless-stopped', 'MaximumRetryCount': 0},
            'Unexpected live restart policy')


def preflight(config):
    require(re.fullmatch(r'sha256:[a-f0-9]{64}', config['image_id']) is not None, 'Invalid reviewed image ID')
    require(config['image_id'] != OLD_IMAGE, 'Replacement image equals existing image')
    old = inspect(LIVE)
    assert_old(old)
    require(old['Id'] == config['old_container_id'], 'Live container identity changed')
    staging = inspect(STAGING)
    require(staging['Id'] == config['staging_container_id'], 'Staging container identity changed')
    require(staging['Image'] == config['image_id'] and staging['State']['Running'], 'Staging image/state changed')
    require(staging['HostConfig']['PortBindings'].get('5000/tcp') ==
            [{'HostIp': '127.0.0.1', 'HostPort': '15488'}], 'Staging health port mapping changed')
    require(docker('image', 'inspect', IMAGE, '--format', '{{.Id}}') == config['image_id'], 'Image tag changed')
    for name in (READY, ROLLBACK):
        require(optional_inspect(name) is None, f'Reserved container already exists: {name}')
    require(not (ROOT / 'switch.private.json').exists(), 'Deployment journal already exists; inspect or rollback')
    compose = COMPOSE.read_text(encoding='utf-8')
    require(digest(compose.encode()) == config['compose_sha256'], 'Compose changed since review')
    updated, previous = replace_pesten_image(compose, IMAGE)
    require(previous == 'pesten-server:teacher4387-orbit24-20260926', 'Unexpected existing Pesten Compose image')
    health = checked_health('http://127.0.0.1:15488/health')
    checked_files(staging['Id'], config['image_files'])
    return old, compose, updated, previous, health


def create_replacement(old, config):
    envfile = ROOT / 'environment.private.tmp'
    require(not envfile.exists(), 'Private environment temporary file already exists')
    env = old['Config'].get('Env') or []
    require(all('\n' not in item and '\r' not in item for item in env),
            'Multiline environment requires a reviewed deployment path')
    try:
        with envfile.open('x', encoding='utf-8') as handle:
            handle.write('\n'.join(env) + '\n')
        envfile.chmod(0o600)
        return docker('create', '--name', READY, '--restart', 'unless-stopped',
                      '--network', 'caddy_default', '--network-alias', LIVE,
                      '--user', old['Config']['User'], '--workdir', old['Config']['WorkingDir'],
                      '--env-file', str(envfile), config['image_id'], *old['Config']['Cmd'])
    finally:
        envfile.unlink(missing_ok=True)


def assert_preserved(old, new):
    require(sorted(old['Config'].get('Env') or []) == sorted(new['Config'].get('Env') or []),
            'Replacement environment differs from existing service')
    for key in ('User', 'WorkingDir', 'Cmd', 'Entrypoint', 'Labels', 'Volumes',
                'Tty', 'OpenStdin', 'StdinOnce', 'StopSignal', 'StopTimeout'):
        require(old['Config'].get(key) == new['Config'].get(key), f'Replacement setting differs: {key}')
    # The existing daemon serializes its older container's default DNS/ulimits
    # as [], and its new container's defaults as null; unset OOM disable is
    # similarly null versus false. These three spellings have identical Docker
    # defaults. Keep every other setting, and all non-default values, strict.
    def host_settings(value):
        result = dict(value)
        for key in ('Dns', 'Ulimits'):
            if key in result and result[key] is None:
                result[key] = []
        if 'OomKillDisable' in result and result['OomKillDisable'] is None:
            result['OomKillDisable'] = False
        return result
    before, after = host_settings(old['HostConfig']), host_settings(new['HostConfig'])
    missing = object()
    changed = sorted(key for key in before.keys() | after.keys()
                     if before.get(key, missing) != after.get(key, missing))
    # Only field names are safe diagnostics; settings may contain credentials.
    names = [key if re.fullmatch(r'[A-Za-z0-9_]+', key) else 'unknown_field' for key in changed]
    require(not changed, 'Replacement host settings differ: ' + ', '.join(names))
    require(not new['Mounts'] and sorted(new['NetworkSettings']['Networks']) == ['caddy_default'],
            'Replacement mounts/network differ')


def rollback(journal):
    """Recover actual IDs, including interruption between any stop/rename/start."""
    old_id, new_id = journal['old_id'], journal.get('new_id')
    old = inspect(old_id)
    require(old['Image'] == OLD_IMAGE, 'Rollback container image changed')
    occupant = optional_inspect(LIVE)
    require(occupant is None or occupant['Id'] in (old_id, new_id),
            'Live name belongs to another container; refusing to remove it')
    replacement = optional_inspect(new_id) if new_id else None
    if replacement is not None:
        require(replacement['Image'] == journal['new_image_id'], 'Replacement identity changed')
        docker('rm', '-f', new_id)
    if not journal.get('switch_started', False):
        journal['status'] = 'aborted_before_switch'
        private_json(ROOT / 'switch.private.json', journal)
        return {'status': 'aborted_before_switch', 'container': LIVE, 'image_id': OLD_IMAGE}
    old = inspect(old_id)
    if old['Name'] != '/' + LIVE:
        docker('rename', old_id, LIVE)
    docker('update', '--restart=unless-stopped', old_id)
    docker('start', old_id)
    current = COMPOSE.read_text(encoding='utf-8')
    restored, image = replace_pesten_image(current, journal['old_compose_image'])
    require(image in (IMAGE, journal['old_compose_image']), 'Pesten Compose image changed; manual review required')
    if image == IMAGE:
        # Preserve unrelated Compose edits made since deployment.
        require(COMPOSE.read_text(encoding='utf-8') == current, 'Compose changed during rollback')
        atomic_compose(restored)
    # A health failure must not leave Compose pointing at the rejected image.
    wait_healthy(old_id)
    checked_health('https://pesten.tommiekalb.net/health', origin=True,
                   expected_fields={key: HEALTH[key] for key in
                                    ('status', 'source_checkpoint_sha256', 'runtime_artifact_sha256')}
                                   | {'profile': 'teacher4387_orbit24'})
    journal['status'] = 'rolled_back'
    private_json(ROOT / 'switch.private.json', journal)
    return {'status': 'rolled_back', 'container': LIVE, 'image_id': OLD_IMAGE}


def deploy(config):
    old, compose, updated, previous, _ = preflight(config)
    private_json(ROOT / 'before_container.private.json', old)
    (ROOT / 'compose_before.private.yml').write_text(compose, encoding='utf-8')
    (ROOT / 'compose_before.private.yml').chmod(0o600)
    journal = {'status': 'preparing', 'old_id': old['Id'], 'new_id': None, 'switch_started': False,
               'new_image_id': config['image_id'], 'old_compose_image': previous}
    private_json(ROOT / 'switch.private.json', journal)
    try:
        # Creation may succeed before the client reports a failure. Recovery
        # finds only our previously absent reserved name in that narrow window.
        try:
            journal['new_id'] = create_replacement(old, config)
        finally:
            if not journal['new_id']:
                ready = optional_inspect(READY)
                if ready is not None and ready['Image'] == config['image_id']:
                    journal['new_id'] = ready['Id']
            private_json(ROOT / 'switch.private.json', journal)
        replacement = inspect(journal['new_id'])
        require(replacement['Image'] == config['image_id'], 'Replacement image differs from review')
        assert_preserved(old, replacement)
        require(stable_container(inspect(LIVE)) == stable_container(old), 'Live container changed before switch')
        require(COMPOSE.read_text(encoding='utf-8') == compose, 'Compose changed before switch')
        require(docker('image', 'inspect', IMAGE, '--format', '{{.Id}}') == config['image_id'], 'Image tag changed before switch')
        checked_files(config['staging_container_id'], config['image_files'])
        checked_health('http://127.0.0.1:15488/health')
        journal['switch_started'] = True
        private_json(ROOT / 'switch.private.json', journal)
        docker('stop', '--time', '15', old['Id'])
        docker('rename', old['Id'], ROLLBACK)
        docker('update', '--restart=no', old['Id'])
        docker('rename', journal['new_id'], LIVE)
        docker('start', journal['new_id'])
        wait_healthy(journal['new_id'])
        checked_files(journal['new_id'], config['image_files'])
        health = checked_health('https://pesten.tommiekalb.net/health', origin=True)
        require(COMPOSE.read_text(encoding='utf-8') == compose, 'Compose changed during switch')
        atomic_compose(updated)
        journal['status'] = 'deployed'
        private_json(ROOT / 'switch.private.json', journal)
        result = {'status': 'deployed', 'image': IMAGE, 'image_id': config['image_id'],
                  'rollback_container': ROLLBACK, 'origin_health': health,
                  'compose_pesten_image_updated': True}
        private_json(ROOT / 'deployment_result.json', result)
        return result
    except BaseException as failure:
        journal['failure'] = {
            'exception_type': type(failure).__name__,
            'reason': str(failure) if isinstance(failure, DeploymentError)
                      else 'Unexpected ' + type(failure).__name__,
        }
        try:
            private_json(ROOT / 'switch.private.json', journal)
        except BaseException:
            # Diagnostics must never prevent recovery after an I/O failure.
            pass
        # Ignore repeated termination during recovery. Never remove a container
        # merely by a name that another process could have claimed.
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        try:
            rollback(journal)
        except BaseException:
            raise DeploymentError('Deployment failed and automatic rollback needs review; use private journal') from None
        raise DeploymentError('Deployment failed; previous container restored') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'deploy', 'rollback'))
    parser.add_argument('--config', type=Path, default=ROOT / 'deployment_config.json')
    args = parser.parse_args()
    require(os.geteuid() == 0, 'Run on the origin as root')
    os.umask(0o077)
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(ROOT.stat().st_uid == 0 and ROOT.stat().st_mode & 0o077 == 0,
            'Deployment directory must be root-owned and private')
    with (ROOT / 'deployment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(signum, frame):
            raise DeploymentError('Deployment interrupted')
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, interrupted)
        if args.action == 'rollback':
            result = rollback(json.loads((ROOT / 'switch.private.json').read_text()))
        else:
            config = json.loads(args.config.read_text(encoding='utf-8'))
            if args.action == 'preflight':
                preflight(config)
                result = {'status': 'preflight_passed', 'image_id': config['image_id']}
            else:
                result = deploy(config)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except DeploymentError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
    except BaseException:
        print('Deployment operation failed; inspect private state on the origin', file=sys.stderr)
        sys.exit(1)
