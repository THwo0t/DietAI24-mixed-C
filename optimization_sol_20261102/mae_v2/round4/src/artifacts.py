"""Small private stage cache and immutable run seal; no provider or label I/O.

``stage_runner(name, inputs, call, validator)`` reparses cached and fresh raw
results. A valid JSON result with ``failure`` is still a terminal cached result.
Other call exceptions propagate unchanged; retries rely on the caller's durable
Provider namespace and payload budget, never on a counter reset here.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z')


class ArtifactError(RuntimeError):
    pass


class ArtifactBusy(ArtifactError):
    pass


class TerminalStageError(ArtifactError):
    """Explicit safe-code-only permanent blocker; ordinary failures use JSON."""
    def __init__(self, code):
        if not isinstance(code, str) or re.fullmatch(r'[a-z][a-z0-9_]{0,95}', code) is None:
            raise ValueError('Terminal stage error needs a fixed safe code')
        self.code = code
        super().__init__(code)


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def decode(raw):
    def pairs(values):
        out = {}
        for key, value in values:
            if key in out:
                raise ArtifactError('duplicate_json_key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ArtifactError('nonfinite_json')))


def normalized(value):
    return decode(canonical(value))


def name_token(value):
    if not isinstance(value, str) or TOKEN.fullmatch(value) is None or '..' in value:
        raise ArtifactError('unsafe_artifact_name')
    return value


def key_path(value):
    return hashlib.sha256(name_token(value).encode()).hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ArtifactStore:
    def __init__(self, directory, binding, expected_records, *, verify_sources, root=None):
        self.root = Path(root) if root is not None else ROOT
        directory = Path(directory)
        self.directory = directory if directory.is_absolute() else self.root / directory
        if (self.root.is_symlink() or self.root.resolve() != self.root or not self.root.is_dir()
                or self.directory == self.root or not self.directory.is_relative_to(self.root)):
            raise ArtifactError('unsafe_artifact_scope')
        self._path('')
        if not callable(verify_sources):
            raise ArtifactError('source_verifier_required')
        self.verify_sources = verify_sources
        self._binding_input = binding
        self.binding = normalized(binding)
        if not isinstance(self.binding, dict):
            raise ArtifactError('binding_object_required')
        names = list(expected_records)
        if not names or len(names) != len(set(names)):
            raise ArtifactError('unique_expected_records_required')
        self.expected_records = sorted(name_token(n) for n in names)
        self.intent = {'version': 1, 'binding': self.binding, 'binding_sha256': fingerprint(self.binding),
                       'expected_records': self.expected_records,
                       'expected_records_sha256': fingerprint(self.expected_records)}

    def _path(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ArtifactError('unsafe_relative_artifact_path')
        path = self.directory / relative
        if not path.is_relative_to(self.root):
            raise ArtifactError('artifact_path_escape')
        current = self.root
        for part in path.relative_to(self.root).parts:
            current /= part
            if current.is_symlink():
                raise ArtifactError('artifact_symlink_rejected')
        if path.resolve() != path:
            raise ArtifactError('artifact_path_escape')
        return path

    def _sources(self):
        if normalized(self._binding_input) != self.binding or self.verify_sources() is False:
            raise ArtifactError('source_or_binding_changed')

    def _mkdir(self, path):
        current = self.root
        for part in path.relative_to(self.root).parts:
            current /= part
            if current.is_symlink():
                raise ArtifactError('artifact_symlink_rejected')
            if current.exists():
                if not current.is_dir() or stat.S_IMODE(current.stat().st_mode) != 0o700:
                    raise ArtifactError('private_artifact_directory_required')
            else:
                try:
                    current.mkdir(mode=0o700)
                except FileExistsError:
                    # Independent stages can create the shared stages parent
                    # concurrently; accept only the same private directory.
                    if (current.is_symlink() or not current.is_dir()
                            or stat.S_IMODE(current.stat().st_mode) != 0o700):
                        raise ArtifactError('private_artifact_directory_required') from None

    def _bytes(self, relative):
        path = self._path(relative)
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode) or stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ArtifactError('private_regular_artifact_required')
        return path.read_bytes()

    def _json(self, relative):
        return decode(self._bytes(relative))

    def _recover_link(self, relative):
        """Clean only a completed atomic link's identical named pending file."""
        path, pending = self._path(relative), self._path(str(relative) + '.pending')
        if pending.exists() and path.exists():
            if self._bytes(str(relative) + '.pending') != self._bytes(relative):
                raise ArtifactError('pending_artifact_conflict')
            pending.unlink()
            sync_dir(path.parent)

    def _write_once(self, relative, value):
        path = self._path(relative)
        raw = canonical(value)
        if path.exists():
            if self._bytes(relative) != raw:
                raise ArtifactError('immutable_artifact_changed')
            self._recover_link(relative)
            return
        self._mkdir(path.parent)
        pending = self._path(str(relative) + '.pending')
        if pending.exists():
            self._bytes(str(relative) + '.pending')
            pending.unlink()  # Incomplete, unpublished write; never a sealed result.
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
        sync_dir(path.parent)
        pending.unlink()
        sync_dir(path.parent)

    @contextmanager
    def _lock(self, relative, *, shared=False):
        path = self._path(relative)
        self._mkdir(path.parent)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if (not stat.S_ISREG(os.fstat(fd).st_mode)
                    or stat.S_IMODE(os.fstat(fd).st_mode) != 0o600):
                raise ArtifactError('private_artifact_lock_required')
            try:
                fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ArtifactBusy('artifact_operation_in_progress') from None
            yield
        finally:
            os.close(fd)

    def _initialize(self):
        self._sources()
        if self._path('intent.json').exists():
            if self._json('intent.json') != self.intent:
                raise ArtifactError('run_binding_changed')
            return
        with self._lock('.lock'):
            if self._path('completion.json').exists():
                return
            if not self._path('intent.json').exists():
                if {p.name for p in self.directory.iterdir()} - {'.lock', 'intent.json.pending'}:
                    raise ArtifactError('missing_run_intent')
            self._write_once('intent.json', self.intent)

    def initialize(self):
        """Create intent once before parallel workers; complete runs stay read-only."""
        if self._path('completion.json').exists():
            self.verify_completed()
        else:
            self._initialize()

    def _event(self, stage, phase, error):
        relative = stage + '/events.jsonl'
        kind = type(error).__name__
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,95}', kind) is None:
            kind = 'Exception'
        fd = os.open(self._path(relative), os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if (not stat.S_ISREG(os.fstat(fd).st_mode)
                    or stat.S_IMODE(os.fstat(fd).st_mode) != 0o600):
                raise ArtifactError('private_event_log_required')
            with os.fdopen(fd, 'ab', closefd=False) as stream:
                stream.write(canonical({'event': phase, 'exception_type': kind}))
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
        sync_dir(self._path(stage))

    def _stage_intent(self, name, inputs):
        return {'name': name_token(name), 'inputs': normalized(inputs),
                'inputs_sha256': fingerprint(inputs), 'run_binding_sha256': fingerprint(self.intent)}

    def _save_result(self, stage, intent, result):
        raw = canonical(result)
        receipt = {'result_sha256': hashlib.sha256(raw).hexdigest(),
                   'stage_intent_sha256': fingerprint(intent), 'run_binding_sha256': fingerprint(self.intent)}
        # Persist the body before the independent receipt, then publish with a
        # no-replace link. After the receipt exists, recovery needs no Provider
        # replay (whose cache-observation metadata could legitimately differ).
        pending = self._path(stage + '/result.json.pending')
        if pending.exists():
            self._bytes(stage + '/result.json.pending')
            pending.unlink()  # No receipt can exist on the fresh-call path.
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        sync_dir(pending.parent)
        self._write_once(stage + '/result.seal.json', receipt)
        self._publish_result(stage, intent)

    def _publish_result(self, stage, intent):
        raw = self._bytes(stage + '/result.json.pending')
        expected = {'result_sha256': hashlib.sha256(raw).hexdigest(),
                    'stage_intent_sha256': fingerprint(intent), 'run_binding_sha256': fingerprint(self.intent)}
        if self._json(stage + '/result.seal.json') != expected:
            raise ArtifactError('stage_pending_external_seal_mismatch')
        pending, target = self._path(stage + '/result.json.pending'), self._path(stage + '/result.json')
        os.link(pending, target)
        sync_dir(target.parent)
        pending.unlink()
        sync_dir(target.parent)

    def _cached(self, stage, intent, validator):
        if self._json(stage + '/intent.json') != intent:
            raise ArtifactError('stage_input_or_source_changed')
        receipt = self._json(stage + '/result.seal.json')
        raw = self._bytes(stage + '/result.json')
        expected = {'result_sha256': hashlib.sha256(raw).hexdigest(),
                    'stage_intent_sha256': fingerprint(intent), 'run_binding_sha256': fingerprint(self.intent)}
        if receipt != expected:
            raise ArtifactError('stage_external_seal_mismatch')
        result = decode(raw)
        if result.get('kind') == 'terminal':
            if set(result) != {'kind', 'error_code'}:
                raise ArtifactError('invalid_terminal_result')
            raise TerminalStageError(result['error_code'])
        if set(result) != {'kind', 'raw', 'parsed'} or result['kind'] != 'result':
            raise ArtifactError('invalid_stage_result')
        parsed = normalized(validator(deepcopy(result['raw'])))
        if parsed != result['parsed']:
            raise ArtifactError('cached_reparse_changed')
        self._sources()
        return parsed

    def stage_runner(self, name, inputs, call, validator):
        intent = self._stage_intent(name, inputs)
        stage = 'stages/' + key_path(name)
        self._sources()
        if self._path('completion.json').exists():
            self.verify_completed()
            return self._cached(stage, intent, validator)
        self._initialize()
        with self._lock('.lock', shared=True):
            self._sources()
            if self._path('completion.json').exists():
                self.verify_completed()
                return self._cached(stage, intent, validator)
            with self._lock(stage + '/.lock'):
                return self._run_stage(stage, name, inputs, intent, call, validator)

    def _run_stage(self, stage, name, inputs, intent, call, validator):
        if self._json('intent.json') != self.intent:
            raise ArtifactError('run_binding_changed')
        self._write_once(stage + '/intent.json', intent)
        for filename in ('result.seal.json', 'result.json'):
            self._recover_link(stage + '/' + filename)
        if self._path(stage + '/result.json').exists():
            return self._cached(stage, intent, validator)
        if self._path(stage + '/result.seal.json').exists():
            self._publish_result(stage, intent)
            return self._cached(stage, intent, validator)
        phase = 'call_exception'
        try:
            raw = normalized(call())
            phase = 'validator_exception'
            parsed = normalized(validator(deepcopy(raw)))
        except TerminalStageError as error:
            self._sources()
            if self._stage_intent(name, inputs) != intent:
                raise ArtifactError('stage_inputs_changed_during_call')
            self._save_result(stage, intent, {'kind': 'terminal', 'error_code': error.code})
            raise
        except BaseException as error:
            try:
                self._event(stage, phase, error)
            except Exception:
                pass  # Never replace ProviderStop/KeyboardInterrupt with logging errors.
            raise
        self._sources()
        if self._stage_intent(name, inputs) != intent:
            raise ArtifactError('stage_inputs_changed_during_call')
        self._save_result(stage, intent, {'kind': 'result', 'raw': raw, 'parsed': parsed})
        return parsed

    def _inventory(self):
        files, directories = {}, []
        def walk(path):
            for child in sorted(path.iterdir()):
                relative = child.relative_to(self.directory).as_posix()
                self._path(relative)
                if child.is_dir():
                    if stat.S_IMODE(child.stat().st_mode) != 0o700:
                        raise ArtifactError('private_artifact_directory_required')
                    directories.append(relative)
                    walk(child)
                elif child.is_file():
                    files[relative] = hashlib.sha256(self._bytes(relative)).hexdigest()
                else:
                    raise ArtifactError('nonregular_artifact')
        walk(self.directory)
        return files, directories

    def _validate_stages(self):
        root = self._path('stages')
        if not root.exists():
            return
        for directory in root.iterdir():
            if not directory.is_dir() or re.fullmatch(r'[0-9a-f]{64}', directory.name) is None:
                raise ArtifactError('unexpected_stage_artifact')
            relative = directory.relative_to(self.directory).as_posix()
            expected = {'intent.json', '.lock', 'result.seal.json', 'result.json'}
            names = {p.name for p in directory.iterdir()}
            if names - expected - {'events.jsonl'} or not expected.issubset(names):
                raise ArtifactError('stage_incomplete_or_extra_files')
            intent = self._json(relative + '/intent.json')
            if (intent.get('run_binding_sha256') != fingerprint(self.intent)
                    or key_path(intent.get('name')) != directory.name
                    or intent.get('inputs_sha256') != fingerprint(intent.get('inputs'))):
                raise ArtifactError('stage_intent_changed')
            raw = self._bytes(relative + '/result.json')
            if self._json(relative + '/result.seal.json') != {
                    'result_sha256': hashlib.sha256(raw).hexdigest(), 'stage_intent_sha256': fingerprint(intent),
                    'run_binding_sha256': fingerprint(self.intent)}:
                raise ArtifactError('stage_external_seal_mismatch')

    def finalize(self, records):
        records = normalized(records)
        if not isinstance(records, dict) or set(records) != set(self.expected_records):
            raise ArtifactError('final_record_population_differs')
        if self._path('completion.json').exists():
            saved = self.verify_completed()
            if saved != records:
                raise ArtifactError('completed_records_changed')
            return saved
        self._initialize()
        with self._lock('.lock'):
            self._sources()
            if self._path('completion.json').exists():
                saved = self.verify_completed()
                if saved != records:
                    raise ArtifactError('completed_records_changed')
                return saved
            self._validate_stages()
            for name in self.expected_records:
                self._write_once('records/' + key_path(name) + '.json', {
                    'name': name, 'record': records[name], 'run_binding_sha256': fingerprint(self.intent)})
            if {p.name for p in self._path('records').iterdir()} != {key_path(n) + '.json' for n in self.expected_records}:
                raise ArtifactError('final_record_inventory_differs')
            files, directories = self._inventory()
            if (set(p.split('/')[0] for p in directories) - {'stages', 'records'}
                    or set(p.split('/')[0] for p in files) - {'.lock', 'intent.json', 'stages', 'records',
                        'seal.json', 'seal.json.pending', 'completion.json.pending'}):
                raise ArtifactError('unexpected_run_artifact')
            files = {k: v for k, v in files.items() if k not in ('seal.json', 'seal.json.pending', 'completion.json.pending')}
            self._sources()
            seal = {'version': 1, 'run_binding_sha256': fingerprint(self.intent),
                    'expected_records': self.expected_records, 'files': files, 'directories': directories}
            self._write_once('seal.json', seal)
            self._sources()
            self._write_once('completion.json', {'version': 1, 'run_binding_sha256': fingerprint(self.intent),
                             'seal_sha256': hashlib.sha256(self._bytes('seal.json')).hexdigest(),
                             'record_count': len(self.expected_records)})
        return self.verify_completed()

    def verify_completed(self):
        """Read-only: no lock creation, provider call, reparsing inference or writes."""
        self._sources()
        complete = self._json('completion.json')
        seal_raw = self._bytes('seal.json')
        seal = decode(seal_raw)
        if (complete != {'version': 1, 'run_binding_sha256': fingerprint(self.intent),
                         'seal_sha256': hashlib.sha256(seal_raw).hexdigest(),
                         'record_count': len(self.expected_records)}
                or seal.get('version') != 1 or seal.get('run_binding_sha256') != fingerprint(self.intent)
                or seal.get('expected_records') != self.expected_records or self._json('intent.json') != self.intent):
            raise ArtifactError('completed_source_or_seal_changed')
        files, directories = self._inventory()
        # A crash between the final no-replace hard link and unlink can leave
        # its identical pending inode. Verify it read-only; never repair a
        # completed run or accept an arbitrary extra artifact.
        if 'completion.json.pending' in files:
            if self._bytes('completion.json.pending') != self._bytes('completion.json'):
                raise ArtifactError('completed_pending_marker_changed')
            files.pop('completion.json.pending')
        if (set(files) != set(seal['files']) | {'seal.json', 'completion.json'}
                or directories != seal['directories']
                or any(files[name] != digest for name, digest in seal['files'].items())):
            raise ArtifactError('completed_inventory_or_hash_changed')
        expected_files = {'records/' + key_path(n) + '.json' for n in self.expected_records}
        if {name for name in files if name.startswith('records/')} != expected_files:
            raise ArtifactError('completed_record_population_changed')
        records = {}
        for name in self.expected_records:
            value = self._json('records/' + key_path(name) + '.json')
            if set(value) != {'name', 'record', 'run_binding_sha256'} or value['name'] != name or value['run_binding_sha256'] != fingerprint(self.intent):
                raise ArtifactError('completed_record_binding_changed')
            records[name] = value['record']
        self._sources()
        return records
