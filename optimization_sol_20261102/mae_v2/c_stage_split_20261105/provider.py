"""Fixed dual-provider HTTP adapter, resumable response cache, no corpus embeddings."""
from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
from datetime import datetime, timezone

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from config import MODELS, STAGE_MODELS, CHAT_BASE_URL, EMBEDDING_BASE_URL
CHAT_URL = CHAT_BASE_URL + '/chat/completions'
EMBED_URL = EMBEDDING_BASE_URL + '/embeddings'
CHAT_MODEL = MODELS['llm']
VISION_MODEL = MODELS['llm_vision']
EMBED_MODEL = MODELS['embedding']


class ProviderStop(BaseException):
    """System/provider faults must not be converted into a successful fallback batch."""


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def scoped(path):
    path = Path(path).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError('Output is outside the isolated optimization directory')
    return path


def atomic_json(path, value):
    path = scoped(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(threading.get_ident()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sanitize(value, keys):
    if isinstance(value, dict):
        return {sanitize(str(k), keys): sanitize(v, keys) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, keys) for v in value]
    if isinstance(value, str):
        for key in keys:
            value = value.replace(key, '[REDACTED]')
            head, tail = re.escape(key[:8]), re.escape(key[-4:])
            value = re.sub(rf'(?<![A-Za-z0-9_-]){head}[^\s\x22\x27]*{tail}(?![A-Za-z0-9_-])',
                           '[REDACTED]', value)
            # Match known credential prefixes only when an explicit truncation
            # marker follows. Do not replace short suffixes or arbitrary text
            # that merely shares a prefix with a credential. Include JSON's
            # escaped ellipsis because chat content itself can be JSON text.
            prefixes = '|'.join(re.escape(key[:n]) for n in range(len(key) - 1, 7, -1))
            if prefixes:
                value = re.sub(rf'(?<![A-Za-z0-9_-])(?:{prefixes})(?:\.{{3,}}|…+|(?:\\u2026)+|\*{{3,}})',
                               '[REDACTED]', value)
        value = re.sub(r'sk-[A-Za-z0-9_-]{8,}', '[REDACTED]', value)
        value = re.sub(r'[^\s\x22\x27]*\*{3,}[^\s\x22\x27.,;]*', '[REDACTED]', value)
    return value


def read_credentials():
    from dotenv import dotenv_values
    path = HERE / 'private.env'
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ProviderStop('Private local credential file missing or permissions too broad')
    values = dotenv_values(path, interpolate=False)
    keys = tuple(values.get(k) for k in ('OPENAI_API_KEY','EMBEDDING_API_KEY','VISION_API_KEY'))
    if any(not k or '$' in k for k in keys):
        raise ProviderStop('Three local role credentials required')
    return keys


def extract_unique_json(text):
    """Extract one exact object substring; ambiguous/array/truncated text stays raw."""
    if not isinstance(text, str):
        return text
    try:
        json.loads(text)
        return text  # Never extract from any other valid JSON value, e.g. arrays.
    except ValueError:
        pass
    start = text.find('{')
    if start < 0:
        return text
    try:
        value, end = json.JSONDecoder().raw_decode(text, start)
    except ValueError:
        return text
    if not isinstance(value, dict):
        return text
    outside = (text[:start], text[end:])
    if any(re.search(r'[{}\[\]]', part) for part in outside):
        return text  # Also rejects incomplete outer arrays and multiple objects.
    for part in outside:
        stripped = re.sub(r'```(?:json)?', '', part).strip()
        if stripped:
            try:
                json.loads(stripped)
            except ValueError:
                pass
            else:
                return text
    return text[start:end]


def stage_model(messages, dataset=None, stage=None):
    images = any(isinstance(m.get('content'), list) and any(
        b.get('type') == 'image_url' for b in m['content']) for m in messages)
    if dataset is not None and dataset not in ('nutrition5k', 'asa24'):
        raise ProviderStop('Unknown dataset for stage route')
    if images:
        if dataset not in ('nutrition5k', 'asa24') or stage not in STAGE_MODELS:
            raise ProviderStop('Image input requires an explicit dataset and visual stage')
        selected = STAGE_MODELS[stage]
        model = selected[dataset] if isinstance(selected, dict) else selected
    else:
        if stage not in (None, 'query_rewrite'):
            raise ProviderStop('Text-only input requires query_rewrite stage')
        model = CHAT_MODEL
    if model not in (CHAT_MODEL, VISION_MODEL):
        raise ProviderStop('Unsupported stage model')
    return model, images


class Provider:
    def __init__(self, run_dir, allow_paid=False, namespace='development', keys=None,
                 transport=None, timeout=120, sleep=time.sleep):
        if not allow_paid:
            raise ProviderStop('Explicit --allow-paid is required')
        self.run_dir = scoped(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keys = tuple(keys) if keys is not None else read_credentials()
        if len(self.keys) != 3 or not all(self.keys):
            raise ProviderStop('Three role credentials are required')
        self.namespace, self.sleep = namespace, sleep
        self.cache = scoped(ROOT / 'runtime_cache' / 'responses')
        self.lock, self.key_locks = threading.Lock(), {}
        self.client = httpx.Client(transport=transport, trust_env=False, follow_redirects=False,
                                   timeout=httpx.Timeout(timeout, connect=30))

    def close(self):
        self.client.close()

    def _log(self, record):
        with self.lock:
            with (self.run_dir / 'requests.jsonl').open('a') as f:
                f.write(json.dumps(sanitize(record, self.keys), allow_nan=False) + '\n')
                f.flush()

    def _request(self, role, payload, *, route=None):
        model = payload.get('model')
        allowed = (CHAT_MODEL, VISION_MODEL) if role == 'chat' else (EMBED_MODEL,)
        url = CHAT_URL if role == 'chat' else EMBED_URL
        key = self.keys[2 if model == VISION_MODEL else 0] if role == 'chat' else self.keys[1]
        if role == 'chat':
            if not isinstance(route, dict) or set(route) != {'dataset', 'stage'}:
                raise ProviderStop('Explicit chat route metadata required')
            expected, image_input = stage_model(payload.get('messages', []), **route)
        elif role == 'embedding' and route is None:
            expected, image_input = EMBED_MODEL, False
        else:
            raise ProviderStop('Invalid request role or route')
        if model not in allowed or model != expected:
            raise ProviderStop('Requested model violates the provider route')
        # Embeddings are reusable across runs; chat namespace separates formal and development runs.
        cache_id = fingerprint({'url': url, 'payload': payload,
                                'namespace': self.namespace if role == 'chat' else 'official-query-v1'})
        cache_path = self.cache / role / (cache_id + '.json')
        with self.lock:
            key_lock = self.key_locks.setdefault(cache_id, threading.Lock())
        with key_lock:
            inherited = ROOT.parent / 'runtime_cache/responses/embedding' / (cache_id + '.json')
            if role == 'embedding' and not cache_path.exists() and inherited.is_file():
                saved = json.loads(inherited.read_text())
                if (saved.get('request_hash') != cache_id or saved.get('model') != model
                        or saved.get('value_hash') != fingerprint(saved.get('value'))
                        or not self._valid_vector(saved.get('value'))):
                    raise ProviderStop('Inherited official embedding cache provenance mismatch')
                atomic_json(cache_path, saved)
            if cache_path.exists():
                saved = json.loads(cache_path.read_text())
                if saved.get('request_hash') != cache_id or saved.get('model') != model:
                    raise ProviderStop('Response cache provenance mismatch')
                value = saved.get('value')
                if saved.get('value_hash') != fingerprint(value):
                    raise ProviderStop('Response cache content checksum mismatch')
                if role == 'chat' and not isinstance(value, str):
                    raise ProviderStop('Invalid cached chat value')
                if role == 'embedding' and not self._valid_vector(value):
                    raise ProviderStop('Invalid cached embedding value')
                return value
            for _ in range(3):
                # Reserve before sending. A crash consumes an attempt; restarting
                # cannot silently reset the three-transmission protocol bound.
                attempt = self._reserve_attempt(cache_id, role) - 1
                record = {'started': stamp(), 'role': role, 'endpoint': url, 'model_requested': model,
                          'request_hash': cache_id, 'attempt': attempt + 1, 'input_has_images': image_input,
                          'dataset': (route or {}).get('dataset'), 'stage': (route or {}).get('stage')}
                try:
                    response = self.client.post(url, json=payload,
                                                headers={'Authorization': 'Bearer ' + key})
                except httpx.HTTPError as exc:
                    record.update(status=None, error_type=type(exc).__name__)
                    self._log(record)
                    if attempt == 2:
                        raise ProviderStop('Provider connection failed after three attempts; resume pending work') from None
                    self.sleep((2, 8)[attempt])
                    continue
                record['status'] = response.status_code
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                record.update(model_served=body.get('model'), usage=body.get('usage'))
                if response.status_code != 200:
                    error = body.get('error')
                    record['error_type'] = error.get('type') if isinstance(error, dict) else None
                    self._log(record)
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < 2:
                            self.sleep((2, 8)[attempt])
                            continue
                    raise ProviderStop('Provider HTTP ' + str(response.status_code) + '; batch stopped')
                if body.get('model') != model:
                    self._log(record)
                    raise ProviderStop('Provider-reported model differs from the requested model')
                if role == 'chat':
                    choices = body.get('choices') or []
                    if not isinstance(choices, list) or any(not isinstance(c, dict) for c in choices):
                        self._log(record)
                        raise ProviderStop('Malformed chat response envelope')
                    if any(not isinstance(c.get('message'), dict) for c in choices):
                        self._log(record)
                        raise ProviderStop('Malformed chat message envelope')
                    record['finish_reason'] = choices[0].get('finish_reason') if len(choices) == 1 else None
                    content = (choices[0].get('message') or {}).get('content') if len(choices) == 1 else None
                    # Incomplete content enters the same one-repair contract as malformed JSON.
                    value = content if isinstance(content, str) and record['finish_reason'] == 'stop' else ''
                    value = sanitize(value, self.keys)
                else:
                    data = body.get('data') or []
                    if not isinstance(data, list) or any(not isinstance(d, dict) for d in data):
                        self._log(record)
                        raise ProviderStop('Malformed embedding response envelope')
                    value = data[0].get('embedding') if len(data) == 1 else None
                    if not self._valid_vector(value):
                        self._log(record)
                        raise ProviderStop('Official query embedding is incompatible with the frozen index')
                self._log(record)
                atomic_json(cache_path, {'request_hash': cache_id, 'model': model, 'value': value,
                                         'value_hash': fingerprint(value)})
                return value
        raise ProviderStop('Unexpected provider execution state')

    def _reserve_attempt(self, cache_id, role):
        directory = scoped(ROOT / 'runtime_cache/attempts' / role)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (cache_id + '.json')
        with (directory / (cache_id + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(path.read_text()) if path.exists() else {
                'request_hash': cache_id, 'attempts': 0}
            if state.get('request_hash') != cache_id or type(state.get('attempts')) is not int:
                raise ProviderStop('Attempt ledger provenance mismatch')
            count = state['attempts']
            if not 0 <= count < 3:
                raise ProviderStop('Payload has exhausted its three persistent HTTP attempts')
            atomic_json(path, {'request_hash': cache_id, 'attempts': count + 1,
                               'last_reserved': stamp(), 'namespace': self.namespace})
            return count + 1

    @staticmethod
    def _valid_vector(value):
        return (isinstance(value, list) and len(value) == 3072
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) for v in value) and any(value))

    def chat(self, messages, max_completion_tokens=4096, *, dataset=None, stage=None):
        # Images must be local bytes; the provider must not fetch arbitrary user-controlled URLs.
        for message in messages:
            content = message.get('content')
            if isinstance(content, list):
                for block in content:
                    if block.get('type') == 'image_url':
                        url = block.get('image_url', {}).get('url', '')
                        if not url.startswith(('data:image/png;base64,', 'data:image/jpeg;base64,')):
                            raise ProviderStop('Only inline local image bytes may be sent')
        if max_completion_tokens not in (4096, 8192):
            raise ProviderStop('Unregistered completion-token limit')
        model, has_images = stage_model(messages, dataset, stage)
        route = {'dataset': dataset, 'stage': stage or 'query_rewrite'}
        payload = {'model': model, 'temperature': 1,
                   'max_completion_tokens': max_completion_tokens, 'messages': messages}
        raw = self._request('chat', payload, route=route)
        value = extract_unique_json(raw) if model == VISION_MODEL else raw
        if value != raw:
            audit = {'utc': stamp(), 'adapter': 'unique_JSON_object_exact_substring_v1',
                'model': model, 'raw_sha256': fingerprint(raw), 'extracted_sha256': fingerprint(value),
                'request_hash': fingerprint({'url': CHAT_URL, 'payload': payload, 'namespace': self.namespace})}
            with self.lock:
                with (self.run_dir / 'response_adapters.jsonl').open('a') as stream:
                    stream.write(json.dumps(audit) + '\n')
        return value

    def embed(self, texts):
        if not isinstance(texts, list) or not texts or len(texts) > 12:
            raise ProviderStop('Only bounded query embedding batches are supported')
        result = []
        for text in texts:
            if not isinstance(text, str) or not text.strip() or len(text) > 12000:
                raise ProviderStop('Invalid retrieval query')
            result.append(self._request('embedding', {'model': EMBED_MODEL,
                          'input': [text], 'encoding_format': 'float'}))
        return result


def usage(run_dir):
    totals = {k: {'requests': 0, 'errors': 0, 'prompt_tokens': 0,
                  'completion_tokens': 0, 'reasoning_tokens': 0, 'total_tokens': 0}
              for k in ('chat', 'embedding')}
    path = Path(run_dir) / 'requests.jsonl'
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        out = totals[row['role']]
        out['requests'] += 1
        out['errors'] += row.get('status') != 200
        tokens = row.get('usage') or {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            out[key] += tokens.get(key, 0) or 0
        out['reasoning_tokens'] += (tokens.get('completion_tokens_details') or {}).get('reasoning_tokens', 0) or 0
    return {'providers': totals, 'third_party_cost_usd': None,
            'official_embedding_estimated_usd': totals['embedding']['prompt_tokens'] * .13 / 1e6,
            'embedding_price_checked': '2026-11-02',
            'cost_notice': 'Third-party price unverified; reasoning tokens are included in completion tokens.'}


def prepare_vectors():
    """Copy portable saved float32 vectors; no native index mutation or API call."""
    import numpy as np
    output = scoped(ROOT / 'data' / 'vectors')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if any(sha(output / name) != value for name, value in saved['files'].items()):
            raise ValueError('Portable vector checksum mismatch')
        return saved
    source = ROOT.parent / '.demo' / 'assets'
    if not source.is_dir():
        raise ValueError('Saved portable runtime missing; install the pinned demo runtime first')
    bundle_manifest = json.loads((source / 'manifest.json').read_text())
    for name in ('embeddings.npy', 'collection.json'):
        if sha(source / name) != bundle_manifest['files'][name]:
            raise ValueError('Source runtime file checksum mismatch')
        shutil.copyfile(source / name, output / name)
    matrix = np.load(output / 'embeddings.npy', allow_pickle=False)
    descriptor = json.loads((output / 'collection.json').read_text())
    if matrix.shape != (5624, 3072) or not np.isfinite(matrix).all():
        raise ValueError('Saved vector shape or values are invalid')
    codes = []
    for doc in descriptor['documents']:
        match = re.search(r'Food code: (\d{8})(?!\d)', doc)
        if not match:
            raise ValueError('Saved vector has no exact food-code association')
        codes.append(match.group(1))
    if len(set(codes)) != 5624:
        raise ValueError('Saved vector food codes are not unique')
    atomic_json(output / 'codes.json', codes)
    saved = {'vectors': 5624, 'dimensions': 3072, 'api_calls': 0,
             'embedding_model': EMBED_MODEL,
             'files': {name: sha(output / name) for name in ('embeddings.npy', 'collection.json', 'codes.json')}}
    atomic_json(manifest_path, saved)
    return saved
