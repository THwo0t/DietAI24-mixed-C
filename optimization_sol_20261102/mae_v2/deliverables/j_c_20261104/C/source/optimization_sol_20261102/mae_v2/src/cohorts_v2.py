"""Blind, offline MAE-v2 linkage audit and immutable cohort preparation.

Only this module may mechanically inspect sealed source labels/images during
preparation. Its CLI emits aggregate counts and hashes, never records or values.
Development callers use verify_dev_only/load_dev, which never open validation.
No network/client/credential imports. New images require a separately authorized
download of the exact private proposal; this module never downloads anything.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from PIL import Image

V2 = Path(__file__).resolve().parents[1]
SEED = 'dietai24-mae-v2-20261102'
WINDOW_SECONDS = 600
MAX_IMAGE_BYTES = 5_000_000
BASE_URL = 'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead'
DISH = re.compile(r'dish_[0-9]{10}\Z')
INGREDIENT = re.compile(r'ingr_[0-9]{10}\Z')
TARGETS = {'weight': 'total_mass', 'energy': 'total_calories', 'fat': 'total_fat',
           'carbs': 'total_carbs', 'protein': 'total_protein'}
RULES = {
    'version': 2, 'seed': SEED, 'window_seconds': WINDOW_SECONDS,
    'mass': 'sum positive per-ingredient Decimal grams; zero masses give no anchors',
    'same_vector': 'same cafe and identical complete positive mass vectors',
    'exact_extension': 'same cafe, 0<dt<=600, earlier >=2 ingredients, every earlier mass identical, at least one new ingredient',
    'adjacent_extension': 'adjacent full raw cafe sequence, 0<dt<=600, >=2 shared IDs and >=1 identical positive mass, all earlier IDs retained and nondecreasing',
    'ambiguity': 'same adjacent/time/shared/anchor eligibility, any earlier ID missing or decreasing; quarantine endpoint components after full closure',
    'images': 'equal file SHA256 or equal dimensions and decoded RGB SHA256',
    'interpretation': 'engineering leakage components; not certified physical plate identities',
}


class AuditError(RuntimeError):
    """Stable non-sensitive error code; never include source row contents."""


def fail(code):
    raise AuditError(code)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rank(kind, value):
    return digest((SEED + '|' + kind + '|' + value).encode())


def child(base, relative):
    base = Path(base).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        fail('unsafe_relative_path')
    path = base / relative
    if path.is_symlink() or not path.resolve().is_relative_to(base):
        fail('unsafe_relative_path')
    return path


def frozen_write(base, relative, data, *, private=False):
    path = child(base, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        os.chmod(path.parent, 0o700)
    if path.exists():
        if path.read_bytes() != data:
            fail('immutable_artifact_conflict')
        if private and path.stat().st_mode & 0o077:
            fail('unsafe_private_permissions')
        return
    fd, temporary = tempfile.mkstemp(prefix='.cohorts-v2-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
        os.chmod(temporary, 0o600 if private else 0o644)
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def decimal_mass(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        fail('invalid_mass')
    if not result.is_finite() or result < 0:
        fail('invalid_mass')
    return result


@dataclass(frozen=True)
class Scan:
    dish_id: str
    cafe: str
    timestamp: int
    masses: tuple
    recipe_hash: str = ''


def make_scan(dish_id, cafe, pairs, recipe_hash=''):
    if not DISH.fullmatch(dish_id):
        fail('invalid_dish_id')
    masses = defaultdict(Decimal)
    for ingredient, grams in pairs:
        if not INGREDIENT.fullmatch(ingredient):
            fail('invalid_ingredient_id')
        value = decimal_mass(grams)
        if value > 0:
            masses[ingredient] += value
    return Scan(dish_id, cafe, int(dish_id[5:]), tuple(sorted(masses.items())), recipe_hash)


def parse_metadata(paths):
    scans = {}
    for path in paths:
        cafe = Path(path).stem
        with Path(path).open(newline='', encoding='utf-8-sig') as handle:
            for row in csv.reader(handle):
                if len(row) < 6 or not DISH.fullmatch(row[0]):
                    fail('invalid_metadata_row')
                # Published raw CSVs use six dish fields, then seven fields per
                # ingredient. A declared-count variant is accepted only when
                # the complete seven-field structure and count both validate.
                start, declared = 6, None
                if len(row) > 6 and not row[6].startswith('ingr_'):
                    try:
                        declared = int(row[6])
                    except ValueError:
                        fail('invalid_ingredient_structure')
                    start = 7
                if (len(row) - start) % 7:
                    fail('invalid_ingredient_structure')
                chunks = [row[i:i + 7] for i in range(start, len(row), 7)]
                if declared is not None and declared != len(chunks):
                    fail('ingredient_count_mismatch')
                pairs = [(chunk[0], chunk[2]) for chunk in chunks]
                recipe = digest(', '.join(sorted(chunk[1] for chunk in chunks)).encode())
                scan = make_scan(row[0], cafe, pairs, recipe)
                if scan.dish_id in scans:
                    fail('duplicate_raw_dish_id')
                scans[scan.dish_id] = scan
    return scans


class Groups:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        if node not in self.parent:
            fail('unknown_graph_node')
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while node != root:
            node, self.parent[node] = self.parent[node], root
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_components(scans, fingerprints=()):
    """Pure complete-universe linkage; fingerprints contain no image bytes."""
    graph = Groups(scans)
    edge_counts = Counter()
    ambiguous = set()
    cafes = defaultdict(list)
    for scan in scans.values():
        cafes[scan.cafe].append(scan)
    for records in cafes.values():
        records.sort(key=lambda s: (s.timestamp, s.dish_id))
        seen_vectors = {}
        for scan in records:
            # Empty scans are graph nodes but provide no positive-food evidence.
            if scan.masses:
                if scan.masses in seen_vectors:
                    graph.union(scan.dish_id, seen_vectors[scan.masses])
                    edge_counts['same_complete_mass_vector'] += 1
                else:
                    seen_vectors[scan.masses] = scan.dish_id
        for index, earlier in enumerate(records):
            old = dict(earlier.masses)
            if len(old) < 2:
                continue
            for later in records[index + 1:]:
                delta = later.timestamp - earlier.timestamp
                if delta > WINDOW_SECONDS:
                    break
                if delta <= 0:
                    continue
                new = dict(later.masses)
                if old.keys() < new.keys() and all(new[k] == v for k, v in old.items()):
                    graph.union(earlier.dish_id, later.dish_id)
                    edge_counts['exact_increment'] += 1
        for earlier, later in zip(records, records[1:]):
            if not 0 < later.timestamp - earlier.timestamp <= WINDOW_SECONDS:
                continue
            old, new = dict(earlier.masses), dict(later.masses)
            shared = old.keys() & new.keys()
            if len(shared) < 2 or not any(old[k] == new[k] for k in shared):
                continue
            if old.keys() <= new.keys() and all(new[k] >= v for k, v in old.items()):
                if old != new:
                    graph.union(earlier.dish_id, later.dish_id)
                    edge_counts['adjacent_increment'] += 1
            else:
                ambiguous.update((earlier.dish_id, later.dish_id))
                edge_counts['ambiguous_adjacent_pairs'] += 1
    seen_hashes = {}
    for item in fingerprints:
        dish = item['dish_id']
        if dish not in scans:
            fail('image_missing_metadata')
        for kind in ('bytes_sha256', 'rgb_sha256'):
            key = (kind, item[kind])
            if key in seen_hashes and seen_hashes[key] != dish:
                graph.union(dish, seen_hashes[key])
                edge_counts[kind] += 1
            else:
                seen_hashes[key] = dish
    members = defaultdict(list)
    for dish in sorted(scans):
        members[graph.find(dish)].append(dish)
    names = {root: 'nutrition5k-v2-' + digest('\n'.join(ids).encode())
             for root, ids in members.items()}
    mapping = {dish: names[graph.find(dish)] for dish in scans}
    quarantine = {mapping[dish] for dish in ambiguous}
    return mapping, quarantine, dict(edge_counts), {names[r]: ids for r, ids in members.items()}


def image_fingerprint(path, dish_id):
    try:
        with Image.open(path) as image:
            image.load()
            if image.format not in ('PNG', 'JPEG'):
                fail('unsupported_image_format')
            rgb = image.convert('RGB')
            payload = str(rgb.size).encode() + b'\0' + rgb.tobytes()
            dimensions = list(rgb.size)
            image_format, image_mode = image.format, image.mode
    except AuditError:
        raise
    except Exception:
        fail('image_decode_failed')
    return {'dish_id': dish_id, 'bytes_sha256': sha256(path),
            'rgb_sha256': digest(payload), 'bytes': Path(path).stat().st_size,
            'dimensions': dimensions, 'format': image_format, 'mode': image_mode}


def five_folds(groups):
    ordered = sorted(set(groups), key=lambda g: (rank('fold', g), g))
    if len(ordered) < 5:
        fail('fewer_than_five_development_groups')
    return {group: index % 5 for index, group in enumerate(ordered)}


def select_round_robin(retained, candidates, mapping, target=200):
    """Keep clean originals, then prefer unseen/least represented components."""
    if len(retained) > target or len(retained) != len(set(retained)):
        fail('invalid_retained_selection')
    selected = list(retained)
    counts = Counter(mapping[dish] for dish in selected)
    pool = defaultdict(list)
    for dish in set(candidates) - set(selected):
        pool[mapping[dish]].append(dish)
    for group in pool:
        pool[group].sort(key=lambda dish: (rank('image', dish), dish))
    while len(selected) < target:
        available = [group for group, rows in pool.items() if rows]
        if not available:
            fail('fewer_than_200_eligible_validation_images')
        group = min(available, key=lambda g: (counts[g], rank('component', g), g))
        selected.append(pool[group].pop(0))
        counts[group] += 1
    return selected


def _rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _sample_dish(row):
    value = row.get('dish_id', '')
    if not value:
        value = row.get('sample_id', '').removeprefix('nutrition5k-')
    if DISH.fullmatch(value):
        return value
    image = row.get('image_path', '')
    return _path_dish(Path(image)) if image else None


def _path_dish(path):
    for value in (path.stem, *reversed(path.parts[:-1])):
        match = re.match(r'(dish_[0-9]{10})(?:\Z|[_-])', value)
        if match:
            return match[1]
    return None


def _truth(row):
    result = {key: float(decimal_mass(row[column])) for key, column in TARGETS.items()}
    if not all(math.isfinite(value) for value in result.values()):
        fail('nonfinite_target')
    return {'truth': result}


def _line_payload(rows):
    return ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows).encode()


def prepare(*, root=None, v2_root=None, audit_only=False, proposal_only=False):
    """Audit all sources; prepare dev and private proposal; never download.

    A proposal persists exact replacement IDs/URLs in a 0600 private file. When
    every proposed incoming image exists, rerunning without proposal_only seals
    the final validation cohort. New image evidence that changes selection is a
    hard stop, never a silent replacement.
    """
    v2 = Path(v2_root or V2).resolve()
    repo = Path(root or v2.parents[1]).resolve()
    v1 = repo / 'optimization_sol_20261102'
    sources = {}
    expected_sources = {}

    def source(path):
        path = Path(path)
        if not path.is_file():
            fail('required_source_missing')
        sources[str(path)] = sha256(path)
        if str(path) in expected_sources and sources[str(path)] != expected_sources[str(path)]:
            fail('v1_original_source_mismatch')
        return path

    # Mechanically verify frozen artifacts without importing/running v1 code.
    for base, relative in ((v1 / 'data', 'freeze/protocol.json'),
                           (v1, 'data/cohorts/protocol.json')):
        protocol = json.loads(source(base / relative).read_text())
        for name, expected in protocol.get('source_sha256', {}).items():
            if name in expected_sources and expected_sources[name] != expected:
                fail('conflicting_v1_source_seals')
            expected_sources[name] = expected
        for name, expected in protocol['frozen_sha256'].items():
            if sha256(source(child(base, name))) != expected:
                fail('v1_frozen_source_mismatch')
    metadata_paths = [source(repo / 'Nutrition5k/metadata' / name) for name in
                      ('dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv')]
    scans = parse_metadata(metadata_paths)
    available = {}
    for row in _rows(source(repo / 'Nutrition5k/metadata/dish_metadata_available.csv')):
        dish = row['dish_id']
        if dish not in scans or dish in available:
            fail('available_frame_metadata_mismatch')
        truth = _truth(row)['truth']
        if all(truth[key] == 0 for key in ('energy', 'fat', 'carbs', 'protein')) and truth['weight'] > 0:
            continue
        if scans[dish].masses:
            available[dish] = row
    original = {split: _jsonl(source(v1 / f'data/cohorts/{split}.jsonl'))
                for split in ('dev', 'validation')}
    expected = {'dev': {'asa24': 99, 'nutrition5k': 100},
                'validation': {'asa24': 78, 'nutrition5k': 200}}
    for split, rows in original.items():
        if dict(Counter(row['dataset'] for row in rows)) != expected[split]:
            fail('v1_cohort_count_mismatch')
    original_dev = {_sample_dish(row) for row in original['dev'] if row['dataset'] == 'nutrition5k'}
    original_val = {_sample_dish(row) for row in original['validation'] if row['dataset'] == 'nutrition5k'}
    if None in original_dev | original_val or original_dev & original_val:
        fail('invalid_v1_dish_ids')
    old = repo.parent / 'DietAI24/data'
    snapshot = repo.parent / 'DietAI24_mac_snapshot_20261022/data'
    history_sources = [repo / 'dish_metadata.csv', repo / 'dish_metadata_dev.csv',
        old / 'reproduce_manifests/paper50_all.csv', snapshot / 'reproduce_manifests/paper50_all.csv',
        snapshot / 'nutrition5k/smoke_manifest.csv']
    history_sources += sorted((repo / 'runs').glob('*/nutrition5k_food_codes.csv'))
    historical = set()
    for path in history_sources:
        for row in _rows(source(path)):
            if row.get('dataset', '').lower() not in ('', 'nutrition5k'):
                continue
            dish = _sample_dish(row)
            if dish:
                historical.add(dish)
    registry = json.loads(source(v1 / 'data/freeze/exclusion_registry.json').read_text())
    historical.update(registry['exposed_dish_ids'])
    historical.update(registry['local_dish_ids'])
    fingerprints, image_paths, by_path = [], {}, {}

    def inspect_image(path, dish):
        path = Path(path).resolve()
        if path not in by_path:
            fp = image_fingerprint(source(path), dish)
            by_path[path] = fp
            fingerprints.append(fp)
        fp = by_path[path]
        if fp['dish_id'] != dish:
            fail('image_path_has_multiple_dish_ids')
        image_paths.setdefault(dish, path)
        return fp

    legacy_roots = [repo / 'Nutrition5k', old, snapshot,
        Path('/home/lelouch/Documents/Codex/DietAI24_Claude_Handoff_20261023/workspace/work/audit_data')]
    image_inventory = []
    for base in legacy_roots:
        count = 0
        if base.exists():
            for path in sorted(base.rglob('*')):
                if path.suffix.lower() not in ('.png', '.jpg', '.jpeg') or not path.is_file():
                    continue
                dish = _path_dish(path)
                if dish:
                    historical.add(dish)
                    inspect_image(path, dish)
                    count += 1
        image_inventory.append({'root': str(base), 'exists': base.exists(), 'images': count})
    v1_image_sizes = []
    for split, rows in original.items():
        for row in rows:
            path = child(v1, row['image_path'])
            if row['dataset'] != 'nutrition5k':
                if sha256(source(path)) != row['image_sha256']:
                    fail('v1_image_hash_mismatch')
                continue
            dish = _sample_dish(row)
            fp = inspect_image(path, dish)
            if fp['bytes_sha256'] != row['image_sha256']:
                fail('v1_image_hash_mismatch')
            v1_image_sizes.append(fp['bytes'])
    if not historical <= scans.keys() or not (original_dev | original_val) <= scans.keys():
        fail('exposure_missing_raw_metadata')
    proposal_path = child(v2, 'data/freeze/cohort_proposal.json')
    previous = json.loads(proposal_path.read_text()) if proposal_path.exists() else None
    if previous:
        if previous['rules'] != RULES:
            fail('existing_proposal_rules_mismatch')
        for entry in previous['downloads']:
            path = child(v2, entry['destination'])
            if path.exists():
                fp = inspect_image(path, entry['dish_id'])
                if (fp['dimensions'] != [640, 480] or fp['format'] != 'PNG'
                        or fp['mode'] != 'RGB' or fp['bytes'] > MAX_IMAGE_BYTES):
                    fail('incoming_image_format_or_size_mismatch')
    mapping, quarantine, edges, members = build_components(scans, fingerprints)
    historical_groups = {mapping[dish] for dish in historical}
    exposed_dev_groups = {mapping[dish] for dish in original_dev}
    old_recipe_hashes = {scans[dish].recipe_hash for dish in historical}
    old_recipe_hashes.update(registry['exposed_ingredient_group_sha256'])
    by_dish_fp = {dish: by_path[path] for dish, path in image_paths.items()}
    active_dev, removed_dev = [], Counter()
    seen_dev_pixels = set()
    for dish in sorted(original_dev, key=lambda d: (rank('image', d), d)):
        if mapping[dish] in historical_groups:
            removed_dev['historical_component'] += 1
        elif mapping[dish] in quarantine:
            removed_dev['ambiguous_component'] += 1
        elif by_dish_fp[dish]['rgb_sha256'] in seen_dev_pixels:
            removed_dev['identical_rgb'] += 1
        else:
            active_dev.append(dish)
            seen_dev_pixels.add(by_dish_fp[dish]['rgb_sha256'])
    n5_folds = five_folds(mapping[dish] for dish in active_dev)
    blocked = historical_groups | exposed_dev_groups | quarantine
    eligible = {dish for dish in available if mapping[dish] not in blocked
                and scans[dish].recipe_hash not in old_recipe_hashes}
    retained, retired = [], Counter()
    seen_val_pixels = set()
    for dish in sorted(original_val, key=lambda d: (rank('image', d), d)):
        if mapping[dish] in historical_groups:
            retired['historical_component'] += 1
        elif mapping[dish] in exposed_dev_groups:
            retired['original_development_component'] += 1
        elif mapping[dish] in quarantine:
            retired['ambiguous_component'] += 1
        elif dish not in eligible:
            retired['legacy_eligibility_or_recipe_exclusion'] += 1
        elif by_dish_fp[dish]['rgb_sha256'] in seen_val_pixels:
            retired['identical_rgb'] += 1
        else:
            retained.append(dish)
            seen_val_pixels.add(by_dish_fp[dish]['rgb_sha256'])
    # Retired validation remains retired, including exact-pixel duplicate rows.
    replacement_candidates = eligible - original_val - original_dev
    replacement_candidates = {dish for dish in replacement_candidates
        if dish not in by_dish_fp or by_dish_fp[dish]['rgb_sha256'] not in seen_val_pixels}
    selected = select_round_robin(retained, replacement_candidates, mapping)
    if previous and previous['selected_dish_ids'] != selected:
        fail('new_image_evidence_changed_frozen_proposal')
    selected_pixel_hashes = [by_dish_fp[d]['rgb_sha256'] for d in selected if d in by_dish_fp]
    if len(selected_pixel_hashes) != len(set(selected_pixel_hashes)):
        fail('new_image_exact_duplicate_requires_new_authorization')
    downloads = [{'dish_id': dish, 'source_url': f'{BASE_URL}/{dish}/rgb.png',
                  'destination': f'data/incoming/{rank("incoming", dish)}/rgb.png',
                  'max_bytes': MAX_IMAGE_BYTES}
                 for dish in selected if dish not in image_paths]
    selected_groups = Counter(mapping[dish] for dish in selected)
    summary = {
        'status': 'awaiting_replacement_download' if downloads else 'all_images_local',
        'rules_sha256': digest(json_bytes(RULES)), 'raw_scans': len(scans),
        'metadata_components': len(members), 'ambiguous_components': len(quarantine),
        'link_evidence_counts': edges, 'historical_exposed_scans': len(historical),
        'development': {'asa24_images': 99, 'asa24_groups': 23,
            'nutrition5k_images': len(active_dev), 'nutrition5k_groups': len(n5_folds),
            'nutrition5k_retired_counts': dict(removed_dev)},
        'validation': {'asa24_images': 78, 'asa24_groups': 23,
            'nutrition5k_target_images': 200, 'nutrition5k_retained_original_images': len(retained),
            'nutrition5k_retired_counts': dict(retired), 'nutrition5k_groups': len(selected_groups),
            'nutrition5k_group_size_histogram': dict(sorted(Counter(selected_groups.values()).items())),
            'nutrition5k_max_group_fraction': max(selected_groups.values()) / 200},
        'downloads': {'exact_file_count': len(downloads),
            'estimated_bytes': math.ceil(len(downloads) * sum(v1_image_sizes) / len(v1_image_sizes)),
            'estimate_basis': 'mean encoded byte size of the 300 existing v1 RGB images',
            'maximum_file_bytes': MAX_IMAGE_BYTES, 'maximum_total_bytes': len(downloads) * MAX_IMAGE_BYTES},
        'independence_claim': '200 images, engineering linkage components; physical plate independence is not proved',
    }
    if audit_only:
        return summary
    proposal = {'schema_version': 2, 'rules': RULES, 'selected_dish_ids': selected,
        'selected_component_ids': {dish: mapping[dish] for dish in selected},
        'retained_original_dish_ids': retained, 'downloads': downloads,
        'retired_validation_dish_ids': sorted(original_val - set(retained)),
        'original_development_exposure_ids': sorted(original_dev)}
    if previous:
        # Preserve the exact pre-download request, including its initial URLs.
        proposal = previous
    frozen_write(v2, 'data/freeze/cohort_proposal.json', json_bytes(proposal), private=True)
    for name in sources.keys() & expected_sources.keys():
        if sources[name] != expected_sources[name]:
            fail('v1_original_source_mismatch')
    snapshot = {'source_sha256': sources, 'legacy_image_inventory': image_inventory,
        'component_members': members, 'ambiguous_component_ids': sorted(quarantine),
        'historical_exposure_ids': sorted(historical), 'active_development_ids': active_dev,
        'preparer_sha256': sha256(Path(__file__))}
    snapshot_name = 'final_source_snapshot.json' if previous and not downloads else 'source_snapshot.json'
    if not previous or not downloads:
        frozen_write(v2, 'data/freeze/' + snapshot_name, json_bytes(snapshot), private=True)
    frozen_write(v2, 'data/freeze/rules.json', json_bytes(RULES), private=True)
    dev_labels_source = json.loads(source(v1 / 'data/cohorts/dev_labels.json').read_text())
    val_labels_source = json.loads(source(v1 / 'data/cohorts/sealed/validation_labels.json').read_text())
    rows_out, labels_out = {'dev': [], 'validation': []}, {'dev': {}, 'validation': {}}
    files_dev = {}

    def copy_row(row, source_path, split):
        item = dict(row)
        name = rank('sample', row['sample_id']) + '.png'
        relative = f'data/{row["dataset"]}/{split}/images/{name}'
        data = Path(source_path).read_bytes()
        item['image_path'], item['image_sha256'] = relative, digest(data)
        frozen_write(v2, relative, data, private=split == 'validation')
        if split == 'dev':
            files_dev[relative] = item['image_sha256']
        rows_out[split].append(item)

    for split, rows in original.items():
        label_source = dev_labels_source if split == 'dev' else val_labels_source
        for row in rows:
            if row['dataset'] != 'asa24':
                continue
            copy_row(row, child(v1, row['image_path']), split)
            labels_out[split][row['sample_id']] = label_source[row['sample_id']]
    for split, dishes in (('dev', active_dev), ('validation', selected)):
        for dish in dishes:
            sample = 'nutrition5k-' + dish
            row = {'sample_id': sample, 'dataset': 'nutrition5k', 'split': split,
                'group_id': mapping[dish], 'fold': n5_folds[mapping[dish]] if split == 'dev' else None}
            if dish in image_paths:
                copy_row(row, image_paths[dish], split)
            labels_out[split][sample] = _truth(available[dish])
    rows_out['dev'].sort(key=lambda row: row['sample_id'])
    dev_payloads = {'data/cohorts/dev.jsonl': _line_payload(rows_out['dev']),
                    'data/cohorts/dev_labels.json': json_bytes(labels_out['dev'])}
    for relative, blob in dev_payloads.items():
        frozen_write(v2, relative, blob, private=relative.endswith('_labels.json'))
        files_dev[relative] = digest(blob)
    dev_freeze = {'schema_version': 2, 'seed': SEED, 'rules_sha256': digest(json_bytes(RULES)),
                  'artifacts': files_dev, 'counts': summary['development']}
    frozen_write(v2, 'data/cohorts/dev_freeze.json', json_bytes(dev_freeze), private=True)
    frozen_write(v2, 'data/cohorts/sealed/validation_labels.json', json_bytes(labels_out['validation']), private=True)
    ready = not downloads and not proposal_only
    if ready:
        rows_out['validation'].sort(key=lambda row: row['sample_id'])
        if len(rows_out['validation']) != 278:
            fail('final_validation_count_mismatch')
        relative = 'data/cohorts/sealed/validation.jsonl'
        frozen_write(v2, relative, _line_payload(rows_out['validation']), private=True)
        val_artifacts = {row['image_path']: row['image_sha256'] for row in rows_out['validation']}
        for name in (relative, 'data/cohorts/sealed/validation_labels.json'):
            val_artifacts[name] = sha256(child(v2, name))
        frozen_write(v2, 'data/freeze/validation_freeze.json', json_bytes({
            'schema_version': 2, 'artifacts': val_artifacts,
            'proposal_sha256': sha256(proposal_path), 'rules_sha256': digest(json_bytes(RULES)),
            'groups': len(selected_groups), 'images': 200}), private=True)
    summary['validation_ready'] = ready
    summary['development_ready'] = True
    summary['proposal_sha256'] = sha256(proposal_path)
    summary['dev_freeze_sha256'] = sha256(child(v2, 'data/cohorts/dev_freeze.json'))
    name = 'preparation_summary.json' if ready else 'audit_summary.json'
    if not previous or ready:
        frozen_write(v2, 'evidence/' + name, json_bytes(summary))
    verify_dev_only(v2)
    return summary


def verify_dev_only(v2_root=None):
    """Only dev manifest, dev labels, dev images and the dev-only seal are read."""
    base = Path(v2_root or V2).resolve()
    freeze = json.loads(child(base, 'data/cohorts/dev_freeze.json').read_text())
    if freeze.get('seed') != SEED or freeze.get('rules_sha256') != digest(json_bytes(RULES)):
        fail('development_rules_mismatch')
    expected_meta = {'data/cohorts/dev.jsonl', 'data/cohorts/dev_labels.json'}
    for relative, expected in freeze['artifacts'].items():
        if relative not in expected_meta and not re.fullmatch(
                r'data/(asa24|nutrition5k)/dev/images/[0-9a-f]{64}\.png', relative):
            fail('development_seal_has_forbidden_path')
        if sha256(child(base, relative)) != expected:
            fail('development_hash_mismatch')
    if not expected_meta <= freeze['artifacts'].keys():
        fail('development_seal_missing_manifest')
    rows = _jsonl(child(base, 'data/cohorts/dev.jsonl'))
    labels = json.loads(child(base, 'data/cohorts/dev_labels.json').read_text())
    if {row['sample_id'] for row in rows} != set(labels) or len(rows) != len(labels):
        fail('development_label_key_mismatch')
    groups, pixels = {}, {}
    for row in rows:
        if row.get('split') != 'dev' or type(row.get('fold')) is not int or row['fold'] not in range(5):
            fail('invalid_development_fold')
        if row['image_path'] not in freeze['artifacts'] or freeze['artifacts'][row['image_path']] != row['image_sha256']:
            fail('unsealed_development_image')
        for key, seen in ((row['group_id'], groups), (row['image_sha256'], pixels)):
            if key in seen and seen[key] != row['fold']:
                fail('development_group_or_image_crosses_folds')
            seen[key] = row['fold']
    for dataset in ('asa24', 'nutrition5k'):
        if len({row['group_id'] for row in rows if row['dataset'] == dataset}) < 5:
            fail('fewer_than_five_development_groups')
    return {'verified': True, 'counts': freeze['counts'],
            'dev_freeze_sha256': sha256(child(base, 'data/cohorts/dev_freeze.json'))}


def load_dev(v2_root=None):
    base = Path(v2_root or V2).resolve()
    verify_dev_only(base)
    return (_jsonl(child(base, 'data/cohorts/dev.jsonl')),
            json.loads(child(base, 'data/cohorts/dev_labels.json').read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--proposal-only', action='store_true')
    parser.add_argument('--verify-dev-only', action='store_true')
    args = parser.parse_args()
    try:
        result = verify_dev_only() if args.verify_dev_only else prepare(
            audit_only=args.audit_only, proposal_only=args.proposal_only)
    except AuditError as error:
        print(json.dumps({'status': 'blocked', 'error_code': str(error)}, sort_keys=True))
        return 2
    except Exception as error:
        # Do not echo paths, raw CSV text, labels, PIL exceptions or traceback.
        print(json.dumps({'status': 'blocked', 'error_code': 'unexpected_' + type(error).__name__}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
