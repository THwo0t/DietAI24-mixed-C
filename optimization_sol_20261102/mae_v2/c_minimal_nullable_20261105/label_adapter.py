"""Deterministic adapters for already-allowed development references.

No model, image, credential or network access. The paired execution caller loads
these references only after the complete blind-identity seal is checked.
Frozen source labels are never rewritten and equivalent options are not expanded
into extra scoring rows.
"""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

TARGETS = ('weight', 'energy', 'fat', 'carbs', 'protein')


def valid_targets(value):
    return isinstance(value, dict) and all(
        type(value.get(k)) in (int, float) and math.isfinite(value[k]) and value[k] >= 0
        for k in TARGETS)


def adapt_asa_labels(source, sample_ids):
    result = {}
    for sid in sample_ids:
        references = source[sid]['references']
        if not references:
            raise ValueError('empty_reference_set')
        adapted, seen = [], set()
        for ref in references:
            if not valid_targets(ref.get('truth')):
                raise ValueError('invalid_reference_targets')
            options = ref['reference_option_ids']
            if not isinstance(options, list) or not options or len(set(options)) != len(options):
                raise ValueError('invalid_equivalent_options')
            # Identity depends on source association, never a prediction/error.
            key = {'food_code': ref['food_code'], 'portion_code': ref['portion_code'],
                   'reference_option_ids': sorted(options)}
            ref_id = 'asa-ref-' + hashlib.sha256(json.dumps(
                key, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if ref_id in seen:
                raise ValueError('duplicate_source_reference')
            seen.add(ref_id)
            adapted.append({**deepcopy(ref), 'ref_id': ref_id})
        result[sid] = {'references': adapted}
    return result


def parse_allowed_n5_line(line, allowed_dish_to_sample):
    """Check the leading dish identity before CSV decoding or numeric parsing."""
    dish = line.partition(',')[0].strip('\ufeff\r\n')
    if dish not in allowed_dish_to_sample:
        return None
    fields = next(csv.reader([line]))
    if len(fields) < 6:
        raise ValueError('missing_allowed_n5_targets')
    energy, weight, fat, carbs, protein = map(float, fields[1:6])
    truth = dict(zip(TARGETS, (weight, energy, fat, carbs, protein)))
    if not valid_targets(truth) or weight <= 0:
        raise ValueError('invalid_allowed_n5_targets')
    return allowed_dish_to_sample[dish], {'truth': truth}


def load_allowed_n5_labels(source_paths, allowed_rows, source_sha256):
    """Source tables are pinned; only prequalified non-acceptance IDs are parsed.

    allowed_rows contain dish_id/sample_id/source_table from a verified blind
    pool, not an arbitrary list selected after seeing model results.
    """
    if len({r['dish_id'] for r in allowed_rows}) != len(allowed_rows):
        raise ValueError('duplicate_allowed_dish')
    if len({r['sample_id'] for r in allowed_rows}) != len(allowed_rows):
        raise ValueError('duplicate_allowed_sample')
    labels = {}
    for table, path in source_paths.items():
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_sha256[table]:
            raise ValueError('n5_source_binding_changed')
        allowed = {r['dish_id']: r['sample_id'] for r in allowed_rows if r['source_table'] == table}
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                parsed = parse_allowed_n5_line(line, allowed)
                if parsed is None:
                    continue
                sid, label = parsed
                if sid in labels:
                    raise ValueError('duplicate_allowed_source_label')
                labels[sid] = label
    if set(labels) != {r['sample_id'] for r in allowed_rows}:
        raise ValueError('allowed_source_labels_incomplete')
    return labels
