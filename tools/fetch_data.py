#!/usr/bin/env python3
"""Plan or materialize the fixed, public development-image selections."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from urllib.request import HTTPRedirectHandler, build_opener
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("historical", "development", "external50")
DATASETS = ("asa24", "nutrition5k", "nutritionverse-real")
N5_PREFIX = (
    "https://storage.googleapis.com/nutrition5k_dataset/"
    "nutrition5k_dataset/imagery/realsense_overhead/"
)
MAX_IMAGE_BYTES = 64 * 1024 * 1024


def relative_path(value):
    if (
        not isinstance(value, str)
        or "\\" in value
        or any(not part or part in {".", ".."} for part in value.split("/"))
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("invalid relative path")
    return PurePosixPath(value)


def local_path(root, value):
    path = root.joinpath(*relative_path(value).parts)
    if path.resolve() != path:
        raise ValueError(f"linked path refused: {value}")
    return path


def validate_image(row, cohort):
    dataset = row["dataset"]
    digest = row["sha256"]
    if dataset not in DATASETS or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid dataset or SHA-256")
    suffix = ".jpg" if dataset == "nutritionverse-real" else ".png"
    expected = f"data/images/{cohort}/{dataset}/{digest}{suffix}"
    if row["path"] != expected or row["image_id"] != f"{dataset}-{digest}":
        raise ValueError("image path or ID does not match the fixed image digest")
    for field in ("source_path", "historical_source_path"):
        if field in row:
            relative_path(row[field])
    if "bytes" in row and (
        type(row["bytes"]) is not int or not 0 < row["bytes"] <= MAX_IMAGE_BYTES
    ):
        raise ValueError("invalid image length")
    if "url" in row:
        dish = row.get("source_id", "")
        if (
            dataset != "nutrition5k"
            or not re.fullmatch(r"dish_[0-9]+", dish)
            or row["url"] != f"{N5_PREFIX}{dish}/rgb.png"
        ):
            raise ValueError("URL is outside the fixed Nutrition5k object format")


def read_manifest(root, cohort):
    path = root / "data" / "manifests" / f"{cohort}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["cohort"] != cohort:
        raise ValueError("unexpected manifest schema or cohort")
    images = {}
    for row in manifest["images"]:
        validate_image(row, cohort)
        if row["image_id"] in images:
            raise ValueError("duplicate image ID")
        images[row["image_id"]] = row
    sample_ids = set()
    referenced = set()
    for row in manifest["samples"]:
        sample_id = row["sample_id"]
        if sample_id in sample_ids or row["image_id"] not in images:
            raise ValueError("duplicate sample ID or unknown sample image")
        sample_ids.add(sample_id)
        referenced.add(row["image_id"])
    if referenced != images.keys():
        raise ValueError("unreferenced image in manifest")
    return images, manifest["samples"]


def image_bytes(stream, row):
    limit = row.get("bytes", MAX_IMAGE_BYTES)
    data = stream.read(limit + 1)
    if not data or len(data) > limit or ("bytes" in row and len(data) != limit):
        raise ValueError("image length mismatch")
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("image SHA-256 mismatch")
    return data


def check_existing(target, row):
    if not target.exists():
        return False
    if not target.is_file():
        raise ValueError("image destination is not a regular file")
    with target.open("rb") as stream:
        image_bytes(stream, row)
    return True


def zip_index(archive, members):
    entries = defaultdict(list)
    for item in archive.infolist():
        if PurePosixPath(item.filename).name not in members:
            continue
        name = relative_path(item.filename)
        mode = stat.S_IFMT(item.external_attr >> 16)
        if item.is_dir() or mode not in {0, stat.S_IFREG} or item.flag_bits & 1:
            raise ValueError("unsupported ZIP member type")
        entries[name.name].append(item)
    return entries


def from_archive(archive, entries, row):
    candidates = entries.get(row.get("source_path"))
    if not candidates:
        raise ValueError("ASA24 member unavailable in this ZIP")
    for item in candidates:
        try:
            if not 0 < item.file_size <= MAX_IMAGE_BYTES:
                raise ValueError("ZIP member length outside image limit")
            with archive.open(item) as stream:
                return image_bytes(stream, row)
        except ValueError:
            continue
    raise ValueError("ASA24 member bytes differ from the frozen image")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def get_image(row, source_dir, archive, entries, opener):
    if source_dir:
        for field in ("historical_source_path", "source_path"):
            if field not in row:
                continue
            path = local_path(source_dir, row[field])
            if path.is_file():
                with path.open("rb") as stream:
                    return image_bytes(stream, row)
    if row["dataset"] == "asa24" and archive is not None:
        return from_archive(archive, entries, row)
    if opener is not None and "url" in row:
        with opener.open(row["url"], timeout=60) as response:
            return image_bytes(response, row)
    raise ValueError("supply the local frozen source or enable the applicable download")


def materialize(rows, root, source_dir, archive, download):
    entries = {}
    if archive is not None:
        members = {
            row["source_path"]
            for row in rows
            if row["dataset"] == "asa24" and "source_path" in row
        }
        entries = zip_index(archive, members)
    opener = build_opener(NoRedirect()) if download else None
    failures = 0
    for row in rows:
        try:
            target = local_path(root, row["path"])
            if check_existing(target, row):
                continue
            data = get_image(row, source_dir, archive, entries, opener)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temp = Path(stream.name)
                try:
                    stream.write(data)
                    stream.close()
                    # Atomically publish the verified bytes without replacing a concurrent writer.
                    try:
                        target.hardlink_to(temp)
                    except FileExistsError:
                        check_existing(target, row)
                finally:
                    temp.unlink(missing_ok=True)
        except (OSError, RuntimeError, ValueError) as exc:
            failures += 1
            print(f"{row['image_id']}: {exc}")
    print(f"{len(rows) - failures}/{len(rows)} images ready")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=COHORTS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--round", type=int, choices=range(1, 11))
    parser.add_argument("--archive", type=Path, help="local NCI ASA24 ZIP")
    parser.add_argument("--source-dir", type=Path, help="local source tree")
    parser.add_argument("--apply", action="store_true", help="import local images")
    parser.add_argument("--download", action="store_true", help="allow official N5 HTTPS")
    args = parser.parse_args()
    if args.round and args.cohort != "development":
        parser.error("--round is only available for the development pool")
    images, samples = read_manifest(ROOT, args.cohort)
    selected = {
        row["image_id"] for row in samples
        if (args.round is None or row.get("round") == args.round)
        and (args.dataset is None or images[row["image_id"]]["dataset"] == args.dataset)
    }
    rows = [row for image_id, row in images.items() if image_id in selected]
    if not rows:
        parser.error("selection contains no images")
    if not (args.apply or args.download):
        for row in rows:
            source = row.get("url") or row.get("source_path") or row["historical_source_path"]
            print(f"{source}\t{row['path']}\t{row['sha256']}")
        print(f"{len(rows)} images planned")
        return 0
    source_dir = args.source_dir.resolve() if args.source_dir else None
    if source_dir and not source_dir.is_dir():
        parser.error("--source-dir is not a directory")
    if args.archive:
        with ZipFile(args.archive) as archive:
            return bool(materialize(rows, ROOT, source_dir, archive, args.download))
    return bool(materialize(rows, ROOT, source_dir, None, args.download))


if __name__ == "__main__":
    raise SystemExit(main())
