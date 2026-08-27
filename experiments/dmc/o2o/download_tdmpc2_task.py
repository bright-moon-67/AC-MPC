#!/usr/bin/env python3
"""Selectively download one task from the public TD-MPC2 MT30 archive.

The MT30 release is four large, uncompressed ZIP64 ``torch.save`` files.  Each
file stores four dense tensors (task, observation, reward, action).  This tool
first reads only the task tensor via HTTP byte ranges, then fetches the rows
belonging to the requested task.  It therefore does not need to download the
full 44.25 GB release.

The output deliberately preserves the 501-row TD-MPC2 episode layout.  Row 0
is the reset/dummy row; rows 1..500 contain actions and rewards for the 500
outer environment transitions.  Observation and action padding is removed,
but no transition, reward, discount, or termination semantics are invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
from numpy.lib.format import open_memmap
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CANONICAL_DATASET_REPOSITORY = "https://huggingface.co/datasets/nicklashansen/tdmpc2"
DOWNLOAD_REPOSITORY = os.environ.get(
    "ACMPC_TDMPC2_DOWNLOAD_REPOSITORY",
    "https://hf-mirror.com/datasets/nicklashansen/tdmpc2",
)
DATASET_COMMIT = "d0364e9b4451761012e548b4269b7c6c02234c49"
EPISODE_ROWS = 501
REAL_TRANSITIONS_PER_EPISODE = 500
MT30_OBSERVATION_DIM = 24
MT30_ACTION_DIM = 6


@dataclass(frozen=True)
class Chunk:
    name: str
    size: int
    lfs_sha256: str

    @property
    def url(self) -> str:
        return (
            f"{DOWNLOAD_REPOSITORY}/resolve/{DATASET_COMMIT}/mt30/"
            f"{self.name}?download=true"
        )


CHUNKS = (
    Chunk(
        "chunk_0.pt",
        12_825_602_332,
        "1ae6b6337c292fe67a92b5dceef574ffb756a8c1cd15085fa0d202b53c980e51",
    ),
    Chunk(
        "chunk_1.pt",
        12_825_602_332,
        "81a480ac114ef9982634be8e0d81e4a03b08a60497ce136445eec2c03191805b",
    ),
    Chunk(
        "chunk_2.pt",
        12_825_602_332,
        "2cdc0c6b6d357684c5df1aa612d0cb3fc58d7f83bf22ba7edcdce363ca1e1320",
    ),
    Chunk(
        "chunk_3.pt",
        5_771_522_332,
        "0e5f412fd3d7d58fd78b291fcbf139b14a5ce03abed63a715a3844061aa92549",
    ),
)


@dataclass(frozen=True)
class ZipEntry:
    name: str
    local_header_offset: int
    compressed_size: int
    uncompressed_size: int
    compression_method: int
    data_offset: int = -1


TASKS = {
    "reacher-hard": {"id": 5, "observation_dim": 6, "action_dim": 2},
    "hopper-stand": {"id": 17, "observation_dim": 15, "action_dim": 4},
    "hopper-hop": {"id": 18, "observation_dim": 15, "action_dim": 4},
}

DEFAULT_WORKERS = 4


def _session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET", "HEAD")),
    )
    session = requests.Session()
    # The canonical HF endpoint is blocked on this node.  The configured
    # mirror is directly reachable; do not silently route it via the local
    # proxy, which is both slower and prone to breaking long SSH sessions.
    session.trust_env = False
    session.mount(
        "https://",
        HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32),
    )
    session.headers["User-Agent"] = "AC-MPC-TD-MPC2-selective-extractor/1"
    return session


def _range_response(
    session: requests.Session, url: str, start: int, end: int
) -> requests.Response:
    if start < 0 or end < start:
        raise ValueError(f"Invalid byte range {start}-{end}")
    response = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        stream=True,
        timeout=(30, 180),
    )
    response.raise_for_status()
    expected = end - start + 1
    if response.status_code != 206:
        response.close()
        raise RuntimeError(
            f"Server ignored range {start}-{end}: HTTP {response.status_code}"
        )
    content_range = response.headers.get("Content-Range", "")
    if not content_range.startswith(f"bytes {start}-{end}/"):
        response.close()
        raise RuntimeError(
            f"Unexpected Content-Range for {start}-{end}: {content_range!r}"
        )
    content_length = response.headers.get("Content-Length")
    if content_length is not None and int(content_length) != expected:
        response.close()
        raise RuntimeError(
            f"Range {start}-{end} declared {content_length} bytes, expected {expected}"
        )
    return response


def _read_range(
    session: requests.Session, url: str, start: int, end: int
) -> bytes:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(1, 9):
        response: requests.Response | None = None
        try:
            response = _range_response(session, url, start, end)
            payload = response.content
            if len(payload) != expected:
                raise RuntimeError(
                    f"Short range read {start}-{end}: got {len(payload)}, "
                    f"expected {expected}"
                )
            return payload
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt == 8:
                break
            time.sleep(min(2 ** (attempt - 1), 20))
        finally:
            if response is not None:
                response.close()
    raise RuntimeError(
        f"Failed byte range {start}-{end} after 8 attempts"
    ) from last_error


def _zip64_values(extra: bytes, required: tuple[str, ...]) -> dict[str, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        value = extra[cursor : cursor + size]
        cursor += size
        if tag != 0x0001:
            continue
        result: dict[str, int] = {}
        value_cursor = 0
        for field in required:
            width = 4 if field == "disk_start" else 8
            if value_cursor + width > len(value):
                raise RuntimeError("Truncated ZIP64 central-directory extra field")
            fmt = "<I" if width == 4 else "<Q"
            result[field] = struct.unpack_from(fmt, value, value_cursor)[0]
            value_cursor += width
        return result
    raise RuntimeError("ZIP64 values required but ZIP64 extra field is absent")


def _central_directory(
    session: requests.Session, chunk: Chunk
) -> tuple[int, int, bytes]:
    tail_size = min(chunk.size, 1 << 20)
    tail_start = chunk.size - tail_size
    tail = _read_range(session, chunk.url, tail_start, chunk.size - 1)
    zip64_position = tail.rfind(b"PK\x06\x06")
    if zip64_position < 0:
        raise RuntimeError(f"{chunk.name} has no ZIP64 end-of-directory record")
    absolute = tail_start + zip64_position
    record = tail[zip64_position : zip64_position + 56]
    if len(record) < 56:
        record = _read_range(session, chunk.url, absolute, absolute + 55)
    values = struct.unpack("<4sQ2H2I4Q", record)
    entries_total = values[7]
    directory_size = values[8]
    directory_offset = values[9]
    directory = _read_range(
        session,
        chunk.url,
        directory_offset,
        directory_offset + directory_size - 1,
    )
    return entries_total, directory_offset, directory


def _zip_entries(session: requests.Session, chunk: Chunk) -> dict[str, ZipEntry]:
    expected_entries, _offset, directory = _central_directory(session, chunk)
    entries: dict[str, ZipEntry] = {}
    cursor = 0
    while cursor < len(directory):
        if directory[cursor : cursor + 4] != b"PK\x01\x02":
            raise RuntimeError(
                f"Invalid central-directory signature at byte {cursor} in {chunk.name}"
            )
        values = struct.unpack_from("<4s6H3I5H2I", directory, cursor)
        method = values[4]
        compressed_size = values[8]
        uncompressed_size = values[9]
        name_length, extra_length, comment_length = values[10:13]
        disk_start = values[13]
        local_offset = values[16]
        name_start = cursor + 46
        name = directory[name_start : name_start + name_length].decode("utf-8")
        extra_start = name_start + name_length
        extra = directory[extra_start : extra_start + extra_length]
        required = []
        if uncompressed_size == 0xFFFFFFFF:
            required.append("uncompressed_size")
        if compressed_size == 0xFFFFFFFF:
            required.append("compressed_size")
        if local_offset == 0xFFFFFFFF:
            required.append("local_offset")
        if disk_start == 0xFFFF:
            required.append("disk_start")
        if required:
            zip64 = _zip64_values(extra, tuple(required))
            uncompressed_size = zip64.get("uncompressed_size", uncompressed_size)
            compressed_size = zip64.get("compressed_size", compressed_size)
            local_offset = zip64.get("local_offset", local_offset)
        header = _read_range(session, chunk.url, local_offset, local_offset + 29)
        local_values = struct.unpack("<I5H3I2H", header)
        if local_values[0] != 0x04034B50:
            raise RuntimeError(f"Invalid local header for {name!r}")
        local_method = local_values[3]
        local_name_length, local_extra_length = local_values[-2:]
        if local_method != method:
            raise RuntimeError(f"Compression method mismatch for {name!r}")
        data_offset = local_offset + 30 + local_name_length + local_extra_length
        entries[name.rsplit("/", 1)[-1] if "/data/" in name else name] = ZipEntry(
            name=name,
            local_header_offset=local_offset,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            compression_method=method,
            data_offset=data_offset,
        )
        cursor = extra_start + extra_length + comment_length
    if len(entries) != expected_entries:
        raise RuntimeError(
            f"Parsed {len(entries)} entries from {chunk.name}, expected {expected_entries}"
        )
    return entries


def _storage_entries(entries: dict[str, ZipEntry]) -> dict[str, ZipEntry]:
    storages = {key: entries[key] for key in ("0", "1", "2", "3")}
    if any(entry.compression_method != 0 for entry in storages.values()):
        raise RuntimeError("TD-MPC2 storage entries unexpectedly use ZIP compression")
    task = storages["0"]
    if task.uncompressed_size % (EPISODE_ROWS * 4):
        raise RuntimeError("Task storage is not divisible into 501-row int32 episodes")
    episodes = task.uncompressed_size // (EPISODE_ROWS * 4)
    expected_sizes = {
        "0": episodes * EPISODE_ROWS * 4,
        "1": episodes * EPISODE_ROWS * MT30_OBSERVATION_DIM * 4,
        "2": episodes * EPISODE_ROWS * 4,
        "3": episodes * EPISODE_ROWS * MT30_ACTION_DIM * 4,
    }
    actual_sizes = {key: value.uncompressed_size for key, value in storages.items()}
    if actual_sizes != expected_sizes:
        raise RuntimeError(
            f"Unexpected MT30 storage sizes: {actual_sizes}; expected {expected_sizes}"
        )
    return storages


def _scan_task_ids(
    session: requests.Session, chunk: Chunk, entry: ZipEntry, workers: int
) -> np.ndarray:
    row_bytes = EPISODE_ROWS * np.dtype("<i4").itemsize
    episode_count = entry.uncompressed_size // row_bytes
    result = np.empty(episode_count, dtype=np.int32)
    rows_per_request = 512
    requests_to_make = [
        (start, min(episode_count, start + rows_per_request))
        for start in range(0, episode_count, rows_per_request)
    ]

    def read_rows(bounds: tuple[int, int]) -> tuple[int, int, bytes]:
        row_start, row_end = bounds
        byte_start = entry.data_offset + row_start * row_bytes
        byte_end = entry.data_offset + row_end * row_bytes - 1
        return row_start, row_end, _read_range(
            session, chunk.url, byte_start, byte_end
        )

    completed = 0
    next_report = 10
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(read_rows, bounds) for bounds in requests_to_make]
        for future in as_completed(futures):
            row_start, row_end, payload = future.result()
            block = np.frombuffer(payload, dtype="<i4").reshape(
                row_end - row_start, EPISODE_ROWS
            )
            if not np.all(block == block[:, :1]):
                raise RuntimeError(f"Task ID changes inside an episode in {chunk.name}")
            result[row_start:row_end] = block[:, 0]
            completed += row_end - row_start
            percent = completed * 100 // episode_count
            if percent >= next_report:
                print(f"    {chunk.name} task index: {percent}%", flush=True)
                next_report += 10
    if np.any((result < 0) | (result >= 30)):
        raise RuntimeError(f"Invalid MT30 task ID in {chunk.name}")
    return result


def _global_episode_location(
    global_episode: int, episode_counts: list[int]
) -> tuple[int, int]:
    if global_episode < 0 or global_episode >= sum(episode_counts):
        raise IndexError(f"Global episode {global_episode} is outside MT30")
    offset = 0
    for chunk_index, count in enumerate(episode_counts):
        if global_episode < offset + count:
            return chunk_index, global_episode - offset
        offset += count
    raise AssertionError("Unreachable global episode location")


def _probe_task_episode(
    session: requests.Session,
    storage_maps: list[dict[str, ZipEntry]],
    episode_counts: list[int],
    global_episode: int,
) -> int:
    chunk_index, local_episode = _global_episode_location(
        global_episode, episode_counts
    )
    chunk = CHUNKS[chunk_index]
    entry = storage_maps[chunk_index]["0"]
    row_bytes = EPISODE_ROWS * np.dtype("<i4").itemsize
    start = entry.data_offset + local_episode * row_bytes
    payload = _read_range(session, chunk.url, start, start + row_bytes - 1)
    values = np.frombuffer(payload, dtype="<i4")
    if len(values) != EPISODE_ROWS or np.any(values != values[0]):
        raise RuntimeError(
            f"Task ID changes inside global episode {global_episode}"
        )
    task_id = int(values[0])
    if task_id < 0 or task_id >= 30:
        raise RuntimeError(
            f"Invalid task ID {task_id} at global episode {global_episode}"
        )
    return task_id


def _discover_contiguous_task_segment(
    session: requests.Session,
    storage_maps: list[dict[str, ZipEntry]],
    episode_counts: list[int],
    target_task: int,
    workers: int,
) -> tuple[int, int, dict[str, object]]:
    """Locate a task after auditing MT30's one-contiguous-segment layout."""

    total = sum(episode_counts)
    cache: dict[int, int] = {}

    def probe(position: int) -> int:
        if position not in cache:
            cache[position] = _probe_task_episode(
                session, storage_maps, episode_counts, position
            )
        return cache[position]

    probe_stride = 4_000
    while True:
        positions = sorted(set(range(0, total, probe_stride)) | {total - 1})
        missing = [position for position in positions if position not in cache]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _probe_task_episode,
                    session,
                    storage_maps,
                    episode_counts,
                    position,
                ): position
                for position in missing
            }
            completed = 0
            for future in as_completed(futures):
                cache[futures[future]] = future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(missing):
                    print(
                        f"    segment probes: {completed}/{len(missing)}",
                        flush=True,
                    )
        observed = {cache[position] for position in positions}
        if observed == set(range(30)):
            break
        if probe_stride <= 500:
            raise RuntimeError(
                f"Coarse task probes found IDs {sorted(observed)}, expected 0..29"
            )
        probe_stride //= 2
        print(
            f"    missing task IDs {sorted(set(range(30)) - observed)}; "
            f"refining probe stride to {probe_stride}",
            flush=True,
        )

    ordered_ids = [cache[position] for position in positions]
    run_order = [ordered_ids[0]]
    for task_id in ordered_ids[1:]:
        if task_id != run_order[-1]:
            run_order.append(task_id)
    if len(run_order) != 30 or len(set(run_order)) != 30:
        raise RuntimeError(
            "MT30 task IDs are not one unique contiguous segment per task: "
            f"{run_order}"
        )

    inside_positions = [
        position for position in positions if cache[position] == target_task
    ]
    if not inside_positions:
        raise RuntimeError(f"Task ID {target_task} was not found by segment probes")
    first_inside, last_inside = min(inside_positions), max(inside_positions)
    before = max((p for p in positions if p < first_inside), default=-1)
    after = min((p for p in positions if p > last_inside), default=total)
    if before >= 0 and probe(before) == target_task:
        raise AssertionError("Left task boundary bracket is invalid")
    if after < total and probe(after) == target_task:
        raise AssertionError("Right task boundary bracket is invalid")

    low, high = before, first_inside
    while high - low > 1:
        middle = (low + high) // 2
        if probe(middle) == target_task:
            high = middle
        else:
            low = middle
    segment_start = high

    low, high = last_inside, after
    while high - low > 1:
        middle = (low + high) // 2
        if probe(middle) == target_task:
            low = middle
        else:
            high = middle
    segment_end = high

    audit = {
        "layout": "one_contiguous_global_episode_segment_per_task",
        "probe_stride": probe_stride,
        "all_task_ids_observed": sorted(observed),
        "task_order": run_order,
        "coarse_probes": [
            {"global_episode": position, "task_id": cache[position]}
            for position in positions
        ],
        "target_segment_global_start": segment_start,
        "target_segment_global_end_exclusive": segment_end,
    }
    return segment_start, segment_end, audit


def _verify_selected_task_rows(
    *,
    session: requests.Session,
    chunk: Chunk,
    entry: ZipEntry,
    selected_indices: np.ndarray,
    expected_task: int,
    workers: int,
) -> None:
    if not len(selected_indices):
        return
    row_bytes = EPISODE_ROWS * np.dtype("<i4").itemsize
    batches: list[tuple[int, int]] = []
    for run_start, run_end in _consecutive_runs(selected_indices):
        for start in range(run_start, run_end, 512):
            batches.append((start, min(run_end, start + 512)))

    def read_batch(bounds: tuple[int, int]) -> bytes:
        start_episode, end_episode = bounds
        start = entry.data_offset + start_episode * row_bytes
        end = entry.data_offset + end_episode * row_bytes - 1
        return _read_range(session, chunk.url, start, end)

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(read_batch, bounds): bounds for bounds in batches
        }
        for future in as_completed(futures):
            start_episode, end_episode = futures[future]
            values = np.frombuffer(future.result(), dtype="<i4").reshape(
                end_episode - start_episode, EPISODE_ROWS
            )
            if np.any(values != expected_task):
                raise RuntimeError(
                    f"Task verification failed in {chunk.name} episodes "
                    f"{start_episode}:{end_episode}"
                )
            completed += end_episode - start_episode
    if completed != len(selected_indices):
        raise RuntimeError(
            f"Verified {completed} selected task rows, expected {len(selected_indices)}"
        )


def _consecutive_runs(indices: np.ndarray) -> Iterator[tuple[int, int]]:
    if not len(indices):
        return
    start = previous = int(indices[0])
    for value_raw in indices[1:]:
        value = int(value_raw)
        if value != previous + 1:
            yield start, previous + 1
            start = value
        previous = value
    yield start, previous + 1


def _download_rows(
    *,
    session: requests.Session,
    chunk: Chunk,
    entry: ZipEntry,
    source_dims: int,
    kept_dims: int,
    selected_indices: np.ndarray,
    output: np.memmap,
    output_offset: int,
    max_request_bytes: int,
    workers: int,
) -> int:
    row_values = EPISODE_ROWS * source_dims
    row_bytes = row_values * np.dtype("<f4").itemsize
    max_rows = max(1, max_request_bytes // row_bytes)
    batches: list[tuple[int, int, int]] = []
    cursor = output_offset
    for run_start, run_end in _consecutive_runs(selected_indices):
        batch_start = run_start
        while batch_start < run_end:
            batch_end = min(run_end, batch_start + max_rows)
            batches.append((batch_start, batch_end, cursor))
            cursor += batch_end - batch_start
            batch_start = batch_end

    expected_cursor = output_offset + len(selected_indices)
    if cursor != expected_cursor:
        raise RuntimeError(
            f"Planned output row {cursor}, expected {expected_cursor}"
        )

    def read_batch(batch: tuple[int, int, int]) -> tuple[int, int, bytes]:
        batch_start, batch_end, destination = batch
        start = entry.data_offset + batch_start * row_bytes
        end = entry.data_offset + batch_end * row_bytes - 1
        return destination, batch_end - batch_start, _read_range(
            session, chunk.url, start, end
        )

    completed = 0
    next_report = 10
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(read_batch, batch) for batch in batches]
        for future in as_completed(futures):
            destination, count, payload = future.result()
            array = np.frombuffer(payload, dtype="<f4").reshape(
                count, EPISODE_ROWS, source_dims
            )
            if kept_dims < source_dims and np.any(array[..., kept_dims:] != 0):
                raise RuntimeError(
                    f"Non-zero padding in {chunk.name} storage {entry.name!r}"
                )
            kept = array[..., :kept_dims]
            if output.ndim == 2:
                output[destination : destination + count] = kept[..., 0]
            else:
                output[destination : destination + count] = kept
            completed += count
            percent = completed * 100 // len(selected_indices)
            if percent >= next_report:
                print(f"    selected rows: {percent}%", flush=True)
                next_report += 10
    return expected_cursor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while payload := handle.read(8 << 20):
            digest.update(payload)
    return digest.hexdigest()


def _output_metadata(path: Path) -> dict[str, object]:
    array = np.load(path, mmap_mode="r")
    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), default="reacher-hard")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for losslessly unpadded 501-row episode arrays.",
    )
    parser.add_argument(
        "--max-request-mib",
        type=int,
        default=1,
        help="Maximum payload per selected data request (default: 1 MiB).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent HTTP range requests (default: 4; keep low on shared nodes).",
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Validate existing output arrays and write their manifest without downloading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")
    spec = TASKS[args.task]
    task_id = int(spec["id"])
    observation_dim = int(spec["observation_dim"])
    action_dim = int(spec["action_dim"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "source_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed extraction at {output_dir}"
        )

    session = _session()
    index_dir = output_dir / "source_task_indices"
    index_dir.mkdir(exist_ok=True)
    chunk_records: list[dict[str, object]] = []
    selections: list[np.ndarray] = []
    storage_maps: list[dict[str, ZipEntry]] = []

    print(f"Reading TD-MPC2 MT30 ZIP64 storage maps", flush=True)
    for chunk in CHUNKS:
        entries = _storage_entries(_zip_entries(session, chunk))
        storage_maps.append(entries)

    episode_counts = [
        entries["0"].uncompressed_size // (EPISODE_ROWS * 4)
        for entries in storage_maps
    ]
    print(
        f"Locating contiguous MT30 task {task_id} ({args.task}) segment",
        flush=True,
    )
    segment_start, segment_end, segment_audit = _discover_contiguous_task_segment(
        session, storage_maps, episode_counts, task_id, args.workers
    )
    global_offset = 0
    for chunk, entries, chunk_episode_count in zip(
        CHUNKS, storage_maps, episode_counts, strict=True
    ):
        local_start = max(0, segment_start - global_offset)
        local_end = min(chunk_episode_count, segment_end - global_offset)
        if local_end > local_start:
            selected = np.arange(local_start, local_end, dtype=np.int64)
        else:
            selected = np.empty(0, dtype=np.int64)
        selections.append(selected)
        run_count = sum(1 for _ in _consecutive_runs(selected))
        index_path = index_dir / f"{chunk.name}.task_ids.npy"
        if index_path.exists():
            full_index = np.load(index_path, mmap_mode="r")
            expected = np.flatnonzero(full_index == task_id)
            if not np.array_equal(expected, selected):
                raise RuntimeError(
                    f"Full cached task index disagrees with segment discovery for "
                    f"{chunk.name}"
                )
            print(
                f"  {chunk.name}: segment agrees with complete cached task index",
                flush=True,
            )
        elif len(selected):
            print(
                f"  {chunk.name}: verifying all {len(selected):,} selected task rows",
                flush=True,
            )
            _verify_selected_task_rows(
                session=session,
                chunk=chunk,
                entry=entries["0"],
                selected_indices=selected,
                expected_task=task_id,
                workers=args.workers,
            )
        print(
            f"  {chunk.name}: episodes={chunk_episode_count:,}, "
            f"selected={len(selected):,}, runs={run_count:,}",
            flush=True,
        )
        record: dict[str, object] = {
            "name": chunk.name,
            "url": chunk.url,
            "file_bytes": chunk.size,
            "lfs_sha256": chunk.lfs_sha256,
            "episode_count": int(chunk_episode_count),
            "selected_episode_count": int(len(selected)),
            "selected_run_count": run_count,
            "storage": {
                key: {
                    "entry": entry.name,
                    "data_offset": entry.data_offset,
                    "bytes": entry.uncompressed_size,
                }
                for key, entry in entries.items()
            },
        }
        if index_path.exists():
            record["complete_task_index"] = {
                "path": str(index_path.relative_to(output_dir)),
                "sha256": _sha256(index_path),
            }
        chunk_records.append(record)
        global_offset += chunk_episode_count

    episode_count = sum(len(selection) for selection in selections)
    if episode_count == 0:
        raise RuntimeError(f"No episodes found for task ID {task_id}")
    print(
        f"Found {episode_count:,} episodes / "
        f"{episode_count * REAL_TRANSITIONS_PER_EPISODE:,} real outer transitions",
        flush=True,
    )

    paths = {
        "observation": output_dir / "observation.npy",
        "action": output_dir / "action.npy",
        "reward": output_dir / "reward.npy",
        "source_chunk": output_dir / "source_chunk.npy",
        "source_episode": output_dir / "source_episode.npy",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not args.finalize_existing:
        raise FileExistsError(f"Refusing to overwrite partial output files: {existing}")
    if args.finalize_existing:
        expected_paths = {str(path) for path in paths.values()}
        if set(existing) != expected_paths:
            raise FileExistsError(
                "--finalize-existing requires all output arrays; missing: "
                f"{sorted(expected_paths - set(existing))}"
            )
        arrays = {
            "observation": np.load(paths["observation"], mmap_mode="r"),
            "action": np.load(paths["action"], mmap_mode="r"),
            "reward": np.load(paths["reward"], mmap_mode="r"),
        }
        source_chunk = np.load(paths["source_chunk"], mmap_mode="r")
        source_episode = np.load(paths["source_episode"], mmap_mode="r")
        output_offset = episode_count
        print("Finalizing existing arrays; no payload download", flush=True)
    else:
        arrays = {
            "observation": open_memmap(
                paths["observation"],
                mode="w+",
                dtype=np.float32,
                shape=(episode_count, EPISODE_ROWS, observation_dim),
            ),
            "action": open_memmap(
                paths["action"],
                mode="w+",
                dtype=np.float32,
                shape=(episode_count, EPISODE_ROWS, action_dim),
            ),
            "reward": open_memmap(
                paths["reward"],
                mode="w+",
                dtype=np.float32,
                shape=(episode_count, EPISODE_ROWS),
            ),
        }
        source_chunk = open_memmap(
            paths["source_chunk"], mode="w+", dtype=np.int16, shape=(episode_count,)
        )
        source_episode = open_memmap(
            paths["source_episode"], mode="w+", dtype=np.int32, shape=(episode_count,)
        )

        max_request_bytes = args.max_request_mib << 20
        output_offset = 0
        for chunk_index, (chunk, entries, selected) in enumerate(
            zip(CHUNKS, storage_maps, selections, strict=True)
        ):
            if not len(selected):
                continue
            end = output_offset + len(selected)
            source_chunk[output_offset:end] = chunk_index
            source_episode[output_offset:end] = selected
            print(
                f"Downloading {chunk.name}: {len(selected):,} selected episodes",
                flush=True,
            )
            field_specs = (
                ("observation", "1", MT30_OBSERVATION_DIM, observation_dim),
                ("reward", "2", 1, 1),
                ("action", "3", MT30_ACTION_DIM, action_dim),
            )
            for field, storage_key, source_dims, kept_dims in field_specs:
                started = time.monotonic()
                cursor = _download_rows(
                    session=session,
                    chunk=chunk,
                    entry=entries[storage_key],
                    source_dims=source_dims,
                    kept_dims=kept_dims,
                    selected_indices=selected,
                    output=arrays[field],
                    output_offset=output_offset,
                    max_request_bytes=max_request_bytes,
                    workers=args.workers,
                )
                if cursor != end:
                    raise RuntimeError(
                        f"{chunk.name} {field} ended at output row {cursor}, expected {end}"
                    )
                arrays[field].flush()
                print(
                    f"  {field}: complete in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
            output_offset = end

        source_chunk.flush()
        source_episode.flush()
        for array in arrays.values():
            array.flush()

    if output_offset != episode_count:
        raise RuntimeError(f"Wrote {output_offset} episodes, expected {episode_count}")
    if not np.all(np.isfinite(arrays["observation"])):
        raise RuntimeError("Extracted observations contain NaN or Inf")
    if not np.all(np.isfinite(arrays["action"][:, 1:])):
        raise RuntimeError("Extracted actions contain NaN or Inf")
    if not np.all(np.isfinite(arrays["reward"][:, 1:])):
        raise RuntimeError("Extracted rewards contain NaN or Inf")
    if np.any(np.abs(arrays["action"][:, 1:]) > 1.00001):
        raise RuntimeError("Extracted actions exceed [-1, 1]")

    output_records = {
        key: _output_metadata(path) for key, path in paths.items()
    }
    manifest = {
        "kind": "tdmpc2_mt30_task_episode_sequences_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": CANONICAL_DATASET_REPOSITORY,
            "download_repository": DOWNLOAD_REPOSITORY,
            "commit": DATASET_COMMIT,
            "dataset": "mt30",
            "chunks": chunk_records,
            "task_segment_audit": segment_audit,
        },
        "selection": {
            "task_name": args.task,
            "task_id": task_id,
            "episode_count": episode_count,
            "real_outer_transition_count": (
                episode_count * REAL_TRANSITIONS_PER_EPISODE
            ),
            "ordering": "source chunk order, then source episode order",
        },
        "schema": {
            "episode_rows": EPISODE_ROWS,
            "dummy_row": 0,
            "dummy_row_values": "preserved from source; action/reward are not validated",
            "real_transition_rows": [1, REAL_TRANSITIONS_PER_EPISODE],
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "source_observation_dim": MT30_OBSERVATION_DIM,
            "source_action_dim": MT30_ACTION_DIM,
            "padding_removed": True,
            "reward": "recorded sum of two native DMC substep rewards",
            "action_repeat": 2,
            "outer_control_timestep_seconds": 0.04,
            "canonical_transition_conversion_applied": False,
        },
        "outputs": output_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Complete: {manifest_path}", flush=True)
    print(json.dumps(manifest["selection"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) from None
