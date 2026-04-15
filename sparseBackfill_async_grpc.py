#!/usr/bin/env python3
"""
Backfill sparse values into an existing dense Pinecone index to make it hybrid.

This version uses:
- Pinecone gRPC client for data operations
- asyncio for pipelined, bounded concurrency
- retry logic with exponential backoff + jitter
- page-level concurrency
- update-level concurrency
- skip logic for records that already have sparse values

Recommended install:
    pip install -U "pinecone[grpc]"

Optional if your environment doesn't already include aiohttp and you want to reuse
this pattern with native async Pinecone clients later:
    pip install -U "pinecone[asyncio]"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from pinecone.grpc import PineconeGRPC as Pinecone


@dataclass
class Stats:
    total_listed: int = 0
    total_fetched: int = 0
    total_with_text: int = 0
    total_updated: int = 0
    total_skipped_existing_sparse: int = 0
    total_missing_text: int = 0
    total_embed_failures: int = 0
    total_update_failures: int = 0
    total_fetch_failures: int = 0
    total_list_failures: int = 0
    pages_started: int = 0
    pages_completed: int = 0


class StatsTracker:
    def __init__(self) -> None:
        self._stats = Stats()
        self._lock = asyncio.Lock()

    async def add(self, **kwargs: int) -> None:
        async with self._lock:
            for key, value in kwargs.items():
                setattr(self._stats, key, getattr(self._stats, key) + value)

    async def snapshot(self) -> Stats:
        async with self._lock:
            return Stats(**self._stats.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Pinecone index name")
    parser.add_argument("--host", default=os.getenv("PINECONE_INDEX_HOST"))
    parser.add_argument("--namespace", default="__default__")
    parser.add_argument("--prefix", default="")
    parser.add_argument(
        "--text-metadata-field",
        default="original_text",
        help="Metadata field containing the source text used to create sparse embeddings",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=96,
        help="Max 96 for pinecone-sparse-english-v0",
    )
    parser.add_argument(
        "--page-workers",
        type=int,
        default=4,
        help="How many fetched pages to process concurrently",
    )
    parser.add_argument(
        "--update-workers",
        type=int,
        default=16,
        help="How many vector updates to run concurrently",
    )
    parser.add_argument(
        "--embed-workers",
        type=int,
        default=4,
        help="How many embed requests to run concurrently across pages",
    )
    parser.add_argument("--pace-list-seconds", type=float, default=0.0)
    parser.add_argument("--pace-fetch-seconds", type=float, default=0.0)
    parser.add_argument("--pace-embed-seconds", type=float, default=0.0)
    parser.add_argument("--pace-update-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--backoff-base-seconds", type=float, default=1.0)
    parser.add_argument("--backoff-max-seconds", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-every-updates",
        type=int,
        default=100,
        help="Print progress every N successful updates (0 disables periodic progress logging)",
    )
    return parser.parse_args()


def normalize_cli_text(value: str) -> str:
    return value.replace("\u00A0", " ").strip()


def chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def async_sleep_if_needed(seconds: float) -> None:
    if seconds > 0:
        await asyncio.sleep(seconds)


def is_retryable_exception(exc: Exception) -> bool:
    status_candidates = [
        getattr(exc, "status", None),
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
    ]
    for status in status_candidates:
        if isinstance(status, int) and (status == 429 or 500 <= status < 600):
            return True

    text = str(exc).lower()
    retry_markers = [
        "429",
        "too many requests",
        "rate limit",
        "rate-limited",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "unavailable",
        "deadline exceeded",
        "resource exhausted",
    ]
    return any(marker in text for marker in retry_markers)


async def call_with_retry_async(
    fn: Callable[[], Awaitable[Any]],
    *,
    operation_name: str,
    max_retries: int,
    backoff_base_seconds: float,
    backoff_max_seconds: float,
    pace_after_success_seconds: float = 0.0,
) -> Any:
    attempt = 0
    while True:
        try:
            result = await fn()
            await async_sleep_if_needed(pace_after_success_seconds)
            return result
        except Exception as exc:
            if not is_retryable_exception(exc) or attempt >= max_retries:
                raise

            delay = min(backoff_max_seconds, backoff_base_seconds * (2 ** attempt))
            jitter = random.uniform(0, delay * 0.25)
            sleep_for = delay + jitter
            print(
                f"  {operation_name} failed with retryable error "
                f"(attempt {attempt + 1}/{max_retries + 1}): {exc}"
            )
            print(f"  Sleeping {sleep_for:.2f}s before retry...")
            await asyncio.sleep(sleep_for)
            attempt += 1


def extract_ids_from_page(page: Any) -> List[str]:
    ids: List[str] = []
    vectors = getattr(page, "vectors", None) or page.get("vectors", [])
    for v in vectors:
        if hasattr(v, "id"):
            ids.append(v.id)
        elif isinstance(v, dict) and "id" in v:
            ids.append(v["id"])
    return ids


def get_page_next_token(page: Any) -> Optional[str]:
    pagination = getattr(page, "pagination", None) or page.get("pagination")
    if not pagination:
        return None
    return getattr(pagination, "next", None) or pagination.get("next")


def get_index_host(pc: Pinecone, index_name: str, host_arg: Optional[str]) -> str:
    if host_arg:
        return host_arg
    desc = pc.describe_index(index_name)
    host = getattr(desc, "host", None) or desc["host"]
    if not host:
        raise RuntimeError(f"Unable to resolve host for index: {index_name}")
    return host


def get_index_metric(pc: Pinecone, index_name: str) -> Optional[str]:
    try:
        desc = pc.describe_index(index_name)
        return getattr(desc, "metric", None) or desc.get("metric")
    except Exception:
        return None


def fetch_vectors(index: Any, ids: List[str], namespace: str) -> Dict[str, Any]:
    resp = index.fetch(ids=ids, namespace=namespace)
    return getattr(resp, "vectors", None) or resp.get("vectors", {})


def get_metadata(vector_obj: Any) -> Dict[str, Any]:
    if hasattr(vector_obj, "metadata"):
        return vector_obj.metadata or {}
    return vector_obj.get("metadata", {}) or {}


def get_sparse_values(vector_obj: Any) -> Optional[Any]:
    sparse_values = getattr(vector_obj, "sparse_values", None)
    if sparse_values is None and isinstance(vector_obj, dict):
        sparse_values = vector_obj.get("sparse_values")
    return sparse_values


def has_sparse_values(vector_obj: Any) -> bool:
    sparse_values = get_sparse_values(vector_obj)
    if not sparse_values:
        return False

    if isinstance(sparse_values, dict):
        indices = sparse_values.get("indices", [])
        values = sparse_values.get("values", [])
    else:
        indices = getattr(sparse_values, "indices", None)
        values = getattr(sparse_values, "values", None)
        if indices is None:
            indices = getattr(sparse_values, "sparse_indices", [])
        if values is None:
            values = getattr(sparse_values, "sparse_values", [])

    return bool(indices) and bool(values)


def sparse_values_from_embedding_record(record: Any) -> Dict[str, List[Any]]:
    if isinstance(record, dict):
        indices = record.get("sparse_indices")
        values = record.get("sparse_values")
    else:
        indices = getattr(record, "sparse_indices", None)
        values = getattr(record, "sparse_values", None)

    if indices is None or values is None:
        raise ValueError(f"Unexpected sparse embedding response format: {record}")

    return {"indices": list(indices), "values": list(values)}


async def process_update_job(
    *,
    index: Any,
    vec_id: str,
    namespace: str,
    sparse_values: Dict[str, List[Any]],
    args: argparse.Namespace,
    update_sem: asyncio.Semaphore,
    stats: StatsTracker,
) -> None:
    if args.dry_run:
        print(
            f"  DRY RUN: would update {vec_id} from metadata.{args.text_metadata_field} "
            f"with {len(sparse_values['indices'])} sparse dimensions"
        )
        return

    async with update_sem:
        try:
            await call_with_retry_async(
                lambda: asyncio.to_thread(
                    index.update,
                    id=vec_id,
                    namespace=namespace,
                    sparse_values=sparse_values,
                ),
                operation_name="update",
                max_retries=args.max_retries,
                backoff_base_seconds=args.backoff_base_seconds,
                backoff_max_seconds=args.backoff_max_seconds,
                pace_after_success_seconds=args.pace_update_seconds,
            )
            await stats.add(total_updated=1)
            current = await stats.snapshot()
            if args.log_every_updates and current.total_updated % args.log_every_updates == 0:
                print(
                    f"Progress: updated={current.total_updated}, "
                    f"skipped_existing_sparse={current.total_skipped_existing_sparse}, "
                    f"missing_text={current.total_missing_text}"
                )
        except Exception as exc:
            await stats.add(total_update_failures=1)
            print(f"ERROR: update failed permanently for {vec_id}: {exc}")


async def process_page(
    *,
    page_num: int,
    ids: List[str],
    index: Any,
    pc: Pinecone,
    args: argparse.Namespace,
    stats: StatsTracker,
    embed_sem: asyncio.Semaphore,
    update_sem: asyncio.Semaphore,
) -> None:
    await stats.add(pages_started=1)
    print(f"\nPage {page_num}: listed {len(ids)} IDs")

    try:
        fetched = await call_with_retry_async(
            lambda: asyncio.to_thread(fetch_vectors, index, ids, args.namespace),
            operation_name="fetch",
            max_retries=args.max_retries,
            backoff_base_seconds=args.backoff_base_seconds,
            backoff_max_seconds=args.backoff_max_seconds,
            pace_after_success_seconds=args.pace_fetch_seconds,
        )
        await stats.add(total_fetched=len(fetched))
    except Exception as exc:
        await stats.add(total_fetch_failures=len(ids))
        print(f"ERROR: fetch failed permanently for page {page_num}: {exc}")
        await stats.add(pages_completed=1)
        return

    ids_to_embed: List[str] = []
    texts_to_embed: List[str] = []

    for vec_id in ids:
        vector_obj = fetched.get(vec_id)
        if vector_obj is None:
            print(f"  Skipping {vec_id}: fetch did not return the record")
            continue

        if has_sparse_values(vector_obj):
            await stats.add(total_skipped_existing_sparse=1)
            continue

        metadata = get_metadata(vector_obj)
        source_text = metadata.get(args.text_metadata_field)
        if not isinstance(source_text, str) or not source_text.strip():
            await stats.add(total_missing_text=1)
            continue

        ids_to_embed.append(vec_id)
        texts_to_embed.append(source_text)
        await stats.add(total_with_text=1)

    embed_batches = list(zip(chunked(ids_to_embed, args.embed_batch_size), chunked(texts_to_embed, args.embed_batch_size)))
    for id_batch, text_batch in embed_batches:
        try:
            async with embed_sem:
                embed_resp = await call_with_retry_async(
                    lambda tb=text_batch: asyncio.to_thread(
                        pc.inference.embed,
                        model="pinecone-sparse-english-v0",
                        inputs=tb,
                        parameters={"input_type": "passage", "truncate": "END"},
                    ),
                    operation_name="embed",
                    max_retries=args.max_retries,
                    backoff_base_seconds=args.backoff_base_seconds,
                    backoff_max_seconds=args.backoff_max_seconds,
                    pace_after_success_seconds=args.pace_embed_seconds,
                )
        except Exception as exc:
            await stats.add(total_embed_failures=len(id_batch))
            print(f"ERROR: embed failed permanently for batch of {len(id_batch)} on page {page_num}: {exc}")
            continue

        records = getattr(embed_resp, "data", None) or embed_resp.get("data", [])
        if len(records) != len(id_batch):
            await stats.add(total_embed_failures=len(id_batch))
            print(
                f"  Embedding count mismatch on page {page_num}: got {len(records)} embeddings "
                f"for {len(id_batch)} records"
            )
            continue

        update_tasks: List[asyncio.Task[None]] = []
        for vec_id, embed_record in zip(id_batch, records):
            try:
                sparse_values = sparse_values_from_embedding_record(embed_record)
            except Exception as exc:
                await stats.add(total_embed_failures=1)
                print(f"  Could not parse sparse embedding for {vec_id}: {exc}")
                continue

            update_tasks.append(
                asyncio.create_task(
                    process_update_job(
                        index=index,
                        vec_id=vec_id,
                        namespace=args.namespace,
                        sparse_values=sparse_values,
                        args=args,
                        update_sem=update_sem,
                        stats=stats,
                    )
                )
            )

        if update_tasks:
            await asyncio.gather(*update_tasks)

    await stats.add(pages_completed=1)


async def main_async() -> int:
    args = parse_args()
    args.namespace = normalize_cli_text(args.namespace)
    args.prefix = normalize_cli_text(args.prefix)
    args.text_metadata_field = normalize_cli_text(args.text_metadata_field)

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY is not set.", file=sys.stderr)
        return 1

    if args.embed_batch_size > 96:
        print("ERROR: --embed-batch-size cannot exceed 96 for pinecone-sparse-english-v0", file=sys.stderr)
        return 1
    if args.page_workers < 1 or args.update_workers < 1 or args.embed_workers < 1:
        print("ERROR: worker counts must all be at least 1", file=sys.stderr)
        return 1

    pc = Pinecone(api_key=api_key)
    metric = get_index_metric(pc, args.index)
    if metric and metric != "dotproduct":
        print(
            f"WARNING: index '{args.index}' has metric '{metric}', not 'dotproduct'. "
            "Single-index hybrid search requires a dense index using dotproduct."
        )

    host = get_index_host(pc, args.index, args.host)
    index = pc.Index(host=host)

    print(f"Index: {args.index}")
    print(f"Namespace: {repr(args.namespace)}")
    print(f"Text metadata field: {args.text_metadata_field}")
    print(f"Page workers: {args.page_workers}")
    print(f"Embed workers: {args.embed_workers}")
    print(f"Update workers: {args.update_workers}")

    stats = StatsTracker()
    embed_sem = asyncio.Semaphore(args.embed_workers)
    update_sem = asyncio.Semaphore(args.update_workers)

    pagination_token: Optional[str] = None
    page_num = 0
    inflight_pages: set[asyncio.Task[None]] = set()

    while True:
        page_num += 1
        try:
            page = await call_with_retry_async(
                lambda: asyncio.to_thread(
                    index.list_paginated,
                    namespace=args.namespace,
                    prefix=args.prefix,
                    limit=args.page_size,
                    pagination_token=pagination_token,
                ),
                operation_name="list_paginated",
                max_retries=args.max_retries,
                backoff_base_seconds=args.backoff_base_seconds,
                backoff_max_seconds=args.backoff_max_seconds,
                pace_after_success_seconds=args.pace_list_seconds,
            )
        except Exception as exc:
            await stats.add(total_list_failures=1)
            print(f"ERROR: list_paginated failed permanently on page {page_num}: {exc}")
            break

        ids = extract_ids_from_page(page)
        if not ids:
            print("No more IDs found.")
            break

        await stats.add(total_listed=len(ids))
        task = asyncio.create_task(
            process_page(
                page_num=page_num,
                ids=ids,
                index=index,
                pc=pc,
                args=args,
                stats=stats,
                embed_sem=embed_sem,
                update_sem=update_sem,
            )
        )
        inflight_pages.add(task)

        if len(inflight_pages) >= args.page_workers:
            done, pending = await asyncio.wait(inflight_pages, return_when=asyncio.FIRST_COMPLETED)
            inflight_pages = set(pending)
            for completed in done:
                await completed

        next_token = get_page_next_token(page)
        if not next_token:
            break
        pagination_token = next_token

    if inflight_pages:
        await asyncio.gather(*inflight_pages)

    final = await stats.snapshot()
    print("\nDone.")
    print(f"  Metadata field used:         {args.text_metadata_field}")
    print(f"  Total IDs listed:            {final.total_listed}")
    print(f"  Total IDs fetched:           {final.total_fetched}")
    print(f"  Total with source text:      {final.total_with_text}")
    print(f"  Skipped existing sparse:     {final.total_skipped_existing_sparse}")
    print(f"  Total updated:               {final.total_updated}")
    print(f"  Missing/empty source text:   {final.total_missing_text}")
    print(f"  List failures:               {final.total_list_failures}")
    print(f"  Fetch failures:              {final.total_fetch_failures}")
    print(f"  Embed failures:              {final.total_embed_failures}")
    print(f"  Update failures:             {final.total_update_failures}")
    print(f"  Pages started/completed:     {final.pages_started}/{final.pages_completed}")

    return 0 if (
        final.total_list_failures == 0
        and final.total_fetch_failures == 0
        and final.total_embed_failures == 0
        and final.total_update_failures == 0
    ) else 2


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
