import asyncio
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List

import aiohttp
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")


@dataclass
class CrawlResult:
    url: str
    title: str | None
    links: List[str]
    word_count: int


def sync_crawl(url: str) -> CrawlResult:
    response = requests.get(url, timeout=30, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.string if soup.title else ""
    links = [a.get("href", "") for a in soup.find_all("a", href=True)]
    text = soup.get_text()
    word_count = len(text.split())

    return CrawlResult(url=url, title=title, links=links, word_count=word_count)


async def async_crawl(session: aiohttp.ClientSession, url: str) -> CrawlResult:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
        html = await response.text()
        soup = BeautifulSoup(html, "html.parser")

        title = soup.title.string if soup.title else ""
        links = [a.get("href", "") for a in soup.find_all("a", href=True)]
        text = soup.get_text()
        word_count = len(text.split())

        return CrawlResult(url=url, title=title, links=links, word_count=word_count)


def process_data(results: List[CrawlResult]) -> dict:
    total_links = sum(len(r.links) for r in results)
    total_words = sum(r.word_count for r in results)
    titles = [r.title for r in results if r.title]

    link_counts = [len(r.links) for r in results]
    word_counts = [r.word_count for r in results]

    return {
        "total_pages": len(results),
        "total_links": total_links,
        "avg_links_per_page": total_links / len(results) if results else 0,
        "total_words": total_words,
        "avg_words_per_page": total_words / len(results) if results else 0,
        "unique_titles": len(set(titles)),
        "median_links": statistics.median(link_counts) if link_counts else 0,
        "median_words": statistics.median(word_counts) if word_counts else 0,
    }


def sync_benchmark(urls: List[str]) -> tuple:
    start = time.perf_counter()

    results = []
    for url in urls:
        result = sync_crawl(url)
        results.append(result)

    process_start = time.perf_counter()
    processed = process_data(results)
    process_time = time.perf_counter() - process_start

    total_time = time.perf_counter() - start

    return total_time, process_time, processed


def threading_benchmark(urls: List[str], num_threads: int = 10) -> tuple:
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(sync_crawl, urls))

    process_start = time.perf_counter()
    processed = process_data(results)
    process_time = time.perf_counter() - process_start

    total_time = time.perf_counter() - start

    return total_time, process_time, processed


async def async_benchmark(urls: List[str]) -> tuple:
    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [async_crawl(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    process_start = time.perf_counter()
    processed = process_data(results)
    process_time = time.perf_counter() - process_start

    total_time = time.perf_counter() - start

    return total_time, process_time, processed


async def coroutine_benchmark(urls: List[str], batch_size: int = 50) -> tuple:
    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        results = []
        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]
            tasks = [async_crawl(session, url) for url in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

    process_start = time.perf_counter()
    processed = process_data(results)
    process_time = time.perf_counter() - process_start

    total_time = time.perf_counter() - start

    return total_time, process_time, processed


def multiprocessing_benchmark(urls: List[str], num_processes: int = 4) -> tuple:
    start = time.perf_counter()

    with Pool(processes=num_processes) as pool:
        results = pool.map(sync_crawl, urls)

    process_start = time.perf_counter()
    processed = process_data(results)
    process_time = time.perf_counter() - process_start

    total_time = time.perf_counter() - start

    return total_time, process_time, processed


TEST_URLS = [
    "https://httpbin.org/html",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/2",
    "https://httpbin.org/uuid",
    "https://httpbin.org/user-agent",
    "https://httpbin.org/get",
    "https://httpbin.org/headers",
    "https://httpbin.org/ip",
    "https://httpbin.org/response-headers",
    "https://httpbin.org/json",
] * 2


def run_benchmarks():
    print("=" * 60)
    print("BENCHMARK: Sync vs Threading vs Multiprocessing")
    print("=" * 60)

    sync_time, sync_process, sync_result = sync_benchmark(TEST_URLS)
    print("\nSync (sequential):")
    print(f"  Total time: {sync_time:.3f}s")
    print(f"  Process time: {sync_process:.3f}s")
    print(f"  Pages crawled: {sync_result['total_pages']}")

    thread_time, thread_process, thread_result = threading_benchmark(
        TEST_URLS, num_threads=10
    )
    print("\nThreading (10 threads):")
    print(f"  Total time: {thread_time:.3f}s")
    print(f"  Process time: {thread_process:.3f}s")
    print(f"  Pages crawled: {thread_result['total_pages']}")

    print(f"\nSpeedup: {sync_time / thread_time:.2f}x")

    mp_time, mp_process, mp_result = multiprocessing_benchmark(
        TEST_URLS, num_processes=4
    )
    print("\nMultiprocessing (4 processes):")
    print(f"  Total time: {mp_time:.3f}s")
    print(f"  Process time: {mp_process:.3f}s")
    print(f"  Pages crawled: {mp_result['total_pages']}")

    print(f"\nSpeedup: {sync_time / mp_time:.2f}x")

    print("\n" + "=" * 60)
    print("BENCHMARK: Async vs Coroutine (batch)")
    print("=" * 60)

    async_time, async_process, async_result = asyncio.run(async_benchmark(TEST_URLS))
    print("\nAsync (gather all):")
    print(f"  Total time: {async_time:.3f}s")
    print(f"  Process time: {async_process:.3f}s")
    print(f"  Pages crawled: {async_result['total_pages']}")

    coro_time, coro_process, coro_result = asyncio.run(
        coroutine_benchmark(TEST_URLS, batch_size=10)
    )
    print("\nCoroutine (batch 10):")
    print(f"  Total time: {coro_time:.3f}s")
    print(f"  Process time: {coro_process:.3f}s")
    print(f"  Pages crawled: {coro_result['total_pages']}")

    print(f"\nSpeedup: {async_time / coro_time:.2f}x")

    print("\n" + "=" * 60)
    print("COMPARISON: All methods")
    print("=" * 60)
    print(f"\n{'Method':<25} {'Time (s)':<12} {'Relative':<10}")
    print("-" * 47)
    print(f"{'Sync':<25} {sync_time:<12.3f} {1.0:<10.2f}")
    print(f"{'Threading':<25} {thread_time:<12.3f} {sync_time / thread_time:<10.2f}")
    print(f"{'Multiprocessing':<25} {mp_time:<12.3f} {sync_time / mp_time:<10.2f}")
    print(f"{'Async':<25} {async_time:<12.3f} {sync_time / async_time:<10.2f}")
    print(f"{'Coroutine batch':<25} {coro_time:<12.3f} {sync_time / coro_time:<10.2f}")


if __name__ == "__main__":
    run_benchmarks()
