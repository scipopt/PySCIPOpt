import ctypes
import gc
import json
import os
import time
import tracemalloc

import pytest

MODE = os.environ.get("BENCH_MODE", "time")
OUT = os.environ.get("BENCH_OUT")
RESULTS = []
_dumps = 0


def pytest_configure(config):
    config.addinivalue_line("markers", "timing: large benchmark compared by wall-clock time")


def _instructions(call):
    global _dumps
    lib = ctypes.CDLL(os.environ["BENCH_CALLGRIND_LIB"])
    gc.collect()
    gc.disable()
    try:
        lib.bench_start()
        call()
        lib.bench_stop()
    finally:
        gc.enable()
    _dumps += 1
    with open(f"{os.environ['BENCH_CALLGRIND_OUT']}.{_dumps}") as f:
        for line in f:
            if line.startswith("totals:"):
                return {"instructions": int(line.split()[1])}
    raise RuntimeError("no totals in callgrind dump")


def _memory(call):
    gc.collect()
    tracemalloc.start()
    result = call()
    peak = tracemalloc.get_traced_memory()[1]
    del result
    gc.collect()
    retained = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return {"peak_bytes": peak, "retained_bytes": retained}


def _time(prepare, rounds):
    best = float("inf")
    for _ in range(rounds):
        call = prepare()
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            call()
            best = min(best, time.perf_counter() - start)
        finally:
            gc.enable()
    return {"seconds": best}


@pytest.fixture
def bench(request):
    def run(target, setup=None, rounds=5):
        def prepare():
            args, kwargs = setup() if setup else ((), {})
            return lambda: target(*args, **kwargs)

        if MODE == "instructions":
            record = _instructions(prepare())
        elif MODE == "memory":
            record = _memory(prepare())
        else:
            record = _time(prepare, rounds)
        record["name"] = request.node.nodeid
        RESULTS.append(record)
        if OUT:
            with open(OUT, "a") as f:
                f.write(json.dumps(record) + "\n")

    return run


def pytest_terminal_summary(terminalreporter):
    if RESULTS:
        terminalreporter.section(f"benchmarks ({MODE})")
        for r in RESULTS:
            values = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items() if k != "name")
            terminalreporter.write_line(f"{r['name']}: {values}")
