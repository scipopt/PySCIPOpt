import argparse
import json

METRICS = (
    ("Instructions", "instructions", "count", 0.01),
    ("Time", "seconds", "time", 0.20),
    ("Peak memory", "peak_bytes", "bytes", 0.05),
    ("Retained memory", "retained_bytes", "bytes", 0.05),
)
MIN_BYTES = 100_000


def load(paths):
    results = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                results.setdefault(record.pop("name"), {}).update(record)
    return results


def fmt(value, kind):
    if value is None:
        return "-"
    if kind == "time":
        for unit, scale in (("s", 1), ("ms", 1e-3), ("µs", 1e-6)):
            if value >= scale:
                return f"{value / scale:.2f} {unit}"
        return f"{value / 1e-9:.0f} ns"
    units = (("G", 1e9), ("M", 1e6), ("k", 1e3))
    suffix = "B" if kind == "bytes" else ""
    digits = 1 if kind == "bytes" else 3
    for unit, scale in units:
        if value >= scale:
            return f"{value / scale:.{digits}f} {unit}{suffix}"
    return f"{value} {suffix}".strip()


def cell(b, p, kind, threshold):
    if b is None and p is None:
        return "", False
    text = f"{fmt(b, kind)} → {fmt(p, kind)}"
    if b is None or p is None or b == 0:
        return text, False
    change = p / b - 1
    if kind == "bytes" and abs(p - b) < MIN_BYTES:
        return f"{text} ({change:+.1%})", False
    mark = ""
    if change > threshold:
        mark = " :red_circle:"
    elif change < -threshold:
        mark = " :green_circle:"
    return f"{text} ({change:+.1%}){mark}", change > threshold


def main(base_paths, pr_paths, title):
    base, pr = load(base_paths), load(pr_paths)
    rows, regressions = [], 0
    for name in sorted(base.keys() | pr.keys()):
        error = pr.get(name, {}).get("error")
        if error:
            rows.append(f"| `{name.split('::')[-1]}` | :x: {error} on PR |" + " |" * (len(METRICS) - 1))
            regressions += 1
            continue
        cells = []
        for _, key, kind, threshold in METRICS:
            text, worse = cell(base.get(name, {}).get(key), pr.get(name, {}).get(key), kind, threshold)
            cells.append(text)
            regressions += worse
        rows.append(f"| `{name.split('::')[-1]}` | " + " | ".join(cells) + " |")

    print(f"## {title}\n")
    print(f":warning: {regressions} regression{'s' * (regressions > 1)}\n" if regressions else "No regressions\n")
    print("| Benchmark | " + " | ".join(m[0] for m in METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    print("\n".join(rows))
    print(
        "\nInstructions from callgrind, time is the best of several runs, memory is Python-side (tracemalloc). "
        "Flagged above " + ", ".join(f"{t:.0%} {label.lower()}" for label, _, _, t in METRICS) + "."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--pr", nargs="+", required=True)
    parser.add_argument("--title", default="Benchmark results (base → PR)")
    args = parser.parse_args()
    main(args.base, args.pr, args.title)
