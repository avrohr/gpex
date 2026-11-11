"""Small synthetic report checks, without JAX or benchmark execution.

Run directly: python profiling/test_summarize.py
"""

import json
import tempfile
import unittest
from pathlib import Path

import summarize
from datasets import dataset_sizes
from run import SCHEMA, digest, measurement_identity


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "report.json"
        self.fingerprint = digest(measurement_identity({}))

    def row(self, workload="datasets", sample=0, backend="cpu", **settings):
        case = dict(
            workload=workload,
            sample=sample,
            backend=backend,
            dtype="float64",
            n=settings.pop("n", 32),
            queries=1,
            repeats=2,
            pattern="distinct",
            min_size=2,
            sequence_length=20,
            **settings,
        )
        counts = (
            dataset_sizes(case)
            if workload.startswith("datasets")
            else list(range(case["start_size"] + 1, case["n"] + 1))
        )
        initial = 5 if workload.startswith("sequential") else 0
        step = sample + 1
        return dict(
            schema_version=SCHEMA,
            measurement="sequence",
            status="complete",
            validated=True,
            fingerprint=self.fingerprint,
            case=case,
            observation_counts=counts,
            initial_ms=initial,
            step_ms=[step] * len(counts),
            cumulative_ms=[initial + step * i for i in range(1, len(counts) + 1)],
            total_ms=initial + step * len(counts),
            trace_count=sample + 1,
            validation=dict(checked_predictions=len(counts) + bool(initial)),
            outputs=[[[1.0]]],
            memory=dict(
                process_peak_rss_bytes=100 + sample, peak_bytes_in_use=200 + sample
            ),
        )

    def load(self, rows):
        self.path.write_text(
            json.dumps(
                dict(
                    schema_version=SCHEMA,
                    synthetic=False,
                    status="complete",
                    fingerprint=self.fingerprint,
                    provenance={},
                    cases=[r["case"] for r in rows],
                    rows=rows,
                )
            )
        )
        return summarize.load([self.path])

    def test_sequence_distributions_and_settings(self):
        rows = [self.row(sample=i) for i in range(3)]
        larger = self.row(n=64)
        rows += [larger, self.row("sequential", backend="gpu", start_size=30)]
        points, coverage, sources, sequences = self.load(rows)
        self.assertEqual(points, {})
        self.assertEqual(sum(coverage.values()), 5)
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(sequences), 3)
        sequence = sequences[0]
        self.assertEqual(
            sequence["total_ms"], dict(median=40, min=20, max=60, q25=30, q75=50)
        )
        self.assertEqual(sequence["step_ms"]["median"], [2] * 20)
        self.assertEqual(sequence["cumulative_ms"]["min"], list(range(1, 21)))
        self.assertEqual(sequence["cumulative_ms"]["max"], list(range(3, 61, 3)))
        self.assertEqual(sequence["cumulative_ms"]["median"][-1], 40)
        self.assertEqual(sequence["process_totals_ms"], [20, 40, 60])
        self.assertEqual(sequence["trace_counts"], [1, 2, 3])
        self.assertEqual(sequence["memory"], 102)
        self.assertEqual(sequences[-1]["memory"], 200)
        self.assertEqual(sequences[-1]["initial_ms"]["median"], 5)
        self.assertEqual(sequences[-1]["total_ms"]["median"], 7)
        json.dumps(sequences, allow_nan=False)

    def test_ordinary_fitting_keeps_per_step_timing(self):
        row = self.row()
        row["case"]["workload"] = "mle"
        row["case"]["steps"] = 100
        row.update(samples_ms=[200, 400], first_call_ms=900, lower_ms=10, compile_ms=20)
        points, _, _, sequences = self.load([row])
        point = next(iter(points.values()))
        self.assertEqual(point["warm_ms"], 3)
        self.assertEqual(point["first_call_ms"], 900)
        self.assertEqual(point["setup_ms"], 30)
        self.assertEqual(sequences, [])

    def test_ranges_use_process_medians_and_paired_setup_times(self):
        rows = []
        for sample, (samples, lower, compile_ms) in enumerate(
            [
                ([100, 100, 100, 100, 900], 10, 100),
                ([200, 200], 100, 20),
                ([400, 400, 400], 30, 100),
            ]
        ):
            row = self.row(sample=sample)
            row["case"].update(workload="mle", steps=100, repeats=len(samples))
            row.update(
                samples_ms=samples,
                first_call_ms=1000 + 100 * sample,
                lower_ms=lower,
                compile_ms=compile_ms,
            )
            rows.append(row)
        points, _, _, _ = self.load(rows)
        point = next(iter(points.values()))
        self.assertEqual(point["warm_ms"], 2)
        self.assertEqual(point["process_ranges"]["warm_ms"], dict(min=1, max=4))
        self.assertEqual(point["process_ranges"]["setup_ms"], dict(min=110, max=130))
        self.assertEqual(
            point["process_ranges"]["first_call_ms"], dict(min=1000, max=1200)
        )
        self.assertEqual(point["process_ranges"]["memory"], dict(min=100, max=102))

    def test_rejects_invalid_sequence_and_nonfinite_outputs(self):
        for field, value in (
            ("total_ms", 999),
            ("outputs", [[[float("nan")]]]),
            ("validated", False),
        ):
            with self.subTest(field=field):
                row = self.row()
                row[field] = value
                with self.assertRaises((ValueError, AssertionError)):
                    self.load([row])

    def test_duplicate_report_is_rejected(self):
        self.load([self.row()])
        with self.assertRaises(AssertionError):
            summarize.load([self.path, self.path])


if __name__ == "__main__":
    unittest.main()
