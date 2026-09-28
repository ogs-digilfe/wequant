from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from wequant.cli import app
from wequant.flows.prepare_quarterly_valuation import prepare_quarterly_valuation_flow
from wequant.tasks.prepare_quarterly_valuation import (
    FEATURE_COLUMNS, PreparationOptions, prepare_quarterly_valuation,
    save_prepared_quarterly_valuation,
)


def dataset():
    return pl.DataFrame({
        "code": [3, 1, 2, 4],
        "name": ["三", "一", "二", "四"],
        "setd": [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30), date(2025, 3, 31)],
        "annd": [date(2024, 5, 1), date(2024, 8, 1), date(2024, 11, 1), date(2025, 5, 1)],
        "qtr": [1, 2, 3, 1],
        "sls": [100., 200., 300., 400.],
        "prft": [-10., 0., 30., 40.],
        "pr": [-10., 0., 10., 10.],
        "grsl": [10., 20., 30., 40.],
        "dgrp": [.1, .2, .3, None],
        "ngrpr": [None, 0., 30., 40.],
        "PER": [10., None, 30., 40.],
        "divr": [1., 2., 3., 4.],
        "perf": [5., -4., 10., 2.],
        "bm": [2., -1., 3., 1.],
    })


class PreparationTests(TestCase):
    def test_targets_order_roles_and_input_unchanged(self):
        original = dataset()
        before = original.clone()
        for target, expected in (("perf", [5., -4., 10., 2.]), ("excess-return", [3., -3., 7., 1.])):
            with self.subTest(target=target):
                result = prepare_quarterly_valuation(original, PreparationOptions(("ngrpr", "sls"), target))
                self.assertEqual(result.columns, ["code", "setd", "annd", "qtr", "ngrpr", "sls", "target"])
                self.assertEqual(result["target"].to_list(), expected)
                self.assertEqual(result["code"].to_list(), [3, 1, 2, 4])
                assert_frame_equal(result.select("ngrpr", "sls"), before.select("ngrpr", "sls"))
        assert_frame_equal(original, before)

    def test_date_bounds_quarters_and_filter_without_feature(self):
        result = prepare_quarterly_valuation(dataset(), PreparationOptions(
            ("sls",), "perf", setd_from=date(2024, 3, 31), setd_to=date(2024, 9, 30),
            qtr=(1, 3), numeric_ranges={"PER": (10., 30.)},
        ))
        self.assertEqual(result["code"].to_list(), [3, 2])
        self.assertNotIn("PER", result.columns)
        for kwargs, expected in (
            ({"setd_from": date(2024, 9, 30)}, [2, 4]),
            ({"setd_to": date(2024, 6, 30)}, [3, 1]),
        ):
            self.assertEqual(prepare_quarterly_valuation(
                dataset(), PreparationOptions(("sls",), "perf", **kwargs)
            )["code"].to_list(), expected)

    def test_each_numeric_range_inclusive_and_one_sided(self):
        for column in FEATURE_COLUMNS:
            frame = dataset().with_columns(pl.Series(column, [1., 2., 3., None]))
            for bounds, expected in (((1., 2.), [3, 1]), ((2., None), [1, 2]), ((None, 2.), [3, 1])):
                with self.subTest(column=column, bounds=bounds):
                    result = prepare_quarterly_valuation(frame, PreparationOptions(
                        ("sls",), "perf", numeric_ranges={column: bounds},
                    ))
                    self.assertEqual(result["code"].to_list(), expected)
        result = prepare_quarterly_valuation(dataset(), PreparationOptions(
            ("sls",), "perf", numeric_ranges={"dgrp": (.2, .2)},
        ))
        self.assertEqual(result["code"].to_list(), [1])

    def test_empty_and_missing_nonfinite_features(self):
        frame = dataset().with_columns(pl.Series("PER", [None, float("nan"), float("inf"), 1.]))
        kept = prepare_quarterly_valuation(frame, PreparationOptions(("PER",), "perf"))
        assert_frame_equal(kept.select("PER"), frame.select("PER"))
        filtered = prepare_quarterly_valuation(frame, PreparationOptions(
            ("PER",), "perf", numeric_ranges={"PER": (0., None)},
        ))
        self.assertEqual(filtered["code"].to_list(), [4])
        empty = prepare_quarterly_valuation(frame, PreparationOptions(
            ("PER",), "perf", numeric_ranges={"PER": (100., None)},
        ))
        self.assertEqual(empty.schema, kept.schema)
        self.assertEqual(empty.height, 0)
        assert_frame_equal(
            prepare_quarterly_valuation(frame.clear(), PreparationOptions(("PER",), "perf")),
            empty,
        )

    def test_invalid_options_and_leakage_rejected(self):
        invalid = [
            {"features": ()}, {"features": ("perf",)}, {"features": ("bm",)},
            {"features": ("code",)}, {"features": ("sls", "sls")}, {"target": "unknown"},
            {"qtr": (0,)}, {"qtr": (5,)},
            {"setd_from": date(2025, 1, 1), "setd_to": date(2024, 1, 1)},
            {"numeric_ranges": {"perf": (0., None)}},
            {"numeric_ranges": {"sls": (2., 1.)}},
            {"numeric_ranges": {"sls": (float("nan"), None)}},
            {"numeric_ranges": {"sls": (None, float("inf"))}},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                options = {"features": ("sls",), "target": "perf"} | overrides
                prepare_quarterly_valuation(dataset(), PreparationOptions(**options))

    def test_bad_schema_and_invalid_target(self):
        bad_frames = [
            dataset().drop("sls"),
            dataset().with_columns(pl.col("sls").cast(pl.String)),
            dataset().with_columns(pl.col("setd").cast(pl.String)),
            dataset().with_columns(pl.col("qtr").cast(pl.String)),
            dataset().with_columns(pl.lit(None).alias("perf")),
            dataset().with_columns(pl.lit(float("inf")).alias("perf")),
        ]
        for frame in bad_frames:
            with self.subTest(schema=frame.schema), self.assertRaises(ValueError):
                prepare_quarterly_valuation(frame, PreparationOptions(("sls",), "perf"))
        # perfだけを目的変数にする場合、bmの欠損は関係しない。
        frame = dataset().with_columns(pl.lit(None).alias("bm"))
        self.assertEqual(prepare_quarterly_valuation(frame, PreparationOptions(("sls",), "perf")).height, 4)
        with self.assertRaises(ValueError):
            prepare_quarterly_valuation(frame, PreparationOptions(("sls",), "excess-return"))

    def test_existing_dataset_builder_compatible(self):
        from test_ds_quarterly_valuation import inputs
        from wequant.tasks.ds_quarterly_valuation import build_ds_quarterly_valuation
        original = build_ds_quarterly_valuation(*inputs())
        result = prepare_quarterly_valuation(original, PreparationOptions(FEATURE_COLUMNS, "excess-return"))
        self.assertEqual(result.height, original.height)


class PreparationSaveTests(TestCase):
    def test_flow_roundtrip_metadata_relative_paths_and_no_overwrite(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.parquet"
            dataset().write_parquet(source)
            before = source.read_bytes()
            options = PreparationOptions(("ngrpr", "sls"), "excess-return", qtr=(1,))
            with patch("wequant.flows.prepare_quarterly_valuation.PROJECT_ROOT", root):
                result, path = prepare_quarterly_valuation_flow(Path("source.parquet"), options, Path("out/prepared.parquet"))
                assert_frame_equal(pl.read_parquet(path), result)
                meta = json.loads(path.with_suffix(".json").read_text())
                self.assertEqual(meta["input_path"], str(source))
                self.assertEqual(meta["input_sha256"], sha256(before).hexdigest())
                self.assertEqual(meta["features"], ["ngrpr", "sls"])
                self.assertEqual(meta["target"]["expression"], "perf - bm")
                self.assertEqual(meta["rows_before"], 4)
                self.assertEqual(meta["rows_after"], 2)
                self.assertEqual(meta["filters"]["qtr"], [1])
                saved = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    prepare_quarterly_valuation_flow(Path("source.parquet"), options, path)
                self.assertEqual(path.read_bytes(), saved)
                with self.assertRaises(FileExistsError):
                    prepare_quarterly_valuation_flow(source, options, source)
                self.assertEqual(source.read_bytes(), before)

    def test_default_filename_empty_roundtrip_and_json_collision(self):
        with TemporaryDirectory() as temp, patch(
            "wequant.tasks.prepare_quarterly_valuation.DATA_DIR", Path(temp)
        ), patch("wequant.tasks.prepare_quarterly_valuation.datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 28, 15, 30, 45)
            empty = prepare_quarterly_valuation(dataset().clear(), PreparationOptions(("sls",), "perf"))
            path = save_prepared_quarterly_valuation(empty, {})
            self.assertEqual(path, Path(temp) / "datasets/prepared/quarterly-valuation-20260928_153045.parquet")
            assert_frame_equal(pl.read_parquet(path), empty)
            with self.assertRaises(FileExistsError):
                save_prepared_quarterly_valuation(empty, {})
            path.unlink()
            with self.assertRaises(FileExistsError):
                save_prepared_quarterly_valuation(empty, {})
            self.assertFalse(path.exists())
            self.assertEqual(json.loads(path.with_suffix(".json").read_text()), {})

    def test_write_failure_cleanup_and_invalid_extension(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "failed.parquet"
            with patch.object(pl.DataFrame, "write_parquet", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    save_prepared_quarterly_valuation(dataset(), {}, path)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".json").exists())
            with patch("wequant.tasks.prepare_quarterly_valuation.json.dumps", side_effect=ValueError("json failed")):
                with self.assertRaises(ValueError):
                    save_prepared_quarterly_valuation(dataset(), {}, path)
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                save_prepared_quarterly_valuation(dataset(), {}, path.with_suffix(".json"))


class PreparationCliTests(TestCase):
    def test_cli_end_to_end_and_errors(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "source.parquet"
            output = Path(temp) / "out.parquet"
            dataset().write_parquet(source)
            base = ["prepare-quarterly-valuation", "--input", str(source),
                    "--target", "excess-return", "--feature", "ngrpr", "--feature", "sls",
                    "--output", str(output)]
            result = CliRunner().invoke(app, base + [
                "--setd-from", "2024-03-31", "--setd-to", "2024-09-30",
                "--qtr", "1", "--qtr", "3", "--per-min", "10", "--per-max", "30",
            ])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(pl.read_parquet(output)["code"].to_list(), [3, 2])
            self.assertIn(str(output), result.output)
            self.assertIn(str(output.with_suffix(".json")), result.output)
            output.unlink()
            output.with_suffix(".json").unlink()
            for args in (
                ["--feature", "perf"], ["--qtr", "5"], ["--setd-from", "invalid"],
                ["--sls-min", "2", "--sls-max", "1"], ["--sls-min", "nan"],
                ["--setd-from", "2025-01-01", "--setd-to", "2024-01-01"],
                ["--target", "invalid"],
            ):
                with self.subTest(args=args):
                    failed = CliRunner().invoke(app, base + args)
                    self.assertNotEqual(failed.exit_code, 0, failed.output)
                    self.assertFalse(output.exists())
            missing = CliRunner().invoke(app, ["prepare-quarterly-valuation", "--input", str(source)])
            self.assertNotEqual(missing.exit_code, 0)

    def test_all_numeric_cli_options_forwarded(self):
        args = ["prepare-quarterly-valuation", "--input", "source.parquet",
                "--target", "perf", "--feature", "sls"]
        for c in FEATURE_COLUMNS:
            args.extend([f"--{c.lower()}-min", "1", f"--{c.lower()}-max", "2"])
        with patch("wequant.flows.prepare_quarterly_valuation.prepare_quarterly_valuation_flow",
                   return_value=(dataset(), Path("prepared.parquet"))) as flow:
            result = CliRunner().invoke(app, args)
            self.assertEqual(result.exit_code, 0, result.output)
            options = flow.call_args.args[1]
            self.assertEqual(options.numeric_ranges, {c: (1., 2.) for c in FEATURE_COLUMNS})
