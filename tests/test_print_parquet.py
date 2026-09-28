"""実データ・ネットワークに依存しないParquet表示の検証。"""
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from wequant.cli import app
from wequant.flows.print_parquet import print_parquet_flow
from wequant.tasks.print_parquet import select_parquet_rows


class PrintParquetTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        patcher = patch('wequant.data_loading.DATA_DIR', self.directory)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.df = pl.DataFrame({'code': ['z', 'a', 'm'], 'value': [3, None, 1]})
        self.df.write_parquet(self.directory / 'arbitrary.parquet')
        self.runner = CliRunner()

    def test_flow_preserves_schema_and_stored_order(self):
        for options, expected in [({}, self.df), ({'head': 2}, self.df[:2]),
                                  ({'tail': 2}, self.df[1:]), ({'tail': 10}, self.df)]:
            with self.subTest(options=options):
                assert_frame_equal(print_parquet_flow(file='arbitrary', **options), expected)
        original = self.df.clone()
        select_parquet_rows(self.df, tail=1)
        assert_frame_equal(self.df, original)

    def test_cli_all_head_tail(self):
        for args, codes in [([], ['z', 'a', 'm']), (['--head', '2'], ['z', 'a']),
                            (['--tail', '2'], ['a', 'm'])]:
            with self.subTest(args=args):
                result = self.runner.invoke(app, ['print-parquet', '--file', 'arbitrary', *args])
                self.assertEqual(result.exit_code, 0, result.output)
                lines = result.stdout.splitlines()
                self.assertEqual(lines[0].split(), ['code', 'value'])
                self.assertEqual([line.split()[0] for line in lines[1:]], codes)
                self.assertIn('null', result.stdout)

    def test_library_rejects_invalid_limits_before_reading(self):
        for options in [{'head': 0}, {'tail': -1}, {'head': 1, 'tail': 1},
                        {'head': True}, {'tail': 1.5}]:
            with self.subTest(options=options), patch('wequant.flows.print_parquet.load_parquet') as load:
                with self.assertRaises(ValueError):
                    print_parquet_flow(file='arbitrary', **options)
                load.assert_not_called()
                with self.assertRaises(ValueError):
                    select_parquet_rows(self.df, **options)

    def test_invalid_names(self):
        for name in ['', '.', '..', '../arbitrary', '/arbitrary', 'dir/arbitrary',
                     'dir\\arbitrary', 'arbitrary.parquet']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                print_parquet_flow(file=name)

    def test_cli_invalid_arguments(self):
        for args in [[], ['--file', '../arbitrary'], ['--file', 'arbitrary', '--head', '0'],
                     ['--file', 'arbitrary', '--tail', '-1'],
                     ['--file', 'arbitrary', '--head', '1', '--tail', '1']]:
            with self.subTest(args=args):
                result = self.runner.invoke(app, ['print-parquet', *args])
                self.assertEqual(result.exit_code, 2, result.output)

    def test_missing_and_corrupt_files(self):
        (self.directory / 'broken.parquet').write_bytes(b'not parquet')
        for name in ['missing', 'broken']:
            with self.subTest(name=name):
                result = self.runner.invoke(app, ['print-parquet', '--file', name])
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIn('Parquetを読み込めません', result.stderr)

    def test_empty_file(self):
        self.df.clear().write_parquet(self.directory / 'empty.parquet')
        assert_frame_equal(print_parquet_flow(file='empty', tail=3), self.df.clear())
        result = self.runner.invoke(app, ['print-parquet', '--file', 'empty'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stdout.split(), ['code', 'value'])
        self.assertIn('該当データなし', result.stderr)

    def test_no_truncation_for_wide_nested_and_typed_data(self):
        df = pl.DataFrame({
            '日本語': ['長い文字列' * 30 + '\n続き\t末尾'] * 12,
            'date': [date(2026, 9, 27)] * 12,
            'decimal': [Decimal('123.45000')] * 12,
            'list': [list(range(20))] * 12,
            'struct': [{'key': 'value'}] * 12,
            **{f'extra{i}': [i] * 12 for i in range(12)},
        })
        df.write_parquet(self.directory / 'wide.parquet')
        with pl.Config(tbl_rows=2, tbl_cols=2, fmt_str_lengths=3):
            result = self.runner.invoke(app, ['print-parquet', '--file', 'wide'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(result.stdout.splitlines()), 13)
        self.assertIn('長い文字列' * 30 + r'\n続き\t末尾', result.stdout)
        for expected in ['2026-09-27', '123.45000', str(list(range(20))), "{'key': 'value'}", 'extra11']:
            self.assertIn(expected, result.stdout)
