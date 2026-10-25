from datetime import date
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from wequant.cli import app
from wequant.data_processing import MeigaralistPl, PricelistPl
from wequant.tasks.new_highs import build_new_highs


def prices():
    # Missing calendar dates do not add trading days. Equal highs reset the reference.
    return pl.DataFrame({
        'code': ['A'] * 6 + ['B'] * 3,
        'date': [date(2026, 1, d) for d in [2, 5, 6, 8, 9, 13, 2, 8, 9]],
        'high': [10., 9., 10., 9., 11., 12., 20., 19., 21.],
    }).reverse()


class NewHighsTests(TestCase):
    def test_history_ties_gaps_groups_sort_and_state(self):
        obj = PricelistPl(prices())
        original = obj.df.clone()
        expected = [(date(2026, 1, 9), 'A', 2),
                    (date(2026, 1, 9), 'B', 2),
                    (date(2026, 1, 13), 'A', 1)]
        self.assertEqual(obj.get_new_highs().rows(), expected)
        assert_frame_equal(obj.df, original)
        # Filtering must retain the history before the start date.
        self.assertEqual(obj.get_new_highs(start_date=date(2026, 1, 9),
                                          end_date=date(2026, 1, 9),
                                          min_prev_high=2).rows(), expected[:2])
        self.assertEqual(obj.get_new_highs(start_date=date(2026, 1, 10)).rows(), expected[2:])
        self.assertEqual(obj.get_new_highs(end_date=date(2026, 1, 8)).height, 0)
        self.assertEqual(obj.get_new_highs(min_prev_high=3).height, 0)
        # A future maximum must not change earlier events.
        assert_frame_equal(PricelistPl(prices().filter(pl.col('date') <= date(2026, 1, 9))).get_new_highs(),
                           obj.get_new_highs(end_date=date(2026, 1, 9)))

    def test_task_names_and_sort(self):
        source = prices().with_columns(
            pl.when((pl.col('code') == 'A') & (pl.col('date') == date(2026, 1, 6)))
            .then(9.).otherwise(pl.col('high')).alias('high'))
        names = MeigaralistPl(pl.DataFrame({'code': ['B'], 'name': ['日本企業']}))
        result = build_new_highs(PricelistPl(source), names)
        self.assertEqual(result.columns, ['date', 'code', 'name', 'prev-high'])
        self.assertEqual(result.rows(), [(date(2026, 1, 9), 'A', None, 4),
                                         (date(2026, 1, 9), 'B', '日本企業', 2),
                                         (date(2026, 1, 13), 'A', None, 1)])

    def test_empty_invalid_values_and_validation(self):
        obj = PricelistPl(prices())
        empty = PricelistPl(prices().head(0)).get_new_highs()
        self.assertEqual(empty.schema, obj.get_new_highs().schema)
        self.assertEqual(empty.height, 0)
        for val in (None, float('nan'), float('inf'), -float('inf')):
            df = pl.DataFrame({'code': ['A'] * 4,
                               'date': [date(2026, 1, d) for d in [2, 5, 6, 9]],
                               'high': [val, 10., val, 11.]})
            self.assertEqual(PricelistPl(df).get_new_highs().rows(), [(date(2026, 1, 9), 'A', 2)])
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                obj.get_new_highs(min_prev_high=value)
        with self.assertRaises(ValueError):
            obj.get_new_highs(start_date=date(2026, 2, 1), end_date=date(2026, 1, 1))
        with self.assertRaises(ValueError):
            PricelistPl(pl.concat([prices(), prices().head(1)])).get_new_highs()
        for column in ('date', 'code'):
            with self.assertRaises(ValueError):
                PricelistPl(prices().with_columns(pl.lit(None).cast(prices().schema[column]).alias(column))).get_new_highs()

    @patch('wequant.tasks.new_highs.MeigaralistPl.from_file')
    @patch('wequant.tasks.new_highs.PricelistPl.from_file')
    def test_cli_flow_loading_and_output(self, load_prices, load_names):
        load_prices.return_value = PricelistPl(prices())
        load_names.return_value = MeigaralistPl(pl.DataFrame({'code': ['A', 'B'], 'name': ['企業A', '企業B']}))
        result = CliRunner().invoke(app, ['new-highs', '--start-date', '2026-01-09',
                                         '--end-date', '2026-01-09', '--min-prev-high', '2'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stdout.splitlines()[0].split(), ['date', 'code', 'name', 'prev-high'])
        self.assertEqual([line.split() for line in result.stdout.splitlines()[1:]],
                         [['2026-01-09', 'A', '企業A', '2'], ['2026-01-09', 'B', '企業B', '2']])
        load_prices.assert_called_once_with('reviced_pricelist.parquet')
        load_names.assert_called_once_with()

    @patch('wequant.cli.new_highs_flow')
    def test_cli_defaults_empty_invalid_and_errors(self, flow):
        flow.return_value = pl.DataFrame(schema={'date': pl.Date, 'code': pl.String,
                                                'name': pl.String, 'prev-high': pl.Int64})
        runner = CliRunner()
        result = runner.invoke(app, ['new-highs'])
        self.assertEqual(result.exit_code, 0, result.output)
        flow.assert_called_once_with(start_date=None, end_date=None, min_prev_high=None)
        self.assertIn('該当データなし', result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        for args in (['--start-date', 'bad'], ['--end-date', '2026-02-30'],
                     ['--start-date', '2026-02-01', '--end-date', '2026-01-01'],
                     ['--min-prev-high', '0'], ['--min-prev-high', '1.5']):
            flow.reset_mock()
            result = runner.invoke(app, ['new-highs', *args])
            self.assertEqual(result.exit_code, 2, result.output)
            flow.assert_not_called()
        flow.side_effect = FileNotFoundError('missing parquet')
        result = runner.invoke(app, ['new-highs'])
        self.assertEqual(result.exit_code, 1)
        self.assertIn('新高値銘柄を取得できません', result.stderr)
