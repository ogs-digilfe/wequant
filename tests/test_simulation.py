from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from wequant.cli import app
from wequant.data_processing import IndexPricelistPl, MeigaralistPl, PricelistPl
from wequant.flows.simulation import threshold_simulation_flow
from wequant.tasks.simulation import build_threshold_simulation, save_threshold_simulation


def day(n):
    return date(2026, 1, n)


def prices(ohlc=None):
    values = ohlc or [(100, 105, 98, 100), (100, 110, 96, 110), (115, 118, 108, 112)]
    return PricelistPl(pl.DataFrame({
        'code': ['7203'] * len(values), 'date': [day(n) for n in [2, 5, 6][:len(values)]],
        **{col: [float(row[i]) for row in values]
           for i, col in enumerate(['open', 'high', 'low', 'close'])},
    }).reverse())


def index():
    return IndexPricelistPl(pl.DataFrame({
        'date': [day(2), day(5), day(6)], 'open': [1000., 1020., 1050.],
        'close': [1010., 1040., 1060.],
    }))


def simulate(obj=None, entries=None, **kwargs):
    return (obj or prices()).get_threshold_simulation(
        entries if entries is not None else [(day(2), '7203')], take_profit=10, stop_loss=5,
        index_df=index().df, **kwargs,
    )


class ThresholdSimulationTests(TestCase):
    def test_close_signal_sells_at_next_open_and_preserves_state(self):
        obj = prices()
        original = obj.df.clone()
        row = simulate(obj, entry_mode='open').row(0, named=True)
        self.assertEqual((row['entry_date'], row['exit_date']), (day(2), day(6)))
        self.assertEqual(row['exit_price'], 115)
        self.assertEqual(row['updown'], 15)
        self.assertAlmostEqual(row['benchmark'], 5)
        self.assertEqual(row['exit_reason'], 'take-profit')
        assert_frame_equal(obj.df, original)

    def test_next_open_ignores_intraday_gaps_and_ranges(self):
        obj = prices([(100, 105, 98, 100), (120, 125, 90, 100), (100, 105, 98, 100)])
        row = simulate(obj, entry_mode='open').row(0, named=True)
        self.assertEqual(row['status'], 'unclosed')
        self.assertIsNone(row['updown'])
        row = simulate(prices([(100, 105, 98, 100), (100, 105, 95, 95), (90, 95, 85, 90)]),
                       entry_mode='open').row(0, named=True)
        self.assertEqual((row['exit_reason'], row['exit_price'], row['updown']), ('stop-loss', 90., -10.))

    def test_intraday_threshold_gap_and_both_priority(self):
        cases = [
            ([(100, 110, 96, 100)], 'take-profit', 110., 10.),
            ([(100, 105, 95, 100)], 'stop-loss', 95., -5.),
            ([(100, 105, 98, 100), (120, 125, 115, 120)], 'take-profit', 120., 20.),
            ([(100, 105, 98, 100), (90, 94, 85, 90)], 'stop-loss', 90., -10.),
            ([(100, 105, 98, 100), (120, 125, 90, 100)], 'both_thresholds', None, 0.),
        ]
        for ohlc, reason, price, rate in cases:
            with self.subTest(reason=reason, price=price):
                row = simulate(prices(ohlc), entry_mode='open', exit_mode='intraday').row(0, named=True)
                self.assertEqual((row['exit_reason'], row['exit_price'], row['updown']), (reason, price, rate))
                if reason == 'both_thresholds':
                    self.assertEqual(row['benchmark'], 0)
                elif len(ohlc) == 1:
                    self.assertAlmostEqual(row['benchmark'], 1)

    def test_calendar_defaults_duplicates_groups_missing_and_empty(self):
        obj = prices()
        obj = PricelistPl(pl.concat([obj.df, obj.df.with_columns(pl.lit('0001').alias('code'))]))
        entries = [(day(2), '7203'), (day(3), '0001'), (day(2), '7203'), (day(7), '7203'), (day(2), 'absent')]
        result = simulate(obj, entries)
        self.assertEqual(result['code'].to_list(), ['7203', '0001', '7203', '7203', 'absent'])
        self.assertEqual(result['entry_date'].to_list(), [day(5), day(5), day(5), None, None])
        self.assertEqual(result['status'].to_list()[-2:], ['no_data', 'no_data'])
        self.assertEqual(simulate(entries=[(day(3), '7203')], entry_mode='open')['entry_date'][0], day(5))
        self.assertEqual(simulate(entries=[]).height, 0)
        self.assertEqual(simulate(PricelistPl(prices().df.head(0)))['status'][0], 'no_data')

    def test_last_day_signal_missing_benchmark_and_invalid_prices(self):
        row = simulate(prices([(100, 105, 98, 100), (100, 110, 96, 110)]),
                       entry_mode='open').row(0, named=True)
        self.assertEqual(row['status'], 'unclosed')
        self.assertIsNone(row['exit_date'])
        result = prices().get_threshold_simulation(
            [(day(2), '7203')], take_profit=10, stop_loss=5, index_df=index().df.head(1), entry_mode='open')
        self.assertIsNone(result['benchmark'][0])
        # intradayでは終値を判定に使わない。
        obj = PricelistPl(prices([(100, 110, 96, 100)]).df.with_columns(
            pl.lit(None).cast(pl.Float64).alias('close')))
        self.assertEqual(simulate(obj, entry_mode='open', exit_mode='intraday')['updown'][0], 10.)
        for column, value in [('open', 0.), ('open', float('nan')), ('close', None)]:
            obj = PricelistPl(prices().df.with_columns(
                pl.when(pl.col('date') == day(2)).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)))
            self.assertEqual(simulate(obj, entry_mode='open')['status'][0], 'invalid_data')

    def test_invalid_arguments_and_duplicate_prices(self):
        for rate in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                prices().get_threshold_simulation([], take_profit=rate, stop_loss=5, index_df=index().df)
        for mode in [{'entry_mode': 'bad'}, {'exit_mode': 'bad'}]:
            with self.assertRaises(ValueError):
                simulate(**mode)
        with self.assertRaises(ValueError):
            simulate(PricelistPl(pl.concat([prices().df, prices().df.head(1)])))

    def test_task_name_and_flow_loaders(self):
        names = MeigaralistPl(pl.DataFrame({'code': [7203], 'name': ['企業A']}))
        original = names.df.clone()
        result = build_threshold_simulation(prices(), names, index(), [(day(2), '7203')],
                                            take_profit=10, stop_loss=5)
        self.assertEqual(result['name'][0], '企業A')
        assert_frame_equal(names.df, original)
        with patch('wequant.tasks.simulation.PricelistPl.from_file', return_value=prices()) as lp, \
             patch('wequant.tasks.simulation.MeigaralistPl.from_file', return_value=names) as ln, \
             patch('wequant.tasks.simulation.IndexPricelistPl.from_file', return_value=index()) as li:
            actual, path = threshold_simulation_flow([(day(2), '7203')], take_profit=10, stop_loss=5)
            assert_frame_equal(actual, result)
            self.assertIsNone(path)
            lp.assert_called_once_with('reviced_pricelist.parquet')
            ln.assert_called_once_with()
            li.assert_called_once_with()

    def test_file_csv_roundtrip_no_overwrite(self):
        frame = simulate(entry_mode='open')
        with TemporaryDirectory(dir='.') as directory, \
             patch('wequant.tasks.simulation.DATA_DIR', Path(directory)), \
             patch('wequant.tasks.simulation.datetime') as clock:
            clock.now.return_value.strftime.return_value = '20260102090000'
            path = save_threshold_simulation(frame)
            self.assertEqual(path.name, 'threshold-20260102090000.csv')
            assert_frame_equal(pl.read_csv(path, try_parse_dates=True, schema_overrides={"code": pl.String}), frame)
            with self.assertRaises(FileExistsError):
                save_threshold_simulation(frame)
            with patch('wequant.flows.simulation.load_simulation_inputs', return_value=(prices(), MeigaralistPl(
                    pl.DataFrame({'code': ['7203'], 'name': ['企業A']})), index())), \
                 patch('wequant.flows.simulation.save_threshold_simulation', return_value=path) as save:
                result, actual_path = threshold_simulation_flow([(day(2), '7203')], take_profit=10,
                                                                 stop_loss=5, output_mode='file')
                self.assertEqual(actual_path, path)
                save.assert_called_once()

    @patch('wequant.flows.simulation.threshold_simulation_flow')
    def test_cli_defaults_multiple_entries_and_errors(self, flow):
        flow.return_value = (simulate(), None)
        runner = CliRunner()
        base = ['sim', 'threshold', '--entry', '2026-01-02:7203', '--take-profit', '10', '--stop-loss', '5']
        response = runner.invoke(app, [*base, '--entry', '2026-01-03:0001'])
        self.assertEqual(response.exit_code, 0, response.output)
        flow.assert_called_once_with([(day(2), '7203'), (day(3), '0001')], take_profit=10., stop_loss=5.,
                                     entry_mode='next-open', exit_mode='next-open', output_mode='standard')
        self.assertIn('benchmark', response.output)
        for extra in [['--entry', 'bad'], ['--entry', '2026-02-30:7203'], ['--entry', '2026-01-02:'],
                      ['--take-profit', 'nan'], ['--stop-loss', '0'], ['--exit-mode', 'bad']]:
            flow.reset_mock()
            response = runner.invoke(app, [*base, *extra])
            self.assertEqual(response.exit_code, 2, response.output)
            flow.assert_not_called()
        flow.return_value = (simulate(), Path('output.csv'))
        response = runner.invoke(app, [*base, '--output-mode', 'file', '--entry-mode', 'open', '--exit-mode', 'intraday'])
        self.assertEqual(response.exit_code, 0, response.output)
        self.assertEqual(response.stdout.strip(), 'output.csv')
        flow.side_effect = OSError('missing parquet')
        self.assertEqual(runner.invoke(app, base).exit_code, 1)
        self.assertEqual(runner.invoke(app, ['sim', 'threshold']).exit_code, 2)
        self.assertEqual(runner.invoke(app, ['sim', '--help']).exit_code, 0)
        self.assertEqual(runner.invoke(app, ['sim', 'threshold', '--help']).exit_code, 0)
