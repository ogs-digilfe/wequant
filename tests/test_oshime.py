from datetime import date, timedelta
from math import isfinite
from random import Random
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from wequant.cli import app
from wequant.data_processing import MeigaralistPl, PricelistPl
from wequant.tasks.oshime import build_oshime_candidates

START = date(2026, 1, 1)
TARGET = START + timedelta(days=119)


def prices(code='130A'):
    close = [100. + .2*i for i in range(120)]
    close[-10:] = [124., 125., 124., 123., 122., 121., 120.5, 121., 121., 125.]
    high = [x+1 for x in close]
    high[-10] = 130.
    return pl.DataFrame({'code': [code]*120,
                         'date': [START+timedelta(days=i) for i in range(120)],
                         'close': close, 'high': high, 'low': [x-1 for x in close]})


def changed(df, column, index, value):
    return df.with_columns(pl.when(pl.col('date') == START+timedelta(days=index))
                           .then(pl.lit(value)).otherwise(pl.col(column)).alias(column))


def reference(df, rate=5.):
    """One-code independent scalar specification, including full-window validity."""
    rows = df.sort('date').to_dicts()
    t = len(rows)-1

    def valid(v):
        return v is not None and isfinite(v)

    def ma(i, n):
        if i < n-1:
            return None
        values = [r['close'] for r in rows[i-n+1:i+1]]
        return sum(values)/n if all(map(valid, values)) else None

    a, b, c, d = ma(t, 25), ma(t, 75), ma(t-5, 25), ma(t-10, 75)
    if any(v is None for v in (a, b, c, d)) or not (a > b and a > c and b > d):
        return False
    if not all(valid(rows[i][col]) for i, col in ((t,'close'), (t-1,'close'), (t-1,'high'))):
        return False
    if ma(t, 5) is None or ma(t-1, 5) is None:
        return False
    if not (rows[t]['close'] >= ma(t, 5) and rows[t-1]['close'] < ma(t-1, 5)
            and rows[t]['close'] > rows[t-1]['high']):
        return False
    if not all(valid(r['high']) for r in rows[-25:]):
        return False
    peak = max(range(t-24,t+1), key=lambda i: (rows[i]['high'], i))
    if peak < 59 or not all(valid(r['high']) for r in rows[peak-59:peak+1]):
        return False
    high = rows[peak]['high']
    if high <= 0 or high != max(r['high'] for r in rows[peak-59:peak+1]) or peak == t:
        return False
    if not all(valid(r['low']) for r in rows[peak+1:]):
        return False
    low_i = min(range(peak+1,t+1), key=lambda i: (rows[i]['low'], -i))
    low, avg = rows[low_i]['low'], ma(low_i, 25)
    return (3 <= low_i-peak <= 15 and 1 <= t-low_i <= 5 and low > 0
            and avg is not None and avg > 0 and .97 <= low/avg <= 1.03
            and rate <= (high-low)/high*100 <= 12
            and (rows[t]['close']-low)/low*100 <= 5)


class OshimeTests(TestCase):
    def matches(self, df, rate=5.):
        return PricelistPl(df).get_oshime_candidates(valuation_date=TARGET, decline_rate=rate).height == 1

    def test_scalar_reference_and_boundaries(self):
        source = prices()
        self.assertTrue(self.matches(source))
        cases = [source.head(n) for n in (59, 74, 84, 85, 119, 120)]
        # High/low dates exercise period limits, recency, and last equal extrema.
        for i in (95, 98, 99, 103, 110, 111, 112, 113, 114, 115, 116, 118, 119):
            cases += [changed(source, 'high', i, 130.), changed(source, 'low', i, 119.5)]
        for col, i, vals in (
            ('high',110,(125.,130.,135.,140.)),
            ('high',80,(129.,130.,131.)),
            ('low',116,(110.,115.,119.5,121.,125.)),
            ('close',119,(121.,122.,125.,126.,130.)),
            ('close',118,(119.,121.,124.,125.)),
        ):
            cases += [changed(source,col,i,v) for v in vals]
        rng = Random(10)
        for _ in range(60):
            df = changed(source, 'close', rng.randrange(35,120), rng.uniform(70,145))
            cases.append(df)
        for df in cases:
            for rate in (0.,5.,8.,12.):
                with self.subTest(rows=df.height, rate=rate):
                    self.assertEqual(self.matches(df,rate),reference(df,rate))
        low = source['low'][116]
        exact = (130.-low)/130.*100
        self.assertTrue(self.matches(source, exact))
        self.assertFalse(self.matches(source, exact+1e-8))
        self.assertFalse(self.matches(changed(source,'high',80,131.)))
        self.assertFalse(self.matches(changed(source,'high',119,130.)))
        # Earlier low before peak is not the pullback low.
        self.assertTrue(self.matches(changed(source,'low',100,100.)))

    def test_missing_nonfinite_empty_future_groups_and_state(self):
        good = prices()
        for col, i in (('close',50), ('high',80), ('low',115)):
            for value in (None,float('nan'),float('inf')):
                self.assertFalse(self.matches(changed(good,col,i,value)))
        stale = prices('STALE').head(119)
        bad = changed(prices('BAD'),'low',116,1.)
        future = changed(prices(),'close',119,1.).tail(1).with_columns(
            pl.lit(TARGET+timedelta(days=10)).alias('date'))
        obj = PricelistPl(pl.concat([good,stale,bad,future]).reverse())
        original = obj.df.clone()
        self.assertEqual(obj.get_oshime_candidates(valuation_date=TARGET)['code'].to_list(),['130A'])
        assert_frame_equal(obj.df,original)
        names = MeigaralistPl(pl.DataFrame({'code':['OTHER'],'name':['別銘柄']}))
        self.assertEqual(build_oshime_candidates(obj,names,valuation_date=TARGET).rows(), [('130A',None)])
        self.assertTrue(obj.get_oshime_candidates(valuation_date=START-timedelta(days=1)).is_empty())
        result = PricelistPl(good.head(0)).get_oshime_candidates(valuation_date=TARGET)
        self.assertEqual(result.schema,good.select('code').schema)
        for rate in (-1.,12.01,100.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                self.matches(good,rate)

    @patch('wequant.tasks.oshime.MeigaralistPl.from_file')
    @patch('wequant.tasks.oshime.PricelistPl.from_file')
    def test_cli_flow_loading_output_and_decline(self, load_prices, load_names):
        load_prices.return_value = PricelistPl(prices())
        load_names.return_value = MeigaralistPl(pl.DataFrame({'code':['130A'],'name':['日本企業']}))
        args = ['get-oshime','--valuation-date',TARGET.isoformat()]
        result = CliRunner().invoke(app,args)
        self.assertEqual(result.exit_code,0,result.output)
        self.assertEqual(result.stdout.splitlines()[0].split(),['code','name'])
        self.assertIn('130A  日本企業',result.stdout)
        load_prices.assert_called_once_with('reviced_pricelist.parquet')
        load_names.assert_called_once_with()
        result = CliRunner().invoke(app,args+['--decline-rate','9'])
        self.assertEqual(result.exit_code,0,result.output)
        self.assertNotIn('130A',result.stdout)

    @patch('wequant.cli.get_oshime_flow')
    def test_cli_defaults_validation_and_errors(self, flow):
        flow.return_value = pl.DataFrame(schema={'code':pl.String,'name':pl.String})
        result = CliRunner().invoke(app,['get-oshime'])
        self.assertEqual(result.exit_code,0,result.output)
        flow.assert_called_once_with(valuation_date=date.today(),decline_rate=5.)
        self.assertIn('該当データなし',result.stderr)
        for rate in ('0','7.5','12'):
            flow.reset_mock()
            result = CliRunner().invoke(app,['get-oshime','--decline-rate',rate])
            self.assertEqual(result.exit_code,0,result.output)
            flow.assert_called_once_with(valuation_date=date.today(),decline_rate=float(rate))
        for args in (['--valuation-date','invalid'], *[['--decline-rate',v] for v in ('-1','12.01','nan','inf')]):
            flow.reset_mock()
            result = CliRunner().invoke(app,['get-oshime',*args])
            self.assertEqual(result.exit_code,2,result.output)
            flow.assert_not_called()
        flow.side_effect = FileNotFoundError('missing parquet')
        result = CliRunner().invoke(app,['get-oshime'])
        self.assertEqual(result.exit_code,1)
        self.assertIn('押し目候補を取得できません',result.stderr)
