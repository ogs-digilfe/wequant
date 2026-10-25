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


def reference(df, rate=5., up_days=50):
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

    a, b, d = ma(t, 25), ma(t, 75), ma(t-10, 75)
    if any(v is None for v in (a, b, d)) or not (a > b and b > d):
        return False
    if not all(valid(rows[i][col]) for i, col in ((t,'close'), (t-1,'close'), (t-1,'high'))):
        return False
    if ma(t, 5) is None or ma(t-1, 5) is None:
        return False
    if not (ma(t, 5) > ma(t-1, 5) and rows[t]['close'] > ma(t, 5) and rows[t]['close'] > a
            and rows[t]['close'] > rows[t-1]['high']):
        return False
    if not all(valid(r['high']) for r in rows[-25:]):
        return False
    peak = max(range(t-24,t+1), key=lambda i: (rows[i]['high'], i))
    if peak < 59 or not all(valid(r['high']) for r in rows[peak-59:peak+1]):
        return False
    for i in range(peak-up_days+1, peak+1):
        current, previous = ma(i,25), ma(i-1,25)
        if current is None or previous is None or current <= previous:
            return False
    high = rows[peak]['high']
    if high <= 0 or high != max(r['high'] for r in rows[peak-59:peak+1]) or peak == t:
        return False
    if not all(valid(r['low']) for r in rows[peak+1:]):
        return False
    low_i = min(range(peak+1,t+1), key=lambda i: (rows[i]['low'], -i))
    recovery = next((i for i in range(low_i+1,t+1)
                     if ma(i,5) is not None and ma(i-1,5) is not None
                     and ma(i,5) > ma(i-1,5) and valid(rows[i]['close'])
                     and rows[i]['close'] > ma(i,5) and ma(i,25) is not None
                     and rows[i]['close'] > ma(i,25)), None)
    if recovery != t:
        return False
    low = rows[low_i]['low']
    return (3 <= low_i-peak <= 15 and 1 <= t-low_i <= 5 and low > 0
            and rate <= (high-low)/high*100 < 12
            and (rows[t]['close']-low)/low*100 <= 10)


class OshimeTests(TestCase):
    def matches(self, df, rate=5.):
        return PricelistPl(df).get_oshime_candidates(valuation_date=TARGET, min_decline_rate=rate).height == 1

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
            for rate in (0.,5.,8.,11.99):
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
        result = CliRunner().invoke(app,args+['--min-decline-rate','9'])
        self.assertEqual(result.exit_code,0,result.output)
        self.assertNotIn('130A',result.stdout)

    @patch('wequant.cli.get_oshime_flow')
    def test_cli_defaults_validation_and_errors(self, flow):
        flow.return_value = pl.DataFrame(schema={'code':pl.String,'name':pl.String})
        result = CliRunner().invoke(app,['get-oshime'])
        self.assertEqual(result.exit_code,0,result.output)
        flow.assert_called_once_with(valuation_date=date.today(),min_decline_rate=5.,max_decline_rate=12.,ma25_up_days=50,min_pullback_days=3,max_pullback_days=16,min_low_age_days=1,max_low_age_days=6,max_rebound_rate=10.)
        self.assertIn('該当データなし',result.stderr)
        for rate in ('0','7.5','11.99'):
            flow.reset_mock()
            result = CliRunner().invoke(app,['get-oshime','--min-decline-rate',rate])
            self.assertEqual(result.exit_code,0,result.output)
            flow.assert_called_once_with(valuation_date=date.today(),min_decline_rate=float(rate),max_decline_rate=12.,ma25_up_days=50,min_pullback_days=3,max_pullback_days=16,min_low_age_days=1,max_low_age_days=6,max_rebound_rate=10.)
        for args in (['--valuation-date','invalid'], *[['--min-decline-rate',v] for v in ('-1','12.01','nan','inf')]):
            flow.reset_mock()
            result = CliRunner().invoke(app,['get-oshime',*args])
            self.assertEqual(result.exit_code,2,result.output)
            flow.assert_not_called()
        flow.side_effect = FileNotFoundError('missing parquet')
        result = CliRunner().invoke(app,['get-oshime'])
        self.assertEqual(result.exit_code,1)
        self.assertIn('押し目候補を取得できません',result.stderr)

    def test_ma25_up_period_at_peak_and_pullback_exclusion(self):
        source = prices()
        def match(df, days):
            return PricelistPl(df).get_oshime_candidates(
                valuation_date=TARGET, ma25_up_days=days).height == 1
        # Peak is row 110. A flat MA25 at row 90 is outside 20 comparisons,
        # but inside 21 comparisons. Equality breaks the streak.
        flat = changed(source, 'close', 90, source['close'][65])
        self.assertTrue(match(flat,20))
        self.assertFalse(match(flat,21))
        self.assertFalse(match(changed(source,'close',90,source['close'][65]-1),21))
        # High day itself must rise; subsequent flat/down days are permitted.
        self.assertFalse(match(changed(source,'close',110,source['close'][85]),1))
        for value in (source['close'][86],source['close'][86]-1):
            self.assertTrue(match(changed(source,'close',111,value),20))
        self.assertTrue(match(source,86))  # first valid comparison at row 25
        self.assertFalse(match(source,87))
        for days in (0,-1,1.5,True):
            with self.assertRaises(ValueError):
                match(source,days)

    @patch('wequant.tasks.oshime.MeigaralistPl.from_file')
    @patch('wequant.tasks.oshime.PricelistPl.from_file')
    def test_cli_ma25_up_days_reaches_processing(self, load_prices, load_names):
        load_prices.return_value = PricelistPl(prices())
        load_names.return_value = MeigaralistPl(pl.DataFrame({'code':['130A'],'name':['日本企業']}))
        args = ['get-oshime','--valuation-date',TARGET.isoformat()]
        for days, expected in (('86',True),('87',False)):
            result = CliRunner().invoke(app,args+['--ma25-up-days',days])
            self.assertEqual(result.exit_code,0,result.output)
            self.assertEqual('130A' in result.stdout,expected)
        for days in ('0','-1','1.5'):
            result = CliRunner().invoke(app,args+['--ma25-up-days',days])
            self.assertEqual(result.exit_code,2,result.output)

    def test_decline_range_boundaries_and_validation(self):
        source = prices()
        exact = (130.-source['low'][116])/130.*100
        def result(lo, hi):
            return PricelistPl(source).get_oshime_candidates(
                valuation_date=TARGET, min_decline_rate=lo, max_decline_rate=hi).height
        self.assertEqual(result(exact,exact+1e-8),1)
        self.assertEqual(result(0,exact),0)
        self.assertEqual(result(exact+1e-8,100),0)
        self.assertEqual(result(0,100),1)
        for lo, hi in ((-1,12),(5,5),(6,5),(0,101),(float('nan'),12),(5,float('inf'))):
            with self.assertRaises(ValueError):
                result(lo,hi)

    @patch('wequant.tasks.oshime.MeigaralistPl.from_file')
    @patch('wequant.tasks.oshime.PricelistPl.from_file')
    def test_cli_maximum_propagation(self, load_prices, load_names):
        load_prices.return_value = PricelistPl(prices())
        load_names.return_value = MeigaralistPl(pl.DataFrame({'code':['130A'],'name':['日本企業']}))
        args = ['get-oshime','--valuation-date',TARGET.isoformat(),'--min-decline-rate','0']
        for maximum, expected in (('8',False),('9',True),('100',True)):
            result = CliRunner().invoke(app,args+['--max-decline-rate',maximum])
            self.assertEqual(result.exit_code,0,result.output)
            self.assertEqual('130A' in result.stdout,expected)

    @patch('wequant.cli.get_oshime_flow')
    def test_cli_invalid_decline_ranges_and_removed_option(self, flow):
        for args in (['--decline-rate','5'], ['--min-decline-rate','12'],
                     ['--min-decline-rate','6','--max-decline-rate','5'],
                     ['--max-decline-rate','101'], ['--max-decline-rate','nan'],
                     ['--max-decline-rate','inf'], ['--max-decline-rate','0']):
            result = CliRunner().invoke(app,['get-oshime',*args])
            self.assertEqual(result.exit_code,2,result.output)
            flow.assert_not_called()

    def test_pullback_low_need_not_be_near_ma25(self):
        source = changed(prices(),'low',116,115.)
        low_i = 116
        avg = source['close'].slice(low_i-24,25).mean()
        self.assertLess(source['low'][low_i]/avg,.97)
        self.assertTrue(reference(source))
        self.assertTrue(self.matches(source))

    def test_pullback_days_boundaries_and_validation(self):
        # Peak row 110, trough row 116: exactly six trading days.
        def count(lo, hi):
            return PricelistPl(prices()).get_oshime_candidates(
                valuation_date=TARGET, min_pullback_days=lo, max_pullback_days=hi).height
        self.assertEqual(count(6,7),1)
        self.assertEqual(count(1,6),0)
        self.assertEqual(count(7,16),0)
        for lo, hi in ((0,16),(-1,16),(3,3),(4,3),(1.5,16),(True,16),(3,16.5)):
            with self.assertRaises(ValueError):
                count(lo,hi)

    @patch('wequant.tasks.oshime.MeigaralistPl.from_file')
    @patch('wequant.tasks.oshime.PricelistPl.from_file')
    def test_cli_pullback_days_propagation(self, load_prices, load_names):
        load_prices.return_value = PricelistPl(prices())
        load_names.return_value = MeigaralistPl(pl.DataFrame({'code':['130A'],'name':['日本企業']}))
        args = ['get-oshime','--valuation-date',TARGET.isoformat()]
        for lo, hi, expected in ((6,7,True),(1,6,False),(7,16,False)):
            result = CliRunner().invoke(app,args+['--min-pullback-days',str(lo),'--max-pullback-days',str(hi)])
            self.assertEqual(result.exit_code,0,result.output)
            self.assertEqual('130A' in result.stdout,expected)

    @patch('wequant.cli.get_oshime_flow')
    def test_cli_pullback_days_invalid(self, flow):
        for args in (['--min-pullback-days','0'],['--max-pullback-days','0'],
                     ['--min-pullback-days','3.5'],['--max-pullback-days','16.5'],
                     ['--min-pullback-days','16'],
                     ['--min-pullback-days','7','--max-pullback-days','6']):
            result = CliRunner().invoke(app,['get-oshime',*args])
            self.assertEqual(result.exit_code,2,result.output)
            flow.assert_not_called()

    def test_low_age_days_boundaries_and_validation(self):
        # Trough row 116, target row 119: exactly three trading days.
        def count(lo, hi):
            return PricelistPl(prices()).get_oshime_candidates(
                valuation_date=TARGET, min_low_age_days=lo, max_low_age_days=hi).height
        self.assertEqual(count(3,4),1)
        self.assertEqual(count(1,3),0)
        self.assertEqual(count(4,6),0)
        for lo, hi in ((0,6),(-1,6),(3,3),(4,3),(1.5,6),(True,6),(1,6.5),(1,False)):
            with self.assertRaises(ValueError):
                count(lo,hi)

    @patch('wequant.tasks.oshime.MeigaralistPl.from_file')
    @patch('wequant.tasks.oshime.PricelistPl.from_file')
    def test_cli_low_age_days_propagation(self, load_prices, load_names):
        load_prices.return_value = PricelistPl(prices())
        load_names.return_value = MeigaralistPl(pl.DataFrame({'code':['130A'],'name':['日本企業']}))
        args = ['get-oshime','--valuation-date',TARGET.isoformat()]
        for lo, hi, expected in ((3,4,True),(1,3,False),(4,6,False)):
            result = CliRunner().invoke(app,args+['--min-low-age-days',str(lo),'--max-low-age-days',str(hi)])
            self.assertEqual(result.exit_code,0,result.output)
            self.assertEqual('130A' in result.stdout,expected)

    @patch('wequant.cli.get_oshime_flow')
    def test_cli_low_age_days_invalid(self, flow):
        for args in (['--min-low-age-days','0'],['--max-low-age-days','0'],
                     ['--min-low-age-days','1.5'],['--max-low-age-days','6.5'],
                     ['--min-low-age-days','6'],
                     ['--min-low-age-days','4','--max-low-age-days','3']):
            result = CliRunner().invoke(app,['get-oshime',*args])
            self.assertEqual(result.exit_code,2,result.output)
            flow.assert_not_called()

    def test_rebound_rate_inclusive_boundary_and_validation(self):
        source = prices()
        exact = (125.-source['low'][116])/source['low'][116]*100
        def count(rate):
            return PricelistPl(source).get_oshime_candidates(
                valuation_date=TARGET,max_rebound_rate=rate).height
        self.assertEqual(count(exact),1)
        self.assertEqual(count(exact-1e-8),0)
        self.assertEqual(count(exact+1e-8),1)
        self.assertEqual(count(0),0)
        for rate in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                count(rate)
        # A rebound above the old 5% limit is included by the new default.
        higher = changed(source,'close',119,127.)
        self.assertTrue(self.matches(higher))
        self.assertEqual(PricelistPl(higher).get_oshime_candidates(
            valuation_date=TARGET,max_rebound_rate=5.).height,0)

    @patch('wequant.tasks.oshime.MeigaralistPl.from_file')
    @patch('wequant.tasks.oshime.PricelistPl.from_file')
    def test_cli_rebound_rate_propagation(self, load_prices, load_names):
        load_prices.return_value = PricelistPl(prices())
        load_names.return_value = MeigaralistPl(pl.DataFrame({'code':['130A'],'name':['日本企業']}))
        args = ['get-oshime','--valuation-date',TARGET.isoformat()]
        for rate, expected in (('4.5',False),('5',True),('0',False),('101',True)):
            result = CliRunner().invoke(app,args+['--max-rebound-rate',rate])
            self.assertEqual(result.exit_code,0,result.output)
            self.assertEqual('130A' in result.stdout,expected)

    @patch('wequant.cli.get_oshime_flow')
    def test_cli_rebound_rate_invalid(self, flow):
        for rate in ('-1','nan','inf','invalid'):
            result = CliRunner().invoke(app,['get-oshime','--max-rebound-rate',rate])
            self.assertEqual(result.exit_code,2,result.output)
            flow.assert_not_called()

    def test_ma5_need_not_exceed_ma25(self):
        source = prices().with_columns((pl.col('close')*4).round()/4)
        values = source['close'].to_list()
        equal_close = (sum(values[-25:-1])-5*sum(values[-5:-1]))/4
        equal = changed(source,'close',119,equal_close)
        self.assertEqual(equal['close'].tail(5).mean(),equal['close'].tail(25).mean())
        self.assertTrue(reference(equal))
        self.assertTrue(self.matches(equal))
        below = changed(equal,'close',119,equal_close-.25)
        self.assertLess(below['close'].tail(5).mean(),below['close'].tail(25).mean())
        self.assertTrue(reference(below))
        self.assertTrue(self.matches(below))
        above = changed(equal,'close',119,equal_close+2.)
        self.assertTrue(reference(above))
        self.assertTrue(self.matches(above))

    def test_ma5_recovery_requires_rising_ma_and_first_recovery_today(self):
        source = prices()
        self.assertTrue(self.matches(source))
        # SMA5 rises precisely when today's close exceeds the close five rows ago.
        for old_close in (125.,126.):
            df = changed(source,'close',114,old_close)
            self.assertGreater(df['close'][-1],df['close'].tail(5).mean())
            self.assertGreater(df['close'].tail(5).mean(),df['close'].tail(25).mean())
            self.assertFalse(self.matches(df))
        # A previous day already satisfying both recovery conditions excludes today.
        df = changed(source,'close',118,124.)
        self.assertFalse(self.matches(df))

    def test_delayed_ma5_turn_after_price_cross_is_included(self):
        # Price is already above MA5 at row 118, but MA5 is flat on that day.
        source = changed(changed(prices(),'close',118,124.),'close',113,124.)
        values = source['close'].to_list()
        self.assertGreater(values[118],sum(values[114:119])/5)
        self.assertEqual(sum(values[114:119]),sum(values[113:118]))
        self.assertTrue(reference(source))
        self.assertTrue(self.matches(source))

    def test_first_recovery_ignores_other_filters_and_excludes_low_day(self):
        source = changed(prices(),'close',118,124.)
        # First recovery (118) fails the previous-high breakout condition.
        # It remains the first recovery, so target 119 must not be selected.
        source = changed(source,'high',117,124.5)
        self.assertFalse(reference(source))
        self.assertFalse(self.matches(source))
        # Move the low to 118: recovery on the low day itself does not count.
        source = changed(source,'low',118,119.)
        self.assertTrue(reference(source))
        self.assertTrue(self.matches(source))

    def test_recovery_waits_for_close_above_ma25(self):
        source = changed(prices(),'close',118,124.).with_columns(
            pl.when(pl.col('date') < START+timedelta(days=110))
            .then(pl.col('close')+5).otherwise(pl.col('close')).alias('close'))
        values = source['close'].to_list()
        self.assertGreater(values[118],sum(values[114:119])/5)
        self.assertGreater(sum(values[114:119]),sum(values[113:118]))
        self.assertLess(values[118],sum(values[94:119])/25)
        self.assertTrue(reference(source))
        self.assertTrue(self.matches(source))

    def test_close_must_strictly_exceed_ma25(self):
        source = prices().with_columns(pl.col('close').round())
        values = source['close'].to_list()
        # Make the preceding 24 closes sum to 24 * today's close exactly.
        source = changed(source,'close',100,values[100]+24*125.-sum(values[95:119]))
        source = changed(source,'close',119,125.)
        self.assertEqual(source['close'][-1],source['close'].tail(25).mean())
        self.assertFalse(self.matches(source))
        below = changed(source,'close',119,124.75)
        self.assertFalse(self.matches(below))
        above = changed(source,'close',119,125.25)
        self.assertTrue(reference(above))
        self.assertTrue(self.matches(above))
