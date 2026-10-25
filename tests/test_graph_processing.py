from datetime import date
import json
from unittest import TestCase

import polars as pl
from plotly.utils import PlotlyJSONEncoder
from polars.testing import assert_frame_equal

from wequant.graph_processing import PricelistFig


class VerticalLineTests(TestCase):
    def setUp(self):
        # 日付の入力順に依存せず直前の取引日を選べることも確認する。
        self.chart = PricelistFig(
            1001,
            pricelist_df=pl.DataFrame({
                "code": [1001] * 3,
                "date": [date(2026, 6, 22), date(2026, 6, 18), date(2026, 6, 19)],
                "open": [10.] * 3,
                "high": [11.] * 3,
                "low": [9.] * 3,
                "close": [10.] * 3,
                "volume": [100] * 3,
            }),
            meigaralist_df=pl.DataFrame({"code": [1001], "name": ["テスト"]}),
            start_date=date(2026, 6, 17),
            end_date=date(2026, 6, 23),
        )

    def test_weekend_uses_previous_trading_day_on_both_panels(self):
        before = self.chart.df.clone()
        self.assertIsNone(self.chart.add_vline(date(2026, 6, 20), "orange"))
        self.assertEqual(len(self.chart.fig.layout.shapes), 2)
        self.assertEqual({s.xref for s in self.chart.fig.layout.shapes}, {"x", "x2"})
        for shape in self.chart.fig.layout.shapes:
            self.assertEqual(shape.x0, "2026-06-19")
            self.assertEqual(shape.x1, "2026-06-19")
            self.assertEqual(shape.line.color, "orange")
        assert_frame_equal(self.chart.df, before)

    def test_existing_trading_day_keeps_its_position(self):
        self.chart.add_vline(date(2026, 6, 22))
        self.assertEqual(len(self.chart.fig.layout.shapes), 2)
        for shape in self.chart.fig.layout.shapes:
            self.assertEqual(shape.x0, "2026-06-22")
            self.assertEqual(shape.x1, "2026-06-22")

    def test_no_line_when_outside_period_or_before_first_data(self):
        for target in (date(2026, 6, 16), date(2026, 6, 17), date(2026, 6, 24)):
            with self.subTest(target=target):
                self.assertIsNone(self.chart.add_vline(target))
                self.assertEqual(len(self.chart.fig.layout.shapes), 0)


class MovingAverageTraceTests(TestCase):
    def setUp(self):
        self.prices = pl.DataFrame({
            "code": [1001] * 5,
            "date": [date(2024, 1, day) for day in range(1, 6)],
            "open": [10., 20., 30., 40., 50.],
            "high": [10., 20., 30., 40., 50.],
            "low": [10., 20., 30., 40., 50.],
            "close": [10., 20., 30., 40., 50.],
            "volume": [100] * 5,
        })
        self.chart = PricelistFig(
            1001,
            pricelist_df=self.prices,
            meigaralist_df=pl.DataFrame({"code": [1001], "name": ["テスト"]}),
            start_date=date(2024, 1, 2),
        )

    def test_multiple_periods_and_update_preserve_other_traces_and_data(self):
        before = self.chart.df.clone()
        original_traces = [
            json.dumps(trace.to_plotly_json(), cls=PlotlyJSONEncoder)
            for trace in self.chart.fig.data
        ]
        self.assertIsNone(self.chart.add_trace_moving_average(2))
        self.chart.add_trace_moving_average(3)
        ma2, ma3 = self.chart.fig.data[2:]
        self.assertEqual(list(ma2.y), [None, 25., 35., 45.])
        self.assertEqual(list(ma3.y), [None, None, 30., 40.])
        self.assertEqual(ma2.name, "MA2")
        self.assertEqual(ma3.name, "MA3")
        self.assertEqual(ma2.xaxis, "x")
        self.assertEqual(ma2.yaxis, "y")
        self.assertEqual(list(ma2.x), before["date"].to_list())
        ma2.y = [0] * 4
        self.chart.add_trace_moving_average(2)
        self.assertEqual(len(self.chart.fig.data), 4)
        self.assertEqual(list(ma2.y), [None, 25., 35., 45.])
        self.assertEqual(list(ma3.y), [None, None, 30., 40.])
        for trace, expected in zip(self.chart.fig.data, original_traces):
            self.assertEqual(
                json.dumps(trace.to_plotly_json(), cls=PlotlyJSONEncoder), expected
            )
        assert_frame_equal(self.chart.df, before)

    def test_missing_values_insufficient_history_and_default_period(self):
        self.chart.df = self.chart.df.with_columns(
            pl.Series("close", [20., None, 40., 50.])
        )
        self.chart.add_trace_moving_average(2)
        self.assertEqual(list(self.chart.fig.data[-1].y), [None, None, None, 45.])
        self.chart.add_trace_moving_average()
        self.assertEqual(self.chart.fig.data[-1].name, "MA25")
        self.assertEqual(list(self.chart.fig.data[-1].y), [None] * 4)

    def test_invalid_period_does_not_change_figure(self):
        for term in (0, -1, True, 2.5, "2", None):
            with self.subTest(term=term), self.assertRaises(ValueError):
                self.chart.add_trace_moving_average(term)
        self.assertEqual(len(self.chart.fig.data), 2)
