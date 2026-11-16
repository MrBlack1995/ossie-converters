# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Tests for filtered aggregates: CALCULATE with KEEPFILTERS or FILTER, and the
X-iterators over FILTER(table, condition). Only filters that narrow the context are
translated; each becomes ``aggregate(...) FILTER (WHERE condition)``."""

import sqlite3

import pytest
import sqlglot

from ossie_microsoft._dax_to_sql import translate_measure


def resolve(table, column):
    model = {
        ("Sales", "Qty"): ("Sales", "Qty", "Integer"),
        ("Sales", "Price"): ("Sales", "Price", "Decimal"),
        ("Sales", "Region"): ("Sales", "Region", "String"),
        ("Sales", "Country"): ("Sales", "Country", "String"),
        ("Sales", "Paid"): ("Sales", "Paid", "Boolean"),
        ("Product", "Cost"): ("Product", "Cost", "Decimal"),
    }
    return model.get((table, column))


MEASURES = {"Total Qty": "SUM('Sales'[Qty])"}


def _parses_everywhere(sql):
    for dialect in (None, "databricks", "snowflake", "postgres", "duckdb"):
        sqlglot.parse_one(sql, read=dialect)


@pytest.mark.parametrize(
    ("dax", "sql"),
    [
        # KEEPFILTERS intersects with the outer filter, as a per-row condition does.
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS('Sales'[Region] = \"EU\"))",
         "SUM(Sales.Qty) FILTER (WHERE COALESCE(Sales.Region, '') ILIKE 'EU')"),
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS('Sales'[Qty] > 1))",
         "SUM(Sales.Qty) FILTER (WHERE COALESCE(Sales.Qty, 0) > 1)"),
        # FILTER(table, condition) runs in the current filter context, so it only narrows it.
        ("CALCULATE(AVERAGE('Sales'[Price]), FILTER('Sales', 'Sales'[Qty] >= 2))",
         "AVG(Sales.Price) FILTER (WHERE COALESCE(Sales.Qty, 0) >= 2)"),
        # Several filters, and nested CALCULATE, combine with AND.
        ("CALCULATE(MAX('Sales'[Price]), KEEPFILTERS('Sales'[Region] = \"EU\"), "
         "KEEPFILTERS('Sales'[Qty] > 1))",
         "MAX(Sales.Price) FILTER (WHERE COALESCE(Sales.Region, '') ILIKE 'EU' AND "
         "COALESCE(Sales.Qty, 0) > 1)"),
        ("CALCULATE(CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS('Sales'[Qty] > 1)), "
         "KEEPFILTERS('Sales'[Region] <> \"EU\"))",
         "SUM(Sales.Qty) FILTER (WHERE NOT (COALESCE(Sales.Region, '') ILIKE 'EU') AND "
         "COALESCE(Sales.Qty, 0) > 1)"),
        # Every aggregate in the expression is filtered, also inside a referenced measure.
        ("CALCULATE(DIVIDE([Total Qty], SUM('Sales'[Price])), KEEPFILTERS('Sales'[Qty] > 1))",
         "SUM(Sales.Qty) FILTER (WHERE COALESCE(Sales.Qty, 0) > 1) / "
         "NULLIF(SUM(Sales.Price) FILTER (WHERE COALESCE(Sales.Qty, 0) > 1), 0)"),
        # Without a filter argument CALCULATE changes nothing outside a row context.
        ("CALCULATE(SUM('Sales'[Qty]))", "SUM(Sales.Qty)"),
        # An iterator over FILTER(table, condition): the aggregate over the kept rows.
        ("SUMX(FILTER('Sales', 'Sales'[Region] <> \"EU\"), 'Sales'[Qty] * 'Sales'[Price])",
         "SUM(Sales.Qty * Sales.Price) FILTER (WHERE NOT (COALESCE(Sales.Region, '') ILIKE "
         "'EU'))"),
        ("MINX(FILTER(Sales, Sales[Qty] > 0 && Sales[Region] = \"US\"), Sales[Price])",
         "MIN(Sales.Price) FILTER (WHERE COALESCE(Sales.Qty, 0) > 0 AND "
         "COALESCE(Sales.Region, '') ILIKE 'US')"),
        ("CALCULATE(PERCENTILE.INC('Sales'[Price], 0.5), KEEPFILTERS('Sales'[Qty] > 1))",
         "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Sales.Price) FILTER (WHERE "
         "COALESCE(Sales.Qty, 0) > 1)"),
        # A variable inside the condition is translated in its row context.
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS(VAR floor = 1 RETURN 'Sales'[Qty] > floor))",
         "SUM(Sales.Qty) FILTER (WHERE COALESCE(Sales.Qty, 0) > 1)"),
        # A variable keeps the filter of where it is defined: total is not filtered.
        ("VAR total = SUM('Sales'[Qty]) RETURN "
         "CALCULATE(DIVIDE(SUM('Sales'[Qty]), total), KEEPFILTERS('Sales'[Region] = \"US\"))",
         "SUM(Sales.Qty) FILTER (WHERE COALESCE(Sales.Region, '') ILIKE 'US') / "
         "NULLIF(SUM(Sales.Qty), 0)"),
    ],
)
def test_filtered_aggregates_translate(dax, sql):
    result, reason = translate_measure(dax, resolve, MEASURES)
    assert reason is None, reason
    assert result == sql
    _parses_everywhere(result)


@pytest.mark.parametrize(
    ("dax", "reason"),
    [
        # A plain filter replaces the outer filter on its column.
        ("CALCULATE(SUM('Sales'[Qty]), 'Sales'[Region] = \"EU\")", "replaces the outer filter"),
        ("CALCULATE(SUM('Sales'[Qty]), ALL('Sales'))", "removes filters"),
        ("CALCULATE(SUM('Sales'[Qty]), REMOVEFILTERS('Sales'[Region]))", "removes filters"),
        # Over no matching row SQL counts 0 and DAX returns BLANK.
        ("CALCULATE(COUNTA('Sales'[Qty]), KEEPFILTERS('Sales'[Qty] > 1))", "counts 0"),
        ("CALCULATE(DISTINCTCOUNT('Sales'[Region]), KEEPFILTERS('Sales'[Qty] > 1))", "counts 0"),
        ("COUNTX(FILTER('Sales', 'Sales'[Qty] > 1), 'Sales'[Region])", "counts 0"),
        # A filter on one table does not restrict another table without relationships.
        ("CALCULATE(SUM('Product'[Cost]), KEEPFILTERS('Sales'[Qty] > 1))", "follow relationships"),
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS('Sales'[Qty] > 'Product'[Cost]))",
         "several tables"),
        # A measure in the condition is evaluated with context transition.
        ("SUMX(FILTER('Sales', [Total Qty] > 1), 'Sales'[Qty])", "context transition"),
        ("SUMX('Sales', CALCULATE(SUM('Sales'[Qty])))", "context transition"),
        ("CALCULATE(SUM('Sales'[Qty]), FILTER(ALL('Sales'), 'Sales'[Qty] > 1))",
         "table named directly"),
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS(TRUE()))", "reads no column"),
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS(FILTER('Sales', 'Sales'[Paid])))",
         "not a condition"),
        # ILIKE wildcards in the literal are refused rather than escaped.
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS('Sales'[Region] = \"E%\"))", "wildcards"),
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS('Sales'[Qty] = \"5\"))", "with text"),
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS('Sales'[Region] < \"M\"))", "text value"),
        ("CALCULATE()", "takes an expression"),
        ("CALCULATE(CALCULATE(SUM('Sales'[Qty]), ALL('Sales')))", "removes filters"),
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS('Sales'[Qty] > 1, 'Sales'[Qty] < 9))",
         "KEEPFILTERS takes one filter"),
        ("CALCULATE(SUM('Sales'[Qty]), TRUE())", "filter argument is a later tier"),
        ("CALCULATE(SUM('Sales'[Qty]), FILTER('Sales'))", "FILTER takes a table and a condition"),
        ("CALCULATE(SUM('Sales'[Qty]), FILTER('Sales', 1 > 0))", "reads no column of 'Sales'"),
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS(-'Sales'[Qty] > 1))", "unary"),
        ("CALCULATE(FOO('Sales'[Qty]))", "the expression in CALCULATE is not translated"),
    ],
)
def test_filtered_aggregate_refusals(dax, reason):
    result, why = translate_measure(dax, resolve, MEASURES)
    assert result is None
    assert reason in why, why


# --- executed: the SQL returns DAX's numbers, per country -------------------------------

ROWS = [
    # Region, Country, Qty, Price
    ("EU", "DE", 2, 10.0),
    ("eu", "FR", None, 5.0),   # DAX compares text without regard to case
    ("US", "US", 5, 1.0),
    (None, "DE", 1, 4.0),      # a BLANK region is "" for DAX
    ("US", "CA", 3, 2.0),
]


def _by_country(sql):
    """Run the measure per country, as a visual grouped by Sales[Country] would. SQLite has
    no ILIKE; its LIKE is case-insensitive for ASCII, which these rows stay within."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE Sales (Region TEXT, Country TEXT, Qty INTEGER, Price REAL)"
        )
        connection.executemany("INSERT INTO Sales VALUES (?, ?, ?, ?)", ROWS)
        query = f"SELECT Country, {sql.replace(' ILIKE ', ' LIKE ')} FROM Sales GROUP BY Country"
        return dict(connection.execute(query).fetchall())
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("dax", "dax_result"),
    [
        # EU and eu match; DE's BLANK-region row does not; FR's only EU row has a BLANK Qty.
        ("CALCULATE(SUM('Sales'[Qty]), KEEPFILTERS('Sales'[Region] = \"EU\"))",
         {"DE": 2, "FR": None, "US": None, "CA": None}),
        # A BLANK region is "" <> "EU", so DE's second row counts.
        ("SUMX(FILTER('Sales', 'Sales'[Region] <> \"EU\"), 'Sales'[Qty] * 'Sales'[Price])",
         {"DE": 4.0, "FR": None, "US": 5.0, "CA": 6.0}),
        # A BLANK Qty compares as 0.
        ("CALCULATE(AVERAGE('Sales'[Price]), FILTER('Sales', 'Sales'[Qty] >= 2))",
         {"DE": 10.0, "FR": None, "US": 1.0, "CA": 2.0}),
        # The variable is evaluated before CALCULATE, so total is the unfiltered sum.
        ("VAR total = SUM('Sales'[Qty]) RETURN "
         "CALCULATE(DIVIDE(SUM('Sales'[Qty]), total), KEEPFILTERS('Sales'[Region] = \"US\"))",
         {"DE": None, "FR": None, "US": 1.0, "CA": 1.0}),
    ],
)
def test_filtered_sql_returns_the_dax_result_per_group(dax, dax_result):
    sql, reason = translate_measure(dax, resolve, MEASURES)
    assert reason is None, reason
    assert _by_country(sql) == dax_result
