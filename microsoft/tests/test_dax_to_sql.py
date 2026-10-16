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

"""Tests for the opt-in DAX-to-OSSIE_SQL measure translator.

The translator must translate exactly the six single-column aggregates whose
SQL equivalent is unambiguous, and refuse everything else with a reason. The
refusals matter as much as the translations: a wrong translation is a silent
wrong number on another engine.
"""

import pytest

from ossie_microsoft._dax_to_sql import translate_measure


def resolve(table, column):
    """A resolver over a small two-dataset model, mirroring the real converter:
    the DAX table name is the dataset name and the DAX column name is the field
    name. Unknown table/column pairs do not resolve."""
    model = {
        ("Sales", "Amount"): ("Sales", "Amount"),
        ("Sales", "Customer"): ("Sales", "Customer"),
        ("Sales", "Status"): ("Sales", "Status"),
        ("Order Lines", "Qty"): ("Order Lines", "Qty"),  # table name needs quoting
    }
    return model.get((table, column))


@pytest.mark.parametrize(
    "dax, expected",
    [
        ("SUM('Sales'[Amount])", "SUM(Sales.Amount)"),
        ("AVERAGE('Sales'[Amount])", "AVG(Sales.Amount)"),
        ("MIN('Sales'[Amount])", "MIN(Sales.Amount)"),
        ("MAX('Sales'[Amount])", "MAX(Sales.Amount)"),
        ("COUNTA('Sales'[Customer])", "COUNT(Sales.Customer)"),
        ("DISTINCTCOUNTNOBLANK('Sales'[Customer])", "COUNT(DISTINCT Sales.Customer)"),
    ],
)
def test_translates_the_six_aggregates(dax, expected):
    sql, reason = translate_measure(dax, resolve)
    assert reason is None
    assert sql == expected


def test_bare_table_reference_is_accepted():
    sql, reason = translate_measure("SUM(Sales[Amount])", resolve)
    assert reason is None
    assert sql == "SUM(Sales.Amount)"


def test_leading_equals_and_whitespace_are_tolerated():
    sql, reason = translate_measure("  =  SUM( 'Sales'[Amount] )  ", resolve)
    assert reason is None
    assert sql == "SUM(Sales.Amount)"


@pytest.mark.parametrize(
    "dax",
    [
        "COUNT('Sales'[Amount])",            # TRUE/FALSE unsupported in DAX
        "DISTINCTCOUNT('Sales'[Customer])",  # BLANK counted as distinct
        "COUNTROWS('Sales')",                # row count, not single column
        "MEDIAN('Sales'[Amount])",           # later tier
        "SUMX('Sales', 'Sales'[Amount])",    # iterator, two args
        "CALCULATE(SUM('Sales'[Amount]), 'Sales'[Status] = \"done\")",  # filter context
        "DIVIDE(SUM('Sales'[Amount]), COUNTA('Sales'[Customer]))",      # division
        "SUM('Sales'[Amount]) + SUM('Sales'[Amount])",                  # arithmetic
        "SUM('Sales'[Amount]) * 1.1",        # arithmetic
        "IF('Sales'[Amount] > 0, 1, 0)",     # scalar logic
    ],
)
def test_refuses_everything_outside_the_safe_set(dax):
    sql, reason = translate_measure(dax, resolve)
    assert sql is None
    assert reason  # a non-empty reason is always given


def test_count_and_distinctcount_refusals_name_the_semantic_difference():
    _, count_reason = translate_measure("COUNT('Sales'[Amount])", resolve)
    _, distinct_reason = translate_measure("DISTINCTCOUNT('Sales'[Customer])", resolve)
    assert "TRUE/FALSE" in count_reason
    assert "BLANK" in distinct_reason


def test_unqualified_column_is_refused():
    sql, reason = translate_measure("SUM([Amount])", resolve)
    assert sql is None
    assert "fully qualified" in reason


def test_unresolved_column_is_refused():
    sql, reason = translate_measure("SUM('Ghost'[Nope])", resolve)
    assert sql is None
    assert "does not resolve" in reason


def test_name_needing_quoting_is_refused_in_this_tier():
    # 'Order Lines' resolves, but the dataset name has a space and would need
    # SQL quoting, which this first tier declines rather than risk getting wrong.
    sql, reason = translate_measure("SUM('Order Lines'[Qty])", resolve)
    assert sql is None
    assert "quoting" in reason


def test_empty_measure_is_refused():
    sql, reason = translate_measure("   ", resolve)
    assert sql is None
    assert reason


def test_nested_call_does_not_parse_as_single_aggregate():
    sql, reason = translate_measure("SUM(ROUND('Sales'[Amount], 2))", resolve)
    assert sql is None
    assert reason
