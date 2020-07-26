import sys
from collections import OrderedDict

import pyspark.sql.functions as func
import pyspark.sql.types as T
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *


def load_trades(spark):
    data = [
        (10, 1546300800000, 37.50, 100.000),
        (10, 1546300801000, 37.51, 100.000),
        (20, 1546300804000, 12.67, 300.000),
        (10, 1546300807000, 37.50, 200.000),
    ]
    schema = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("timestamp", T.LongType()),
            T.StructField("price", T.DoubleType()),
            T.StructField("quantity", T.DoubleType()),
        ]
    )

    return spark.createDataFrame(data, schema)


def load_prices(spark):
    data = [
        (10, 1546300799000, 37.50, 37.51),
        (10, 1546300802000, 37.51, 37.52),
        (10, 1546300806000, 37.50, 37.51),
    ]
    schema = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("timestamp", T.LongType()),
            T.StructField("bid", T.DoubleType()),
            T.StructField("ask", T.DoubleType()),
        ]
    )

    return spark.createDataFrame(data, schema)


def fill_not_null_values(elements):
    ordered_dict = OrderedDict()
    for element in elements:
        ordered_dict[element.timestamp] = element
    bid = None
    ask = None
    price = None
    quantity = None
    for (key, element) in ordered_dict.items():
        if element.bid is not None:
            bid = element.bid
        if element.ask is not None:
            ask = element.ask
        if element.price is not None:
            price = element.price
        if element.quantity is not None:
            quantity = element.quantity
        row = T.Row(id=element.id, timestamp=element.timestamp, bid=bid, ask=ask, price=price, quantity=quantity)
        ordered_dict[key] = row
    return ordered_dict.values()


def fill(trades, prices):
    """
    Combine the sets of events and fill forward the value columns so that each
    row has the most recent non-null value for the corresponding id. For
    example, given the above input tables the expected output is:

    +---+-------------+-----+-----+-----+--------+
    | id|    timestamp|  bid|  ask|price|quantity|
    +---+-------------+-----+-----+-----+--------+
    | 10|1546300799000| 37.5|37.51| null|    null|
    | 10|1546300800000| 37.5|37.51| 37.5|   100.0|
    | 10|1546300801000| 37.5|37.51|37.51|   100.0|
    | 10|1546300802000|37.51|37.52|37.51|   100.0|
    | 20|1546300804000| null| null|12.67|   300.0|
    | 10|1546300806000| 37.5|37.51|37.51|   100.0|
    | 10|1546300807000| 37.5|37.51| 37.5|   200.0|
    +---+-------------+-----+-----+-----+--------+

    :param trades: DataFrame of trade events
    :param prices: DataFrame of price events
    :return: A DataFrame of the combined events and filled.
    """

    result = trades. \
        join(prices, ['id', 'timestamp'], 'outer'). \
        select('id', 'timestamp', 'bid', 'ask', 'price', 'quantity'). \
        orderBy(asc("timestamp"))
    fill_bid_with_not_null = result. \
        withColumn("bid", func.last('bid', True).over(
        Window.partitionBy('id').orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
        withColumn("ask", func.last('ask', True).over(
        Window.partitionBy('id').orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
        withColumn("price", func.last('price', True).over(
        Window.partitionBy('id').orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
        withColumn("quantity", func.last('quantity', True).over(
        Window.partitionBy('id').orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
        orderBy('timestamp')

    return fill_bid_with_not_null


# def fill(trades, prices, spark):
#     """
#     Combine the sets of events and fill forward the value columns so that each
#     row has the most recent non-null value for the corresponding id. For
#     example, given the above input tables the expected output is:
#
#     +---+-------------+-----+-----+-----+--------+
#     | id|    timestamp|  bid|  ask|price|quantity|
#     +---+-------------+-----+-----+-----+--------+
#     | 10|1546300799000| 37.5|37.51| null|    null|
#     | 10|1546300800000| 37.5|37.51| 37.5|   100.0|
#     | 10|1546300801000| 37.5|37.51|37.51|   100.0|
#     | 10|1546300802000|37.51|37.52|37.51|   100.0|
#     | 20|1546300804000| null| null|12.67|   300.0|
#     | 10|1546300806000| 37.5|37.51|37.51|   100.0|
#     | 10|1546300807000| 37.5|37.51| 37.5|   200.0|
#     +---+-------------+-----+-----+-----+--------+
#
#     :param trades: DataFrame of trade events
#     :param prices: DataFrame of price events
#     :return: A DataFrame of the combined events and filled.
#     """
#
#     result = trades.\
#         join(prices, ['id', 'timestamp'], 'outer').\
#         select('id', 'timestamp', 'bid', 'ask', 'price', 'quantity').\
#         orderBy(asc("timestamp"))
#     filled_with_not_null_values = result.rdd. \
#         map(lambda x: (x.id, x)). \
#         groupByKey(). \
#         mapValues(lambda e: fill_not_null_values(e)). \
#         flatMapValues(lambda x: x).values()
#     filled_with_not_null_values.toDF().schema()
#     return spark.createDataFrame(filled_with_not_null_values, schema=T.StructType(
#         [
#             T.StructField("id", T.LongType()),
#             T.StructField("timestamp", T.LongType()),
#             T.StructField("bid", T.DoubleType()),
#             T.StructField("ask", T.DoubleType()),
#             T.StructField("price", T.DoubleType()),
#             T.StructField("quantity", T.DoubleType()),
#         ]
#     )).orderBy('timestamp')
#

def pivot(trades, prices):
    """
    Pivot and fill the columns on the event id so that each row contains a
    column for each id + column combination where the value is the most recent
    non-null value for that id. For example, given the above input tables the
    expected output is:

    +---+-------------+-----+-----+-----+--------+------+------+--------+-----------+------+------+--------+-----------+
    | id|    timestamp|  bid|  ask|price|quantity|10_bid|10_ask|10_price|10_quantity|20_bid|20_ask|20_price|20_quantity|
    +---+-------------+-----+-----+-----+--------+------+------+--------+-----------+------+------+--------+-----------+
    | 10|1546300799000| 37.5|37.51| null|    null|  37.5| 37.51|    null|       null|  null|  null|    null|       null|
    | 10|1546300800000| null| null| 37.5|   100.0|  37.5| 37.51|    37.5|      100.0|  null|  null|    null|       null|
    | 10|1546300801000| null| null|37.51|   100.0|  37.5| 37.51|   37.51|      100.0|  null|  null|    null|       null|
    | 10|1546300802000|37.51|37.52| null|    null| 37.51| 37.52|   37.51|      100.0|  null|  null|    null|       null|
    | 20|1546300804000| null| null|12.67|   300.0| 37.51| 37.52|   37.51|      100.0|  null|  null|   12.67|      300.0|
    | 10|1546300806000| 37.5|37.51| null|    null|  37.5| 37.51|   37.51|      100.0|  null|  null|   12.67|      300.0|
    | 10|1546300807000| null| null| 37.5|   200.0|  37.5| 37.51|    37.5|      200.0|  null|  null|   12.67|      300.0|
    +---+-------------+-----+-----+-----+--------+------+------+--------+-----------+------+------+--------+-----------+

    :param trades: DataFrame of trade events
    :param prices: DataFrame of price events
    :return: A DataFrame of the combined events and pivoted columns.
    """
    result = trades. \
        join(prices, ['id', 'timestamp'], 'outer'). \
        select('id', 'timestamp', 'bid', 'ask', 'price', 'quantity'). \
        orderBy(asc("timestamp"))
    ids = result.select('id').distinct().collect()
    print(ids)
    # result.groupBy().pivot("id", Seq(ids)).
    final = None

    schema=T.StructType(
            [
                T.StructField("id", T.LongType()),
                T.StructField("timestamp", T.LongType()),
                T.StructField("bid", T.DoubleType()),
                T.StructField("ask", T.DoubleType()),
                T.StructField("price", T.DoubleType()),
                T.StructField("quantity", T.DoubleType()),
            ]
        )
    for row in ids:
        id = str(row.id)
        # filter(result.id == row.id). \
        dyn_columns = result. \
            withColumn("bid", when(col("id") != row.id, lit(None).cast(T.DoubleType())).otherwise(lit(col('bid')).cast(T.DoubleType()))).\
            withColumn("ask", when(col("id") != row.id, lit(None).cast(T.DoubleType())).otherwise(lit(col('ask')).cast(T.DoubleType()))).\
            withColumn("price", when(col("id") != row.id, lit(None).cast(T.DoubleType())).otherwise(lit(col('price')).cast(T.DoubleType()))).\
            withColumn("quantity", when(col("id") != row.id, lit(None).cast(T.DoubleType())).otherwise(lit(col('quantity')).cast(T.DoubleType()))).\
            withColumn(id+"_id", when(col("id") == row.id, lit(id).cast(T.IntegerType())).otherwise(lit(id).cast(T.IntegerType()))).\
            withColumn(id + "_bid", func.last('bid', True).over(
            Window.partitionBy(id+"_id").orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
            withColumn(id + "_ask", func.last('ask', True).over(
            Window.partitionBy(id+"_id").orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
            withColumn(id + "_price", func.last('price', True).over(
            Window.partitionBy(id+"_id").orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
            withColumn(id + "_quantity", func.last('quantity', True).over(
            Window.partitionBy(id+"_id").orderBy('timestamp').rowsBetween(-sys.maxsize, 0))).\
            drop('bid', 'ask', 'price', 'quantity', id + "_id")
        schema.add(id + "_bid", T.DoubleType)
        schema.add(id + "_ask", T.DoubleType)
        schema.add(id + "_price", T.DoubleType)
        schema.add(id + "_quantity", T.DoubleType)
        # result = result.join(dyn_columns, how='right')
        # result1 = result.join(dyn_columns, on=['id', 'timestamp'], how='left_outer')
        # dyn_columns.show()
        if final is None:
            final = dyn_columns
        else:
            final = final.join(dyn_columns, ['id', 'timestamp'], how='outer')
            # final = final.join(dyn_columns, ['id', 'timestamp'], how='outer').select(final["*"],dyn_columns[id + "_bid"],dyn_columns[id + "_ask"],dyn_columns[id + "_price"],dyn_columns[id + "_quantity"])
        final.show()

    # result.orderBy(asc("timestamp"))
    final_result = final.join(result, ['id', 'timestamp'], how='outer').c
    final_result.orderBy('timestamp').show()
    result.show()
    # result = trades. \
    #     join(prices, ['id', 'timestamp'], 'outer'). \
    #     select('id', 'timestamp', 'bid', 'ask', 'price', 'quantity'). \
    #     orderBy(asc("timestamp"))
    # fill_bid_with_not_null = result. \
    #     withColumn("bid", func.last('bid', True).over(
    #     Window.partitionBy('id').orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
    #     withColumn("ask", func.last('ask', True).over(
    #     Window.partitionBy('id').orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
    #     withColumn("price", func.last('price', True).over(
    #     Window.partitionBy('id').orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
    #     withColumn("quantity", func.last('quantity', True).over(
    #     Window.partitionBy('id').orderBy('timestamp').rowsBetween(-sys.maxsize, 0))). \
    #     orderBy('timestamp')


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    trades = load_trades(spark)
    trades.show()

    prices = load_prices(spark)
    prices.show()

    fill(trades, prices).show()

    pivot(trades, prices)