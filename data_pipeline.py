import argparse
import os

import pyspark

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold


def make_dirs(*directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def build_label_store(date_str, spark):
    bronze_lms = "datamart/bronze/lms/"
    silver_loan_daily = "datamart/silver/loan_daily/"
    gold_label_store = "datamart/gold/label_store/"
    make_dirs(bronze_lms, silver_loan_daily, gold_label_store)

    bronze.process_bronze_table(date_str, bronze_lms, spark, "loan_daily", "data/lms_loan_daily.csv")
    silver.process_silver_table(date_str, bronze_lms, silver_loan_daily, spark)
    # label: 30+ days past due, 6 months after the loan starts
    gold.process_labels_gold_table(date_str, silver_loan_daily, gold_label_store, spark, dpd=30, mob=6)


def build_feature_store(date_str, spark):
    bronze_clickstream = "datamart/bronze/clickstream/"
    bronze_attributes = "datamart/bronze/attributes/"
    bronze_financials = "datamart/bronze/financials/"
    silver_clickstream = "datamart/silver/clickstream/"
    silver_attributes = "datamart/silver/attributes/"
    silver_financials = "datamart/silver/financials/"
    gold_feature_store = "datamart/gold/feature_store/"
    make_dirs(bronze_clickstream, bronze_attributes, bronze_financials,
              silver_clickstream, silver_attributes, silver_financials, gold_feature_store)

    bronze.process_bronze_table(date_str, bronze_clickstream, spark, "clickstream", "data/feature_clickstream.csv")
    bronze.process_bronze_table(date_str, bronze_attributes, spark, "attributes", "data/features_attributes.csv")
    bronze.process_bronze_table(date_str, bronze_financials, spark, "financials", "data/features_financials.csv")

    silver.process_silver_clickstream_table(date_str, bronze_clickstream, silver_clickstream, spark)
    silver.process_silver_attributes_table(date_str, bronze_attributes, silver_attributes, spark)
    silver.process_silver_financials_table(date_str, bronze_financials, silver_financials, spark)

    # the clickstream window only looks back from date_str, so no future data gets in
    gold.process_features_gold_table(date_str, silver_attributes, silver_financials, silver_clickstream,
                                     gold_feature_store, spark, lookback_months=6)


def main(store, date_str):
    spark = (
        pyspark.sql.SparkSession.builder.appName("datamart")
        .master("local[*]")
        # small monthly partitions: 200 shuffle partitions would only add overhead
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        if store == "label":
            build_label_store(date_str, spark)
        else:
            build_feature_store(date_str, spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build the label or feature store for one month")
    parser.add_argument("--store", choices=["label", "feature"], required=True)
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.store, args.snapshotdate)
