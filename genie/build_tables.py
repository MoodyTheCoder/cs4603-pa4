"""Create structured Delta table from Meridian financial data."""
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

# Financial data (extracted from annual_report.pdf)
data = [
    Row(year=2019, net_revenue_billion=11.28, operating_profit_billion=0.89, net_income_billion=0.55),
    Row(year=2020, net_revenue_billion=12.50, operating_profit_billion=0.92, net_income_billion=0.58),
    Row(year=2021, net_revenue_billion=13.80, operating_profit_billion=0.98, net_income_billion=0.62),
    Row(year=2022, net_revenue_billion=14.55, operating_profit_billion=0.905, net_income_billion=0.68),
    Row(year=2023, net_revenue_billion=16.91, operating_profit_billion=1.124, net_income_billion=0.75),
]

schema = StructType([
    StructField("year", IntegerType(), False),
    StructField("net_revenue_billion", DoubleType(), False),
    StructField("operating_profit_billion", DoubleType(), False),
    StructField("net_income_billion", DoubleType(), False),
])

df = spark.createDataFrame(data, schema=schema)
df.write.mode("overwrite").saveAsTable("main.default.meridian_financials")

# Add column comments (improves Genie understanding)
spark.sql("ALTER TABLE main.default.meridian_financials ALTER COLUMN net_revenue_billion COMMENT 'Net revenue in billions of JPY'")
spark.sql("ALTER TABLE main.default.meridian_financials ALTER COLUMN operating_profit_billion COMMENT 'Operating profit in billions of JPY'")
spark.sql("ALTER TABLE main.default.meridian_financials ALTER COLUMN net_income_billion COMMENT 'Net income in billions of JPY'")

print("Table main.default.meridian_financials created.")