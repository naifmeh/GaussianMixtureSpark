import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

object main {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("GaussianMixture").master("local")
      .getOrCreate()

    import spark.implicits._
    val values = Seq.fill(200)(Random.nextDouble).toList.toDF("label")

    val estimator = new GaussianMixtureEstimator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setK(3)

    estimator.fit(values).transform(values).show(200)
  }
}
