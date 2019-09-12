import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object main {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("GaussianMixture").master("local")
      .getOrCreate()

    import spark.implicits._
    val values = List[Double](1.0,1.0,1.0,2.0,2,3,3).toDF("label")

    val estimator = new GaussianMixtureEstimator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setK(3)

    estimator.fit(values).transform(values)
  }
}
