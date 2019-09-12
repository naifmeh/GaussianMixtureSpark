import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{DoubleType, FloatType, StructField, StructType}
import org.apache.spark.ml.Model
import org.apache.spark.sql.functions.lit

import scala.collection.mutable.ListBuffer
import scala.util.Random

trait GaussianMixtureParams extends Params
  with HasLabelCol with HasPredictionCol {

  val k = new IntParam(this, "k", "Number of clusters to train for.")

  def setK(value: Int): this.type = set(k -> value)
  setDefault(k -> 3)

  def getK: Int = $(k)

  def pdf(X: List[Double], mean: Double, variance: Double) = {
    val s1 = 1/(Math.sqrt(2*Math.PI*variance))
    val s2 = X.map(value => Math.exp(-1* (Math.pow(value - mean, 2)/(2 * variance))))

    s2.map(s1 * _)
  }
}

class GaussianMixtureEstimator(override val uid: String) extends Estimator[GaussianMixtureNMModel]
  with GaussianMixtureParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("gmm"))

  def setLabelCol(value: String): this.type = set(labelCol -> value)

  def setPredictionCol(value: String): this.type = set(predictionCol -> value)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): Estimator[GaussianMixtureNMModel] = defaultCopy(extra)


  override def fit(dataset: Dataset[_]): GaussianMixtureNMModel = {

    val multiplyList  = (a: List[Double], b: List[Double]) => {
      a.zip(b).map {
        case (i, j) => i.toDouble * j.toDouble
      }
    }

    val X = dataset
      .select($(labelCol))
      .collect.map(r => r.getDouble(0))
      .toList

    val weights = Array.fill($(k))(1.0)
    val means = Random.shuffle(X).take($(k)).toArray
    val variances = Seq.fill($(k))(Random.nextDouble).toArray

    (0 to 25).foreach(_ => {
      val likelihood = new ListBuffer[List[Double]]()
      val b = new ListBuffer[List[Double]]
      (0 until $(k)).foreach(j => {
        likelihood.append(pdf(X, means(j), Math.sqrt(variances(j))))
      })
      //likelihood.foreach(println)
      (0 until $(k)).foreach(j => {
        val updatedLocalLikelihood = likelihood(j).map(_* weights(j))
        val updatedGlobalLikelihood = (0 until $(k)).foldLeft(ListBuffer[List[Double]]())((sum, step) => {
          sum.append(likelihood(step).map(_ * weights(step)))
          sum
        })
          val finalGlobalLikelihood = updatedGlobalLikelihood(0).zipWithIndex.map{
            case (elem, indice) => {
            (1 until $(k)).foldLeft(elem)((sum, indice2) => sum + updatedGlobalLikelihood(indice2)(indice))
          }}


        b.append(updatedLocalLikelihood.zip(finalGlobalLikelihood).map {
          case (a, b) => a.toDouble / b.toDouble
        })

        val sumB = b(j).sum
        means(j) = multiplyList(b(j), X).sum / sumB
        variances(j) = multiplyList(b(j), X.map(x => Math.pow(x - means(j), 2))).sum / sumB
        weights(j) = b(j).sum / b(j).length
      })

    })

    GaussianMixtureNMModel(uid, weights, variances, means)
  }
}

object GaussianMixtureEstimator extends DefaultParamsReadable[GaussianMixtureEstimator] {
  override def load(path: String): GaussianMixtureEstimator = super.load(path)
}

case class GaussianMixtureNMModel(override val uid: String,
                                   weightsModel: Array[Double],
                                    variancesModel: Array[Double],
                                    meanModel: Array[Double]
                             )

  extends Model[GaussianMixtureNMModel] with DefaultParamsWritable
  with GaussianMixtureParams {

  override def copy(extra: ParamMap): GaussianMixtureNMModel = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    StructType(Seq(StructField($(predictionCol), DoubleType, true)).++(schema))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val X = dataset
      .select($(labelCol))
      .collect.map(r => r.getDouble(0))
      .toList

    val likelihood = new ListBuffer[List[Double]]
    val b = new ListBuffer[List[Double]]
    (0 until $(k)).foreach(j => {
      likelihood.append(pdf(X, meanModel(j), Math.sqrt(variancesModel(j))))
    })
    //likelihood.foreach(println)
    (0 until $(k)).foreach(j => {
      val updatedLocalLikelihood = likelihood(j).map(_* weightsModel(j))
      val updatedGlobalLikelihood = (0 until $(k)).foldLeft(ListBuffer[List[Double]]())((sum, step) => {
        sum.append(likelihood(step).map(_ * weightsModel(step)))
        sum
      })
      val finalGlobalLikelihood = updatedGlobalLikelihood(0).zipWithIndex.map{
        case (elem, indice) => {
          (1 until $(k)).foldLeft(elem)((sum, indice2) => sum + updatedGlobalLikelihood(indice2)(indice))
        }}

      b.append(updatedLocalLikelihood.zip(finalGlobalLikelihood).map {
        case (a, b) => a.toDouble / b.toDouble
      })
    })

    val predictions = new ListBuffer[Double]()
    X.zipWithIndex.foreach {
      case (_, i) => {
        predictions.append(
          (0 until $(k)).foldLeft(new ListBuffer[Double]())((list, index) => {
          list.append(b(index)(i))
          list
        })
          .zipWithIndex.maxBy(_._1)._2)
      }
    }

    dataset.withColumn($(predictionCol), lit(predictions)).toDF
  }
}

object GaussianMixtureNMModel  extends DefaultParamsReadable[GaussianMixtureNMModel] {
  override def load(path: String): GaussianMixtureNMModel = super.load(path)
}