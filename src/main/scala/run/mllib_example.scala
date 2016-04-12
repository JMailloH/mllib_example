package org.apache.spark.run

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.util._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.{ MultivariateStatisticalSummary, Statistics }
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.classification.{ SVMModel, SVMWithSGD }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.ChiSqSelector
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import scala.collection.mutable.ListBuffer

/**
 * @author Jesus Maillo
 */

object mllib_example extends Serializable {

  def main(arg: Array[String]) {

    //Spark Configuration
    val conf = new SparkConf().setAppName("mllib_example")
    val sc = new SparkContext(conf)

    //Read the dataset
    val rawdata = MLUtils.loadLibSVMFile(sc, "/home/hadoop/workspace/mllib_example/datasets/epsilon/epsilon_normalized_train")
    //    val rawdata = MLUtils.loadLibSVMFile(sc, "hdfs://hadoop-master:8020/user/spark/datasets/epsilon/epsilon_normalized")

    val data = rawdata.map { lp =>
      val newclass = if (lp.label == 1.0) 0 else 1
      new LabeledPoint(newclass, lp.features)
    }

    //Count the number of sample for each class
    val classInfo = data.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()

    val observations = data.map(_.features)
    // Compute column summary statistics. Param -> number of the column
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    var outputString = new ListBuffer[String]
    outputString += "\n\n@Mean (0) --> " + summary.mean(0) + "\n" // a dense vector containing the mean value for each column
    outputString += "@Variance (0) --> " + summary.variance(0) + "\n" // columnwise variance
    outputString += "@NumNonZeros (0) --> " + summary.numNonzeros(0) + "\n" // number of nonzeros in each column

    // Load the test data
    val rawtest = MLUtils.loadLibSVMFile(sc, "/home/hadoop/workspace/mllib_example/datasets/epsilon/epsilon_normalized_test")
    //val rawtest = MLUtils.loadLibSVMFile(sc, "hdfs://hadoop-master:8020/user/spark/datasets/epsilon/epsilon_normalized.t")
    val test = rawtest.map { lp =>
      val newclass = if (lp.label == 1.0) 0 else 1
      new LabeledPoint(newclass, lp.features)
    }

    // Cache data (only training)
    val training = data.cache()

    /** SUPPORT VECTOR MACHINE - SVM **/

    // Run training algorithm to build the model
    val numIterations = 10
    var modelSVM = SVMWithSGD.train(training, numIterations)
    // Clear the default threshold.
    modelSVM.clearThreshold()
    // Compute raw scores on the test set.
    val trainScores = training.map { point =>
      val score = modelSVM.predict(point.features)
      (score, point.label)
    }
    // Get evaluation metrics (do separately).
    val metrics = new BinaryClassificationMetrics(trainScores)
    val measuresByThreshold = metrics.fMeasureByThreshold.collect()
    val maxThreshold = measuresByThreshold.maxBy { _._2 }
    println("Max (Threshold, Precision):" + maxThreshold)
    modelSVM.setThreshold(maxThreshold._1)
    // Compute raw scores on the test set.
    var testScores = test.map(p => (modelSVM.predict(p.features), p.label))
    outputString += "@Acc SVM --> " + 1.0 * testScores.filter(x => x._1 == x._2).count() / test.count() + "\n" // number of nonzeros in each column

    /** RANDOM FOREST - RF **/

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 50 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32
    var modelRF = RandomForest.trainClassifier(training, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    modelRF.numTrees
    modelRF.totalNumNodes
    modelRF.trees
    // Evaluate model on test instances and compute test error
    testScores = test.map(p => (modelRF.predict(p.features), p.label))
    outputString += "@Acc RF --> " + 1.0 * testScores.filter(x => x._1 == x._2).count() / test.count() + "\n"

    println(outputString)

    val predictionsTxt = sc.parallelize(outputString, 1)
    predictionsTxt.saveAsTextFile("/home/hadoop/workspace/mllib_example/output.txt")
  }
}
