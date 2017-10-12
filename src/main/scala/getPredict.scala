package com.app

import org.apache.spark.sql.functions._
import java.sql.Timestamp
import java.text.SimpleDateFormat
import org.apache.spark.sql.types._
import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.MinMaxScaler

object getPredict {
  val fmt: SimpleDateFormat = new java.text.SimpleDateFormat("yyyy/MM/dd")


  def main(args: Array[String]):Unit = {
    /* args(0): startDay,"2017/07/ -/"
    *
     */
    //init the spark
    val conf = new SparkConf()
      .setMaster("yarn-cluster")
      .setAppName("feature data")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
//    val featurepath2 ="/datalab/user/frank.zhang/data/feature/action/201706MA"
//
//    val feat2 =sqlContext.read.parquet(featurepath)
//    val labelpath ="/datalab/user/frank.zhang/data/label/label0701"
    // // val labelData = sqlContext.read.parquet(labelpath)
//    val appSavePath ="/datalab/user/frank.zhang/data/feature/app/201706"
    val actionPath = args(0)
    val appSavePath = args(1)
    val labelPath = args(2)
    val numTree = args(3).toInt
    val treeDepth = args(4).toInt


    val training = getTrainData(sqlContext,actionPath,appSavePath,labelPath)
    RFModel(sc,training,numTree,treeDepth)

  }
  def deleteFile(path: String) : Unit = {
    import org.apache.hadoop.conf.Configuration
    import org.apache.hadoop.fs.{FileSystem, Path}
    val hdfs : FileSystem = FileSystem.get(new Configuration)
    val isExist = hdfs.exists(new Path(path))
    if (isExist){
      println("welll")
      hdfs.delete(new Path(path), true)//true: delete files recursively
    }
  }
  /**seve the DataFrame to the specified path
    *detect the path is exists when exists delete it. then save the data
    */
  def saveData(df:org.apache.spark.sql.DataFrame,path:String):Unit ={
    import org.apache.hadoop.conf.Configuration
    import org.apache.hadoop.fs.{FileSystem, Path}
    val hdfs : FileSystem = FileSystem.get(new Configuration)
    val isExist = hdfs.exists(new Path(path))
    if (isExist){
      hdfs.delete(new Path(path), true)//true: delete files recursively
    }

    df.write.format("parquet").save(path)
  }
  // load DataFrame format data from paequet files
  def loadData(sqlContext:org.apache.spark.sql.SQLContext,readPath: String):org.apache.spark.sql.DataFrame ={
    val df = sqlContext.read.parquet(readPath)
    df
  }
  def getTrainData(sqlContext: org.apache.spark.sql.SQLContext,featurePath:String,apppath : String,labelPath:String,ratio :Double = 0.1):org.apache.spark.sql.DataFrame = {
    import sqlContext.implicits._
    // load features form save data
    // val featurePath ="/datalab/user/frank.zhang/data/feature/month4"
    val featData = sqlContext.read.parquet(featurePath).drop("fMin").drop("tMin")
    val appInfo = sqlContext.read.parquet(apppath)
    val newFeat =featData.join(appInfo,Seq("tdid"),"left").na.drop
    // //load features from calculate,slow
    // val rawpath="/datalab/user/jiesheng.ye/data/zhixian.zheng.new/2017/04/*/"
    // val featData = extractFeaturesAll(rawpath)

    // val labelPath ="/datalab/user/frank.zhang/data/label/label0501"
    val labelData = sqlContext.read.parquet(labelPath)


    val preparedData = newFeat.join(labelData,Seq("tdid"),"left").na.fill(1.0,Array("label"))
    println(preparedData.printSchema)
    //  Array(tdid, platform, brand, osStandardVersion, tMean, tMax, tMin, tVar, fMean, fMax, fMin, fVar, fSum, fPeriodActive, fDayActive, tNearUseDiff, tAliveTime, tStay, fException)
    val trainData = preparedData.select("osStandardVersion","tMean","tMax","tVar","fMean","fMax","fVar","fSum","fPeriodActive","fDayActive","tNearUseDiff","tAliveTime","tStay","fException","label").map{row =>

      val label =row.getDouble(row.length-1);
      val features = Array(row.getDouble(0),row.getDouble(1),row.getDouble(2),row.getDouble(3),row.getDouble(4),row.getDouble(5),row.getDouble(6),row.getDouble(7),row.getDouble(8),row.getDouble(9),row.getDouble(10),row.getDouble(11),row.getDouble(12),row.getDouble(13));
      (Vectors.dense(features),label)}.toDF("features","label")
    val positiveSample = trainData.filter("label = 1.0")
    val negativeSample = trainData.filter("label = 0.0").sample(false,ratio) // 0.0-1.0
    // val negativeSample = trainData.filter("label = 0.0")
    val trainDataSample = positiveSample.unionAll(negativeSample)
    trainDataSample


  }

  def setLabel(sqlContext :org.apache.spark.sql.SQLContext,datainput:String,startDayStr:String,endDayStr:String,labelpath:String) : org.apache.spark.sql.DataFrame = {
    // val startDayStr ="2017-05-01"
    // val endDayStr ="2017-05-07"
    // val labelpath="/datalab/user/frank.zhang/data/label"
    // val datainput="/datalab/user/jiesheng.ye/data/zhixian.zheng.new/2017/05/*/"
    import sqlContext.implicits._

    val monthDF = sqlContext.read.parquet(datainput)
    val  fmt:SimpleDateFormat = new java.text.SimpleDateFormat("yyyy-MM-dd")
    val startDay =new java.sql.Date(fmt.parse(startDayStr).getTime())
    val endDay =new java.sql.Date(fmt.parse(endDayStr).getTime())

    val dataDF = monthDF.filter($"receiveTime".between(startDay,endDay)).select("tdid").distinct
    //effective id  label =0
    val resDF = dataDF.rdd.map(row =>(row.getString(0),0.0)).toDF("tdid","label")
    // resDF.write.format("parquet").save(labelpath)

    resDF
  }

  def modelReport(sc:SparkContext,predictionAndLabels:org.apache.spark.rdd.RDD[(Double,Double)]) = {
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
    // AUC
    val metricsBC = new BinaryClassificationMetrics(predictionAndLabels)
    val auROC = metricsBC.areaUnderROC
    println("Area under ROC = " + auROC)

    val metricsMC = new MulticlassMetrics(predictionAndLabels)
    // Confusion matrix
    println("Confusion matrix:")
    println(metricsMC.confusionMatrix)

    // Overall Statistics
    val precision = metricsMC.precision(1.0)
    val recall = metricsMC.recall(1.0) // same as true positive rate
    val f1Score = metricsMC.fMeasure(1.0)
    println("Summary Statistics")
    println(s"Precision = $precision")
    println(s"Recall = $recall")
    println(s"F1 Score = $f1Score")
    sc.parallelize(Array(auROC,precision,recall)).repartition(1).saveAsTextFile("/datalab/user/frank.zhang/data/result/"+System.currentTimeMillis().toString)

  }
  def RFModel(sc:SparkContext,training :org.apache.spark.sql.DataFrame,numTree :Integer,treeDepth : Integer) = {


    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(training)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(training)
//    val scaler = new MinMaxScaler()
//      .setInputCol("indexedFeatures")
//      .setOutputCol("scaledFeatures")
//      .fit(training)

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = training.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(numTree)
      .setMaxDepth(treeDepth)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer,rf, labelConverter))

    // Train model.  This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)
//    val ctime = System.currentTimeMillis().toString
//    predictions.write.saveAsTable(s"/datalab/user/frank.zhang/data/prediction/$ctime")

    // Select example rows to display.
    // predictions.select("predictedLabel", "label", "features").show(5)
    val predictionAndLabels= predictions.select("predictedLabel", "label").map(row=>(row.getString(0).toDouble,row.getDouble(1)))
    modelReport(sc,predictionAndLabels)

    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    rfModel
    // println("Learned classification forest model:\n" + rfModel.toDebugString)
  }
}
