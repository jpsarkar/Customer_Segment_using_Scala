import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors


val data = sc.textFile("telecomdata.csv")
val parsedData = data.map(s => s.split(',')).map(f => (f(3).toDouble,f(4).toDouble)).filter(p => !((p._1 == 0) && (p._2 == 0))).persist()

val RowsDF = parsedData.toDF("latitude", "longitude")

val assembler = new VectorAssembler().setInputCols(Array("latitude", "longitude")).setOutputCol("features")
val telecomDF = assembler.transform(RowsDF).select("features")

val kmeans = new KMeans().setK(3).setSeed(1L).setPredictionCol("prediction")

val model = kmeans.fit(telecomDF)


val WSSSE = model.computeCost(telecomDF)
println(s"Within Set Sum of Squared Errors = $WSSSE")

println("Cluster Centers: ")
model.clusterCenters.foreach(println)

val predictionResult = model.transform(telecomDF)
predictionResult.rdd.coalesce(1, true).saveAsTextFile("output")

System.exit(0)