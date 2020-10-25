package br.com.profch;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.util.MLWritable;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

import scala.collection.JavaConverters;

public final class App {
  // #region Constants
  private static final String HDFS_COVID_DIRECTORY = "hdfs://localhost:9090/covid/";

  private static final String TODAY = new SimpleDateFormat("yyyy-MM-dd-HH-mm").format(new Date());
  private static final String DATA_FILE_NAME = HDFS_COVID_DIRECTORY + "covid_19_casos.csv";
  private static final String RFC_MODEL_FILE_NAME = HDFS_COVID_DIRECTORY + "training/"
                + TODAY + "/covid-argentina-rfc-best-model.mlmdl";
  private static final String RFC_EVALUATIONS_FILE_NAME = HDFS_COVID_DIRECTORY + "training/"
                + TODAY + "/covid-argentina-rfc-best-model-evaluations.txt";
  private static final String LR_MODEL_FILE_NAME = HDFS_COVID_DIRECTORY + "training/" + TODAY
                + "/covid-argentina-lr-best-model.mlmdl";
  private static final String LR_EVALUATIONS_FILE_NAME = HDFS_COVID_DIRECTORY + "training/"
                + TODAY + "/covid-argentina-lr-best-model-evaluations.txt";
  private static final String GBT_MODEL_FILE_NAME = HDFS_COVID_DIRECTORY + "training/"
                + TODAY + "/covid-argentina-gbt-best-model.mlmdl";
  private static final String GBT_EVALUATIONS_FILE_NAME = HDFS_COVID_DIRECTORY + "training/"
                + TODAY + "/covid-argentina-gbt-best-model-evaluations.txt";
  private static final String SVM_MODEL_FILE_NAME = HDFS_COVID_DIRECTORY + "training/"
                + TODAY + "/covid-argentina-svm-best-model.mlmdl";
  private static final String SVM_EVALUATIONS_FILE_NAME = HDFS_COVID_DIRECTORY + "training/"
                + TODAY + "/covid-argentina-svm-best-model-evaluations.txt";
  // #endregion

  private App(){}

  public static void main(String[] args) {
    SparkSession spark = SparkSession.builder().appName("COVID-19 Argentina").getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");

    //trainModels(spark, true);
    //printParameters("Rodada com dias_ate_diagnostico");

    trainModels(spark, false);
    printParameters("Rodada sem dias_ate_diagnostico");

    spark.stop();
  }  

  // #region Métodos privados
  private static void printParameters(String title) {
    System.out.println("*********************************");
    System.out.println(title);
    System.out.println("*********************************");

    RandomForestClassificationModel rfcModel = RandomForestClassificationModel.load(RFC_MODEL_FILE_NAME);
    System.out.println("RANDOM FOREST CLASSIFICATION MODEL:");
    System.out.println("Imputity: " + rfcModel.getImpurity());
    System.out.println("Seed: " + rfcModel.getSeed());
    System.out.println("Number of trees: " + rfcModel.getNumTrees());

    LogisticRegressionModel lrModel = LogisticRegressionModel.load(LR_MODEL_FILE_NAME);
    System.out.println("LOGISTIC REGRESSION CLASSIFICATION MODEL:");
    System.out.println("Elastic Net Param: " + lrModel.getElasticNetParam());
    System.out.println("Max Iter: " + lrModel.getMaxIter());
    System.out.println("Regularization Parameter: " + lrModel.getRegParam());
    System.out.println("Tolerance: " + lrModel.getTol());

    GBTClassificationModel gbtModel = GBTClassificationModel.load(GBT_MODEL_FILE_NAME);
    System.out.println("GRADIENT-BOOSTED TREE CLASSIFICATION MODEL:");
    System.out.println("Seed: " + gbtModel.getSeed());
    System.out.println("Max Iter: " + gbtModel.getMaxIter());
    System.out.println("Step Size: " + gbtModel.getStepSize());

    LinearSVCModel svmModel = LinearSVCModel.load(SVM_MODEL_FILE_NAME);
    System.out.println("LINEAR SUPPORT VECTOR MACHINE CLASSIFICATION MODEL:");
    System.out.println("Max Iter: " + svmModel.getMaxIter());
    System.out.println("Regularization Parameter: " + svmModel.getRegParam());
    System.out.println("Tolerance: " + svmModel.getTol());
  }

  private static void trainModels(SparkSession spark, boolean comDiagnostico) {
    Dataset<Row> ds = processDataset(loadDataset(spark));

    String[] inputCols;
    if (comDiagnostico) {
            inputCols = new String[] { "homem", "idade", "uti", "respirador", "dias_ate_internacao",
                            "dias_ate_diagnostico" };
            ds = ds.where("dias_ate_internacao is not null");
    } else {
            inputCols = new String[] { "homem", "idade", "uti", "respirador", "dias_ate_internacao" };
    }

    VectorAssembler va = new VectorAssembler();
    ds = va.setInputCols(inputCols).setOutputCol("features").setHandleInvalid("skip")
            .transform(ds).select("features", "falecido").withColumnRenamed("falecido", "label");

    Dataset<Row>[] splits = ds.randomSplit(new double[] { 0.7, 0.2, 0.1 }, 2048);
    Dataset<Row> train = splits[0];
    Dataset<Row> test = splits[1];
    Dataset<Row> validation = splits[2];

    runRandomForestClassifier(spark, train, test, validation);
    runLogisticRegressionClassifier(spark, train, test, validation);
    runSupportVectorMachinesClassifier(spark, train, test, validation);
    runGradientBoostedTreeClassifier(spark, train, test, validation);
  }

  private static void runRandomForestClassifier(SparkSession spark, Dataset<Row> train, Dataset<Row> test,
            Dataset<Row> validation) {
    RandomForestClassifier rfc = new RandomForestClassifier().setFeaturesCol("features")
                    .setLabelCol("label");
    MulticlassClassificationEvaluator eval = new MulticlassClassificationEvaluator().setLabelCol("label")
                    .setPredictionCol("prediction").setMetricName("f1");

    ArrayList<String> inpurities = new ArrayList<String>();
    inpurities.add("gini");
    inpurities.add("entropy");

    ParamMap[] paramGrid = new ParamGridBuilder()
                    .addGrid(rfc.impurity(),
                                    JavaConverters.iterableAsScalaIterableConverter(inpurities).asScala())
                    .addGrid(rfc.seed(), new long[] { 512, 1024, 2048 })
                    .addGrid(rfc.numTrees(), new int[] { 100, 200, 400, 600, 800, 1000 }).build();

    CrossValidator cv = new CrossValidator().setEstimator(rfc).setEstimatorParamMaps(paramGrid)
                    .setNumFolds(3).setParallelism(2).setEvaluator(eval);

    RandomForestClassificationModel model = (RandomForestClassificationModel) cv.fit(train).bestModel();
    Dataset<Row> predict = model.transform(test).select("prediction", "label");

    EvaluationUtils evUt = EvaluationUtils.getInstance();
    evUt.saveEvaluationsToFile(validation, model, predict, RFC_EVALUATIONS_FILE_NAME, spark);
    try {
            model.save(RFC_MODEL_FILE_NAME);
    } catch (IOException e) {
            e.printStackTrace();
    }
  }

  private static void runLogisticRegressionClassifier(SparkSession spark, Dataset<Row> train, Dataset<Row> test,
            Dataset<Row> validation) {
    LogisticRegression lr = new LogisticRegression().setFeaturesCol("features").setLabelCol("label");
    MulticlassClassificationEvaluator eval = new MulticlassClassificationEvaluator().setLabelCol("label")
                    .setPredictionCol("prediction").setMetricName("f1");

    ParamMap[] paramGrid = new ParamGridBuilder().addGrid(lr.maxIter(), new int[] { 100, 300, 500, 700 })
                    .addGrid(lr.elasticNetParam(), new double[] { 0.8, 0.7, 0.5, 0.4, 0.3, 0.15 })
                    .addGrid(lr.regParam(), new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.7 })
                    .addGrid(lr.tol(), new double[] { 0, 0.1, 0.2, 0.3 }).build();

    CrossValidator cv = new CrossValidator().setEstimator(lr).setEstimatorParamMaps(paramGrid)
                    .setNumFolds(3).setParallelism(2).setEvaluator(eval);

    @SuppressWarnings("rawtypes")
    Model model = cv.fit(train).bestModel();
    Dataset<Row> predict = model.transform(test).select("prediction", "label");

    EvaluationUtils evUt = EvaluationUtils.getInstance();
    evUt.saveEvaluationsToFile(validation, model, predict, LR_EVALUATIONS_FILE_NAME, spark);
    try {
            ((MLWritable) model).save(LR_MODEL_FILE_NAME);
    } catch (IOException e) {
            e.printStackTrace();
    }
  }

  private static void runGradientBoostedTreeClassifier(SparkSession spark, Dataset<Row> train, Dataset<Row> test,
            Dataset<Row> validation) {
    GBTClassifier gbt = new GBTClassifier().setFeaturesCol("features").setLabelCol("label");
    MulticlassClassificationEvaluator eval = new MulticlassClassificationEvaluator().setLabelCol("label")
                    .setPredictionCol("prediction").setMetricName("f1");

    ParamMap[] paramGrid = new ParamGridBuilder().addGrid(gbt.maxIter(), new int[] { 100, 300, 500 })
                    .addGrid(gbt.seed(), new long[] { 1024, 2048 })
                    .addGrid(gbt.stepSize(), new double[] { 0.2, 0.3 }).build();

    CrossValidator cv = new CrossValidator().setEstimator(gbt).setEstimatorParamMaps(paramGrid)
                    .setNumFolds(3).setParallelism(2).setEvaluator(eval);

    GBTClassificationModel model = (GBTClassificationModel) cv.fit(train).bestModel();
    Dataset<Row> predict = model.transform(test).select("prediction", "label");

    EvaluationUtils evUt = EvaluationUtils.getInstance();
    evUt.saveEvaluationsToFile(validation, model, predict, GBT_EVALUATIONS_FILE_NAME, spark);
    try {
            model.save(GBT_MODEL_FILE_NAME);
    } catch (IOException e) {
            e.printStackTrace();
    }
  }

  private static void runSupportVectorMachinesClassifier(SparkSession spark, Dataset<Row> train,
            Dataset<Row> test, Dataset<Row> validation) {
    LinearSVC svm = new LinearSVC().setFeaturesCol("features").setLabelCol("label");
    MulticlassClassificationEvaluator eval = new MulticlassClassificationEvaluator().setLabelCol("label")
                    .setPredictionCol("prediction").setMetricName("f1");

    ParamMap[] paramGrid = new ParamGridBuilder()
                    .addGrid(svm.maxIter(), new int[] { 100, 300, 500, 700, 900 })
                    .addGrid(svm.regParam(), new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 })
                    .addGrid(svm.tol(), new double[] { 0, 0.1, 0.2, 0.3 }).build();

    CrossValidator cv = new CrossValidator().setEstimator(svm).setEstimatorParamMaps(paramGrid)
                    .setNumFolds(3).setParallelism(2).setEvaluator(eval);

    LinearSVCModel model = (LinearSVCModel) cv.fit(train).bestModel();
    Dataset<Row> predict = model.transform(test).select("prediction", "label");

    EvaluationUtils evUt = EvaluationUtils.getInstance();
    evUt.saveEvaluationsToFile(validation, model, predict, SVM_EVALUATIONS_FILE_NAME, spark);
    try {
            model.save(SVM_MODEL_FILE_NAME);
    } catch (IOException e) {
            e.printStackTrace();
    }
  }

  private static Dataset<Row> processDataset(Dataset<Row> ds) {
    long media_idade_homem = ds.where("sexo = 'M'").select(functions.ceil(functions.avg("idade"))).first()
                    .getLong(0);
    long media_idade_mulher = ds.where("sexo <> 'M'").select(functions.ceil(functions.avg("idade"))).first()
                    .getLong(0);
    ds = ds.withColumn("uti", functions.when(ds.col("uti").equalTo("SI"), 1).otherwise(0))
                    .withColumn("respirador",
                                    functions.when(ds.col("respirador").equalTo("SI"), 1).otherwise(0))
                    .withColumn("falecido",
                                    functions.when(ds.col("falecido").contains("Fallecido"), 1.0)
                                                    .otherwise(0.0))
                    .withColumn("dias_ate_internacao",
                                    functions.datediff(ds.col("data_internacao"), ds.col("data_sintomas")))
                    .withColumn("dias_ate_diagnostico",
                                    functions.datediff(ds.col("data_diagnostico"), ds.col("data_sintomas")))
                    .withColumn("idade",
                                    functions.when(ds.col("sexo").equalTo("M")
                                                    .and(ds.col("idade").isNull()), media_idade_homem)
                                                    .otherwise(media_idade_mulher));

    long media_dias_ate_diagnostico = ds.select(functions.ceil(functions.avg("dias_ate_diagnostico")))
                    .first().getLong(0);

    ds = ds.withColumn("dias_ate_diagnostico",
                    functions.when(ds.col("dias_ate_diagnostico").isNotNull(),
                                    ds.col("dias_ate_diagnostico")).otherwise(media_dias_ate_diagnostico))
                    .withColumn("homem", functions.when(ds.col("sexo").equalTo("M"), 1).otherwise(0))
                    .withColumn("idade", ds.col("idade").cast(DataTypes.DoubleType));

    ds = ds.select("homem", "idade", "uti", "respirador", "dias_ate_internacao", "dias_ate_diagnostico",
                    "falecido");
    return ds;
  }

  private static Dataset<Row> loadDataset(SparkSession spark) {
    Dataset<Row> ds = spark.read().option("header", "true").option("delimiter", ",")
                  .option("inferSchema", "true").csv(DATA_FILE_NAME);

    ds = ds.select("sexo", "edad", "fecha_inicio_sintomas", "fecha_internacion", "cuidado_intensivo",
                  "asistencia_respiratoria_mecanica", "clasificacion", "fecha_diagnostico")
                  .where("clasificacion_resumen = \"Confirmado\"").toDF("sexo", "idade", "data_sintomas",
                                  "data_internacao", "uti", "respirador", "falecido", "data_diagnostico");
    return ds;
  }
  // #endregion

}
