package br.com.profch;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.Model;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public final class EvaluationUtils {
    private static EvaluationUtils instance;

    private EvaluationUtils() {
    }

    /**
     * Utilizado para retornar a referência para a instância da classe.
     * @return A instância Singleton da classe.
     */
    public static EvaluationUtils getInstance() {
        if(instance == null) {
            instance = new EvaluationUtils();
        }

        return instance;
    }

    /**
     * Realiza uma validação do modelo gerado no treinamento com um novo conjunto de dados. 
     * Após a validação, um arquivo texto com os resultados das predições dos testes e desta
     * validação é gerado. Para cada conjunto de dados, estão disponíveis no arquivo os 
     * seguintes indicadores:
     * <br /><br />
     * <ul>
     *  <li>Matriz de confusão</li>
     *  <li>F1-Score</li>
     *  <li>Acurácia</li>
     *  <li>Precisão</li>
     *  <li>Recall</li>
     *  <li>Taxa de verdadeiros positivos</li>
     *  <li>Taxa de falsos positivos</li>
     * </ul>
     * @param validationDataset O conjunto de dados que será aplicado no modelo para validação.
     * @param model O modelo gerado no treinamento.
     * @param testPredictions As predições geradas pelo modelo com o conjunto de dados de teste.
     * @param fileName O caminho completo do arquivo em que os resultados serão salvos.
     * @param sparkSession Referência para a sessão do Spark em que está sendo feito o treinamento.
     */
    @SuppressWarnings("rawtypes")
    public void saveEvaluationsToFile(Dataset<Row> validationDataset, Model model,
                        Dataset<Row> testPredictions, String fileName, SparkSession sparkSession) {
        
        List<Row> evaluations = buildFileHeader(model.getClass().getSimpleName());
        evaluations.addAll(buildEvaluations(validationDataset, model, testPredictions));

        List<StructField> listOfStructField = new ArrayList<StructField>();
        listOfStructField.add(DataTypes.createStructField("Text", DataTypes.StringType, true));
        StructType struct =  DataTypes.createStructType(listOfStructField);

        sparkSession.createDataFrame(evaluations, struct).repartition(1).write().text(fileName);
    }

    // #region Métodos privados
    private List<Row> buildFileHeader(String modelName) {
        List<Row> header = new ArrayList<Row>();
        StringBuilder out = new StringBuilder();

        out.append("\n******************************************************************************************\n");
        out.append("**");
        out.append("TRAINING RESULTS FOR MODEL ");
        out.append(modelName.toUpperCase());
        out.append("**");
        out.append("\n******************************************************************************************\n");

        header.add(RowFactory.create(out.toString()));

        return header;
    }
    
    @SuppressWarnings({"rawtypes"})
    private List<Row> buildEvaluations(Dataset<Row> dataset, Model model,
            Dataset<Row> prediction) {
        Dataset<Row> result;
        List<Row> evaluations = new ArrayList<Row>();
        evaluations.add(RowFactory.create(evaluatePredictions(prediction, "Test set")));

        result = model.transform(dataset);

        prediction = result.select("prediction", "label");
        evaluations.add(RowFactory.create(evaluatePredictions(prediction, "Validation set")));

        return evaluations;
    }

    private String evaluatePredictions(Dataset<Row> dataSet, String setName) {
        StringBuffer out = new StringBuffer();

        out.append("\n*********************************\n");
        out.append("**");
        out.append("DATASET NAME: ");
        out.append(setName.toUpperCase());
        out.append("**");
        out.append("\n*********************************\n");

        MulticlassMetrics metrics = new MulticlassMetrics(dataSet);
        out.append(formatOutput(setName, "Confusion matrix", metrics.confusionMatrix()));
        out.append(formatOutput(setName, "F1-Score", metrics.weightedFMeasure()));
        out.append(formatOutput(setName, "Accuracy", metrics.accuracy()));
        out.append(formatOutput(setName, "Weighted precision", metrics.weightedPrecision()));
        out.append(formatOutput(setName, "Weighted recall", metrics.weightedRecall()));
        out.append(formatOutput(setName, "Weighted true positive rate", metrics.weightedTruePositiveRate()));
        out.append(formatOutput(setName, "Weighted false positive rate", metrics.weightedFalsePositiveRate()));

        return out.toString();
    }

    private StringBuffer formatOutput(String setName, String metricName, Object metricValue) {
        StringBuffer out = new StringBuffer();

        out.append("\n*************\n");
        out.append(setName);
        out.append(" ");
        out.append(metricName);
        out.append("\n");
        out.append(metricValue);
        out.append("\n*************\n");

        return out;

    }
    //#endregion
}
