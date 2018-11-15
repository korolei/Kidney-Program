using System;
using Microsoft.ML.Runtime.Api;
using System.Threading.Tasks;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;

namespace Kidney
{
  class KidneyProgram
  {

    public class KidneyData
    {
      [Microsoft.ML.Runtime.Api.Column(ordinal: "0", name: "Age")]
      public float Age;

      [Microsoft.ML.Runtime.Api.Column(ordinal: "1", name: "Sex")]
      public float Sex;

      [Microsoft.ML.Runtime.Api.Column(ordinal: "2", name: "Kidney")]
      public float Kidney;

      [Microsoft.ML.Runtime.Api.Column(ordinal: "3", name: "Label")]
      public string Label;
    }

    public class KidneyPrediction
    {
      [ColumnName("PredictedLabel")]
      public string PredictedLabels;
    }

    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin ML.NET (v0.3.0 preview) demo run");
      Console.WriteLine("Predict die/survive based on patient age, sex, kidney test");
      var pipeline = new LearningPipeline();

      string dataPath = "KidneyData.txt";
      pipeline.Add(new TextLoader(dataPath).CreateFrom<KidneyData>(separator: ','));
      pipeline.Add(new Dictionarizer("Label"));
      pipeline.Add(new ColumnConcatenator("Features", "Age", "Sex", "Kidney"));
      pipeline.Add(new LogisticRegressionBinaryClassifier());
      pipeline.Add(new PredictedLabelColumnOriginalValueConverter {PredictedLabelColumn = "PredictedLabel"});
      Console.WriteLine("\nStarting training \n");
      var model = pipeline.Train<KidneyData, KidneyPrediction>();
      Console.WriteLine("\nTraining complete \n");

      const string modelPath = "KidneyModel.zip";
      Task.Run(async () =>
      {
        await model.WriteAsync(modelPath);
      }).GetAwaiter().GetResult();

      var testData = new TextLoader(dataPath).CreateFrom<KidneyData>(separator: ',');
      var evaluator = new BinaryClassificationEvaluator();
      var metrics = evaluator.Evaluate(model, testData);
      double acc = metrics.Accuracy * 100;
      Console.WriteLine("Model accuracy = " + acc.ToString("F2") + "%");

      //PredictionModel<KidneyData, KidneyPrediction> model2 = null;
      //Task.Run(async () =>
      //{
      //  model2 = await PredictionModel.ReadAsync<KidneyData, KidneyPrediction>(ModelPath);
      //}).GetAwaiter().GetResult();
      
      Console.WriteLine("Making prediction for 52-year old male with kidney = 9.80: ");
      KidneyData newPatient = new KidneyData() { Age = 52f, Sex = -1f, Kidney = 9.80f };
      KidneyPrediction prediction = model.Predict(newPatient);
      string result = prediction.PredictedLabels;
      Console.WriteLine("Prediction = " + result);

      Console.WriteLine("\nEnd ML.NET demo");
      Console.ReadLine();
    } // Main
  } // Program
} // ns
