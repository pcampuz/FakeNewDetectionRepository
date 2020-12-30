using FakeNewDetectionML.Model;
using Microsoft.ML;
using System;

namespace FakeNewDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                const string descriptionToValidate = "Cristiano Ronaldo é o rei da terra batida?";
                const string sourceToValidate = "Rumor";

                Console.OutputEncoding = System.Text.Encoding.UTF8;

                Console.WriteLine(descriptionToValidate);
                Console.WriteLine("Prima qualquer tecla para avaliar a veracidade na notícia");
                Console.ReadKey();

                // Load the model  
                MLContext mlContext = new MLContext();
                ITransformer mlModel = mlContext.Model.Load(@"C:\Users\nb22895\source\repos\FakeNewDetectionRepository\FakeNewDetectionML.Model\MLModel.zip", out var modelInputSchema);

                // Create predection engine related to the loaded train model
                var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

                ModelInput input = new ModelInput
                {
                    Description = descriptionToValidate,
                    Source = sourceToValidate
                };

                // Try model on sample data and find the score
                ModelOutput result = predEngine.Predict(input);

                Console.WriteLine(string.Format("\nPrevisão que a afirmação em causa seja: {0}", Convert.ToDecimal(result.Prediction) == 0 ? "Falsa!" : "Verdadeira!", result.Score[0]));
                Console.WriteLine(string.Format("Percentagem de acerto sobre a previsão da afirmação: {0} %", Math.Round(Convert.ToDecimal(result.Score[1]) * 100,2)));
                Console.ReadKey();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
                Console.ReadKey();
            }
        }
    }
}
