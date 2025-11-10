using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Vision;

namespace SampahDetection
{
    public class ImageData
    {
        public string ImagePath { get; set; } = string.Empty;
        public string Label { get; set; } = string.Empty;
    }

    public class ImagePrediction : ImageData
    {
        public string PredictedLabel { get; set; } = string.Empty;
        public float[] Score { get; set; } = Array.Empty<float>();
    }

    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            string datasetPath = Path.Combine(Environment.CurrentDirectory, "dataset");
            string trainPath = Path.Combine(datasetPath, "train");
            string testPath = Path.Combine(datasetPath, "test");

            Console.WriteLine("📂 Lokasi dataset:");
            Console.WriteLine($"Train: {trainPath}");
            Console.WriteLine($"Test : {testPath}");

            var trainImages = LoadImagesFromDirectory(trainPath);
            var testImages = LoadImagesFromDirectory(testPath);

            var trainData = mlContext.Data.LoadFromEnumerable(trainImages);
            var testData = mlContext.Data.LoadFromEnumerable(testImages);

            var preprocessingPipeline = mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label")
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: datasetPath,
                    inputColumnName: nameof(ImageData.ImagePath)));

            var preprocessedTrainData = preprocessingPipeline.Fit(trainData).Transform(trainData);
            var preprocessedTestData = preprocessingPipeline.Fit(testData).Transform(testData);

            var trainingPipeline = mlContext.MulticlassClassification.Trainers
                .ImageClassification(new ImageClassificationTrainer.Options
                {
                    FeatureColumnName = "Image",
                    LabelColumnName = "LabelAsKey",
                    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                    Epoch = 10,
                    BatchSize = 10,
                    LearningRate = 0.01f,
                    MetricsCallback = (metrics) => Console.WriteLine(metrics),
                    ValidationSet = preprocessedTestData
                })
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("\n🚀 Training model dimulai...");
            var model = trainingPipeline.Fit(preprocessedTrainData);

            Console.WriteLine("\n🔍 Evaluasi model...");
            var predictions = model.Transform(preprocessedTestData);
            var metrics = mlContext.MulticlassClassification.Evaluate(
                predictions,
                labelColumnName: "LabelAsKey",
                predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"\n✅ Akurasi Micro: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"✅ Akurasi Macro: {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"✅ Log Loss: {metrics.LogLoss:F2}");

            string modelPath = Path.Combine(Environment.CurrentDirectory, "model.zip");
            mlContext.Model.Save(model, preprocessedTrainData.Schema, modelPath);
            Console.WriteLine($"\n💾 Model tersimpan sebagai: {modelPath}");
        }

        static IEnumerable<ImageData> LoadImagesFromDirectory(string folder)
        {
            var files = Directory.GetFiles(folder, "*", SearchOption.AllDirectories);
            foreach (var file in files)
            {
                yield return new ImageData
                {
                    ImagePath = file,
                    Label = Directory.GetParent(file)!.Name
                };
            }
        }
    }
}
