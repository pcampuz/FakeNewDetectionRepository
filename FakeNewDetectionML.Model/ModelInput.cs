// This file was auto-generated by ML.NET Model Builder. 

using Microsoft.ML.Data;

namespace FakeNewDetectionML.Model
{
    public class ModelInput
    {
        [ColumnName("Label"), LoadColumn(0)]
        public string Label { get; set; }


        [ColumnName("Description"), LoadColumn(1)]
        public string Description { get; set; }


        [ColumnName("Source"), LoadColumn(2)]
        public string Source { get; set; }


    }
}
