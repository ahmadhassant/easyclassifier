using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace EasyResearch.Core;

public static class Wire
{
    public static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true
    };
    public const int Version = 1;
}

public sealed record WorkerRequest
{
    public int ProtocolVersion { get; init; } = Wire.Version;
    public string RequestId { get; init; } = Guid.NewGuid().ToString("N");
    public required string Action { get; init; }
    public string? ModuleId { get; init; }
    public string? Dataset { get; init; }
    public string? Target { get; init; }
    public string? OutputRoot { get; init; }
    public Dictionary<string, object>? Settings { get; init; }
}

public sealed record ModuleDescriptor
{
    public required string Id { get; init; }
    public required string Title { get; init; }
    public string Description { get; init; } = "";
    public string InputKind { get; init; } = "";
    public string Package { get; init; } = "";
    public string PackageVersion { get; init; } = "";
    public string? EngineVersion { get; init; }
    public List<ModelDescriptor> Models { get; init; } = [];
    public List<MetricDescriptor> Metrics { get; init; } = [];
    public List<string> Headline { get; init; } = [];
    public List<OptionDescriptor> ValidationOptions { get; init; } = [];
    public List<OptionDescriptor> SelectionOptions { get; init; } = [];
    public ModuleLabels Labels { get; init; } = new();
    /// <summary>Why the module could not be loaded (empty when it works).</summary>
    public string UnavailableReason { get; init; } = "";
    /// <summary>File types offered by the Browse dialog for this module.</summary>
    public string FileFilter { get; init; } = "CSV and Excel files|*.csv;*.xlsx;*.xlsm|All files|*.*";
    /// <summary>Drop-down choices in the Advanced options section (closed by default).</summary>
    public List<ParameterDescriptor> AdvancedChoices { get; init; } = [];
    /// <summary>Tick lists in the Advanced options section, such as the figures or the measures.</summary>
    public List<ListDescriptor> AdvancedLists { get; init; } = [];
}
public sealed record ModelDescriptor
{
    public required string Id { get; init; }
    public required string Name { get; init; }
    public bool Available { get; init; }
    public bool DefaultSelected { get; init; }
    public string Reason { get; init; } = "";
}
public sealed record MetricDescriptor
{
    public required string Key { get; init; }
    public required string Label { get; init; }
    /// <summary>percent, decimal3, decimal4 or number (precision chosen from the size of the value).</summary>
    public string Format { get; init; } = "decimal4";
    public bool HigherIsBetter { get; init; } = true;

    public string Display(double? value)
    {
        if (value is not { } v) return "—";
        var c = CultureInfo.CurrentCulture;
        return Format switch
        {
            "percent" => v.ToString("P2", c),
            "decimal3" => v.ToString("F3", c),
            "number" => Math.Abs(v) >= 1000 ? v.ToString("N0", c) : Math.Abs(v) >= 1 ? v.ToString("N2", c) : v.ToString("G3", c),
            _ => v.ToString("F4", c)
        };
    }
}
public sealed record OptionDescriptor(string Id, string Name);
/// <summary>A data-dependent choice offered after inspection (for example the date column or the forecast horizon).</summary>
public sealed record ParameterDescriptor
{
    public string Id { get; init; } = "";
    public string Label { get; init; } = "";
    public string Hint { get; init; } = "";
    public List<OptionDescriptor> Options { get; init; } = [];
    public string? Default { get; init; }
}
/// <summary>A list of options the user ticks (for example the figures to draw); Default holds the ids ticked at first.</summary>
public sealed record ListDescriptor
{
    public string Id { get; init; } = "";
    public string Label { get; init; } = "";
    public string Hint { get; init; } = "";
    public List<OptionDescriptor> Options { get; init; } = [];
    public List<string> Default { get; init; } = [];
}
/// <summary>A small table of results shown as is (for example the forecast values).</summary>
public sealed record ResultTable
{
    public string Title { get; init; } = "";
    public string Hint { get; init; } = "";
    public List<string> Columns { get; init; } = [];
    public List<List<string>> Rows { get; init; } = [];
}
public sealed record ModuleLabels
{
    public string TargetTitle { get; init; } = "What do you want to predict?";
    public string TargetHint { get; init; } = "";
    public string ModelsTitle { get; init; } = "Compare models";
    public string ModelNoun { get; init; } = "model";
    public string ModelNounPlural { get; init; } = "models";
    public string DemoButton { get; init; } = "Try example";
    public string SelectionMetric { get; init; } = "";
    public string PrepNote { get; init; } = "";
}
public sealed record CatalogResult(List<ModuleDescriptor> Modules);
public sealed record ColumnDescriptor
{
    public string Name { get; init; } = "";
    public string Description { get; init; } = "";
    public string Kind { get; init; } = "";
    public bool Suitable { get; init; } = true;
    public string Note { get; init; } = "";
}
public sealed record InspectionResult
{
    public string Dataset { get; init; } = "";
    public int Rows { get; init; }
    public int ColumnsCount { get; init; }
    public int MissingCells { get; init; }
    public int Duplicates { get; init; }
    public string? SuggestedTarget { get; init; }
    public List<ColumnDescriptor> Columns { get; init; } = [];
    public List<ParameterDescriptor> Parameters { get; init; } = [];
    /// <summary>Plain-language observations about the data, shown after it is read.</summary>
    public List<string> Advice { get; init; } = [];
    public List<Dictionary<string,string>> Preview { get; init; } = [];
}
public sealed record ScoreResult
{
    public string Key { get; init; } = "";
    public string Name { get; init; } = "";
    public string Note { get; init; } = "";
    public bool Reference { get; init; }
    public Dictionary<string,double?> Metrics { get; init; } = [];
    public double? SelectionScore { get; init; }
    public double? BalancedAccuracy { get; init; }
    public double? Metric(string key) => Metrics.GetValueOrDefault(key);
}
public sealed record Artifact(string Name, string Path);
public sealed record FigureInfo
{
    public string Key { get; init; } = "";
    public string Title { get; init; } = "";
    public string Caption { get; init; } = "";
    public string Path { get; init; } = "";
}
public sealed record AnalysisResult
{
    public string ModuleId { get; init; } = "";
    public string SelectedModel { get; init; } = "";
    public string EvaluationMethod { get; init; } = "";
    /// <summary>Plain-language description of the final evaluation, when the module provides one.</summary>
    public string EvaluationText { get; init; } = "";
    public string SelectionMetric { get; init; } = "";
    public required ScoreResult Final { get; init; }
    public ScoreResult? Baseline { get; init; }
    public List<ScoreResult> Comparison { get; init; } = [];
    public int RowsLoaded { get; init; }
    public int RowsUsed { get; init; }
    public int? DevelopmentRows { get; init; }
    public int TestRows { get; init; }
    public List<string> ClassNames { get; init; } = [];
    public string? PositiveClass { get; init; }
    public string OutputDir { get; init; } = "";
    public string SessionDir { get; init; } = "";
    public List<string> Summary { get; init; } = [];
    public List<string> Notes { get; init; } = [];
    public List<string> Warnings { get; init; } = [];
    public List<ResultTable> Tables { get; init; } = [];
    public List<FigureInfo> Figures { get; init; } = [];
    public List<Artifact> Artifacts { get; init; } = [];
}
/// <summary>A progress event; Fraction (0-1) is the share of the work done, when the module reports it.</summary>
public sealed record WorkerProgress(string Message, string? SessionDir, double? Fraction = null);
public sealed class WorkerException(string message, string? detail = null) : Exception(message)
{
    public string? Detail { get; } = detail;
}
