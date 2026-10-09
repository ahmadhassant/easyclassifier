using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Data;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using EasyResearch.Core;

namespace EasyResearch.Desktop;

public sealed class ModelChoice(ModelDescriptor model) : INotifyPropertyChanged
{
    public ModelDescriptor Model { get; } = model;
    public string Name => Model.Name;
    public bool Available => Model.Available;
    public string Reason => Model.Available ? "" : Model.Reason;
    private bool selected = model.Available && model.DefaultSelected;
    public bool Selected { get => selected; set { selected = value; PropertyChanged?.Invoke(this,new(nameof(Selected))); } }
    public event PropertyChangedEventHandler? PropertyChanged;
}

/// <summary>A drop-down choice offered by the module after it has read the data.</summary>
public sealed class ParameterChoice(ParameterDescriptor parameter) : INotifyPropertyChanged
{
    public ParameterDescriptor Parameter { get; } = parameter;
    public string Label => Parameter.Label;
    public string Hint => Parameter.Hint;
    public List<OptionDescriptor> Options => Parameter.Options;
    private OptionDescriptor? selected = parameter.Options.FirstOrDefault(o=>o.Id==parameter.Default) ?? parameter.Options.FirstOrDefault();
    public OptionDescriptor? Selected { get => selected; set { selected = value; PropertyChanged?.Invoke(this,new(nameof(Selected))); } }
    private OptionDescriptor? First => Parameter.Options.FirstOrDefault(o=>o.Id==Parameter.Default) ?? Parameter.Options.FirstOrDefault();
    public bool IsDefault => Selected?.Id==First?.Id;
    public void Reset() => Selected=First;
    public event PropertyChangedEventHandler? PropertyChanged;
}
/// <summary>One option of an advanced tick list.</summary>
public sealed class TickItem(OptionDescriptor option, bool initial) : INotifyPropertyChanged
{
    public OptionDescriptor Option { get; } = option;
    public string Name => Option.Name;
    private bool isChecked = initial;
    public bool IsChecked { get => isChecked; set { isChecked = value; PropertyChanged?.Invoke(this,new(nameof(IsChecked))); } }
    public event PropertyChangedEventHandler? PropertyChanged;
}
/// <summary>A tick list in the Advanced options section (for example the figures or the measures).</summary>
public sealed class TickList
{
    public TickList(ListDescriptor list)
    {
        Descriptor=list;
        Items=list.Options.Select(o=>new TickItem(o,list.Default.Contains(o.Id))).ToList();
    }
    public ListDescriptor Descriptor { get; }
    public string Label => Descriptor.Label;
    public string Hint => Descriptor.Hint;
    public List<TickItem> Items { get; }
    public string[] Ticked => Items.Where(i=>i.IsChecked).Select(i=>i.Option.Id).ToArray();
    public bool IsDefault => Ticked.ToHashSet().SetEquals(Descriptor.Default);
    public void Reset() { foreach(var item in Items) item.IsChecked=Descriptor.Default.Contains(item.Option.Id); }
}
public sealed record TablePreview(string Title, string Hint, DataView View);

/// <summary>An entry in the sidebar: an installed module, or a roadmap step that is not implemented yet.</summary>
public sealed record WorkspaceItem(string Title, string Subtitle, ModuleDescriptor? Module, string? Reason = null)
{
    public bool IsAvailable => Module is not null;
}
public sealed record MetricTile(string Label, string Value);
public sealed record FigurePreview(string Title, string Caption, string Path, ImageSource? Image);

public sealed class MainViewModel : INotifyPropertyChanged
{
    // Roadmap steps shown in the sidebar until their modules are installed. They cannot be selected.
    private static readonly (string Id, string Title, string Subtitle)[] Roadmap =
    [
        ("classification", "Classification", "Predict a category"),
        ("regression", "Regression", "Predict a number"),
        ("forecasting", "Time-series forecasting", "Forecast future values"),
        ("signals", "Signal classification", "Classify ECG, EEG or sensor recordings"),
        ("vision", "Computer vision", "Step 4 · planned"),
    ];
    private static readonly List<OptionDescriptor> DefaultValidation = [new("auto","Automatic (recommended)"),new("kfold5","5-fold cross-validation"),new("kfold10","10-fold cross-validation"),new("holdout","Single 80/20 validation split")];
    private static readonly List<OptionDescriptor> DefaultSelection = [new("auto","Automatic (recommended)"),new("nested","Nested cross-validation"),new("final_test","Separate 20% final test")];

    private IWorkerClient? worker;
    private CancellationTokenSource? cancellation;
    private string? activeSession;
    private bool busy;
    private string status = "Preparing the analysis engine…";
    private string log = "";
    private string dataset = "";
    private string outputRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),"EasyResearch Results");
    private WorkspaceItem? selectedWorkspace;
    private ModuleDescriptor? selectedModule;
    private ColumnDescriptor? target;
    private InspectionResult? inspection;
    private AnalysisResult? result;
    private ModuleDescriptor? resultModule;
    private List<OptionDescriptor> validationOptions = DefaultValidation;
    private List<OptionDescriptor> selectionOptions = DefaultSelection;
    private OptionDescriptor validation = DefaultValidation[0];
    private OptionDescriptor selection = DefaultSelection[0];

    public ObservableCollection<ModuleDescriptor> Modules { get; } = [];
    public ObservableCollection<WorkspaceItem> Workspaces { get; } = [];
    public ObservableCollection<ModelChoice> Models { get; } = [];
    public ObservableCollection<ColumnDescriptor> Columns { get; } = [];
    public ObservableCollection<ParameterChoice> Parameters { get; } = [];
    /// <summary>Optional choices offered by the task (closed by default); the recommended ones are preselected.</summary>
    public ObservableCollection<ParameterChoice> AdvancedChoices { get; } = [];
    public ObservableCollection<TickList> AdvancedLists { get; } = [];
    public bool HasAdvancedOptions => AdvancedChoices.Count>0 || AdvancedLists.Count>0;
    public ObservableCollection<TablePreview> ResultTables { get; } = [];
    public ObservableCollection<MetricTile> HeadlineTiles { get; } = [];
    public ObservableCollection<FigurePreview> Figures { get; } = [];
    public ObservableCollection<Artifact> Artifacts { get; } = [];
    private bool createFigures = true;
    public bool CreateFigures { get=>createFigures; set { createFigures=value; Changed(); RefreshState(); } }
    public DataView? Preview { get; private set; }
    public DataView? ComparisonTable { get; private set; }

    public bool Busy { get => busy; private set { busy=value; Changed(); RefreshState(); Changed(nameof(ProgressIndeterminate)); } }
    private double progressValue;
    private bool progressKnown;
    /// <summary>Share of the analysis done, 0-100, when the module reports it.</summary>
    public double ProgressValue { get=>progressValue; private set { progressValue=value; Changed(); Changed(nameof(ProgressText)); } }
    public bool ProgressKnown { get=>progressKnown; private set { progressKnown=value; Changed(); Changed(nameof(ProgressIndeterminate)); Changed(nameof(ProgressText)); } }
    public bool ProgressIndeterminate => Busy && !ProgressKnown;
    public string ProgressText => ProgressKnown ? $"{ProgressValue:0}%" : "";
    private void ResetProgress() { ProgressKnown=false; ProgressValue=0; }
    public bool Idle => !Busy;
    public bool CanRun => !Busy && worker is not null && RunHint.Length==0;
    public bool HasResult => Result is not null;
    public string Status { get => status; private set { status=value; Changed(); } }
    public string ActivityLog { get => log; private set { log=value; Changed(); } }
    public string Dataset { get => dataset; private set { dataset=value; Changed(); Changed(nameof(DatasetDisplay)); } }
    public string DatasetDisplay => Dataset=="demo" ? "Built-in example data" : Dataset;
    public string OutputRoot { get=>outputRoot; set { outputRoot=value; Changed(); RefreshState(); } }
    public List<OptionDescriptor> ValidationOptions { get=>validationOptions; private set { validationOptions=value; Changed(); } }
    public List<OptionDescriptor> SelectionOptions { get=>selectionOptions; private set { selectionOptions=value; Changed(); } }
    public OptionDescriptor Validation { get=>validation; set { validation=value; Changed(); RefreshState(); } }
    public OptionDescriptor Selection { get=>selection; set { selection=value; Changed(); RefreshState(); } }
    public bool ShowEvaluationSettings => SelectedModule is { } module && (module.ValidationOptions.Count>0 || module.SelectionOptions.Count>0);

    public WorkspaceItem? SelectedWorkspace
    {
        get=>selectedWorkspace;
        set
        {
            // Roadmap entries are shown but cannot be opened.
            if(value is null || value.Module is null || ReferenceEquals(value,selectedWorkspace)) { Changed(); return; }
            selectedWorkspace=value;
            Changed();
            SelectedModule=value.Module;
        }
    }
    public ModuleDescriptor? SelectedModule
    {
        get=>selectedModule;
        private set
        {
            selectedModule=value;
            Models.Clear();
            if(value is not null)
                foreach(var model in value.Models) { var choice=new ModelChoice(model); choice.PropertyChanged+=(_,_)=>RefreshState(); Models.Add(choice); }
            AdvancedChoices.Clear(); AdvancedLists.Clear();
            if(value is not null)
            {
                foreach(var parameter in value.AdvancedChoices.Where(p=>p.Options.Count>0))
                {
                    var choice=new ParameterChoice(parameter);
                    choice.PropertyChanged+=(_,_)=>RefreshState();
                    AdvancedChoices.Add(choice);
                }
                foreach(var list in value.AdvancedLists.Where(l=>l.Options.Count>0))
                {
                    var ticks=new TickList(list);
                    foreach(var item in ticks.Items) item.PropertyChanged+=(_,_)=>RefreshState();
                    AdvancedLists.Add(ticks);
                }
            }
            ValidationOptions=value is { ValidationOptions.Count: >0 } ? value.ValidationOptions : DefaultValidation;
            SelectionOptions=value is { SelectionOptions.Count: >0 } ? value.SelectionOptions : DefaultSelection;
            Validation=ValidationOptions[0]; Selection=SelectionOptions[0];
            // Never keep an inspection or result made for a different task.
            var previous=Dataset;
            Inspection=null; Target=null; Columns.Clear(); Parameters.Clear(); Preview=null; Result=null; Dataset="";
            Changed(); Changed(nameof(Labels)); Changed(nameof(ModuleDescription)); Changed(nameof(Preview)); Changed(nameof(ShowEvaluationSettings)); Changed(nameof(HasAdvancedOptions)); RefreshState();
            // A spreadsheet chosen in one module is re-read for the new one; each module has its own example data.
            if(value is not null && worker is not null && previous.Length>0 && previous!="demo")
                _ = LoadDatasetAsync(previous);
        }
    }
    public ModuleLabels Labels => SelectedModule?.Labels ?? new();
    public string ModuleDescription => SelectedModule?.Description ?? "";
    public ColumnDescriptor? Target { get=>target; set { target=value; Changed(); Changed(nameof(TargetNote)); RefreshState(); } }
    public string TargetNote => Target is null ? "" : Target.Suitable ? Target.Note : "This column cannot be used here. " + Target.Note;
    public InspectionResult? Inspection { get=>inspection; private set { inspection=value; Changed(); Changed(nameof(DataSummary)); Changed(nameof(AdviceText)); Changed(nameof(HasAdvice)); RefreshState(); } }
    public bool HasAdvice => Inspection is { Advice.Count: >0 };
    public string AdviceText => Inspection is null ? "" : string.Join("\n",Inspection.Advice.Select(a=>"•  "+a));

    /// <summary>What is still missing before the analysis can run (empty when ready).</summary>
    public string RunHint
    {
        get
        {
            if(Busy) return "An analysis is running. Cancel stops it at any time.";
            if(worker is null) return "The analysis engine is not ready yet.";
            if(SelectedModule is null) return "Choose a task in the sidebar.";
            if(Inspection is null) return "Step 1: choose your data file, or try the example.";
            if(Target is null) return "Step 2: choose the column to analyse.";
            if(!Target.Suitable) return "Step 2: the chosen column cannot be used here; see the note below it.";
            if(!Models.Any(m=>m.Selected && m.Available)) return $"Step 3: select at least one {Labels.ModelNoun}.";
            if(AdvancedLists.FirstOrDefault(l=>l.Descriptor.Id!="figure_keys" && l.Ticked.Length==0) is { } empty)
                return $"Step 3: tick at least one item under ‘{empty.Label}’ in Advanced options.";
            if(string.IsNullOrWhiteSpace(OutputRoot) || !Path.IsPathFullyQualified(OutputRoot)) return "Step 4: choose a full path for the results folder.";
            return "";
        }
    }
    /// <summary>A plain summary of what the analysis will do, for the user to confirm before running.</summary>
    public string PlanText
    {
        get
        {
            var hint=RunHint;
            if(hint.Length>0 || SelectedModule is null || Inspection is null || Target is null) return hint;
            var chosen=Models.Where(m=>m.Selected && m.Available).ToList();
            var lines=new List<string>
            {
                $"Task: {SelectedModule.Title}",
                $"Data: {DatasetDisplay} ({Inspection.Rows:N0} rows)",
                $"Column: {Target.Name}"
            };
            foreach(var parameter in Parameters)
                if(parameter.Selected is { } option) lines.Add($"{parameter.Label}: {option.Name}");
            lines.Add($"{chosen.Count} {(chosen.Count==1 ? Labels.ModelNoun : Labels.ModelNounPlural)}: {string.Join(", ",chosen.Select(m=>m.Name))}");
            if(ShowEvaluationSettings) lines.Add($"Evaluation: {Validation.Name}; final score: {Selection.Name}");
            // Only the advanced choices that differ from the recommended ones are listed.
            static string Short(string name) => name.Split(" - ")[0];
            var advanced=AdvancedChoices.Where(c=>!c.IsDefault && c.Selected is not null).Select(c=>$"{c.Label}: {Short(c.Selected!.Name)}")
                .Concat(AdvancedLists.Where(l=>!l.IsDefault).Select(l=>$"{l.Label}: {(l.Ticked.Length==0 ? "none" : string.Join(", ",l.Items.Where(i=>i.IsChecked).Select(i=>i.Name)))}"))
                .ToList();
            if(advanced.Count>0) lines.Add("Advanced options: "+string.Join("; ",advanced));
            lines.Add($"Results: a new folder inside {OutputRoot}");
            if(chosen.Any(m=>m.Name.StartsWith("Deep",StringComparison.Ordinal)))
                lines.Add("Deep models are selected, so this may take several minutes.");
            return string.Join("\n",lines);
        }
    }
    public string DataSummary => Inspection is null ? "Choose your spreadsheet or try the built-in example." : $"{Inspection.Rows:N0} rows  ·  {Inspection.ColumnsCount} columns  ·  {Inspection.MissingCells:N0} missing values  ·  {Inspection.Duplicates:N0} duplicate rows";

    public AnalysisResult? Result
    {
        get=>result;
        private set
        {
            result=value;
            resultModule=value is null ? null : Modules.FirstOrDefault(m=>m.Id==value.ModuleId) ?? SelectedModule;
            HeadlineTiles.Clear(); Figures.Clear(); Artifacts.Clear(); ResultTables.Clear(); ComparisonTable=null;
            if(value is not null && resultModule is not null)
            {
                var specs=resultModule.Metrics;
                foreach(var key in resultModule.Headline)
                    if(specs.FirstOrDefault(m=>m.Key==key) is { } spec && value.Final.Metrics.ContainsKey(key))   // measures left out in Advanced options are not shown
                        HeadlineTiles.Add(new(spec.Label,spec.Display(value.Final.Metric(key))));
                ComparisonTable=BuildComparison(value,resultModule);
                foreach(var table in value.Tables) ResultTables.Add(new(table.Title,table.Hint,BuildTable(table)));
                foreach(var figure in value.Figures) Figures.Add(new(figure.Title,figure.Caption,figure.Path,LoadImage(figure.Path)));
                foreach(var file in value.Artifacts) Artifacts.Add(file);
            }
            Changed(); Changed(nameof(HasResult)); Changed(nameof(ComparisonTable)); Changed(nameof(FinalExplanation)); Changed(nameof(ResultNotes));
            Changed(nameof(SummaryText)); Changed(nameof(ComparisonTitle)); Changed(nameof(ComparisonHint)); Changed(nameof(HasFigures));
        }
    }
    private string Noun => resultModule?.Labels.ModelNoun ?? "model";
    public string ComparisonTitle => char.ToUpper(Noun[0])+Noun[1..]+" comparison";
    public string ComparisonHint => $"These scores ({Result?.SelectionMetric ?? ""}) were used to choose the {Noun}. Report the final evaluation above.";
    public bool HasFigures => Figures.Count>0;
    public string SummaryText => Result is null ? "" : string.Join("\n",Result.Summary);
    public string FinalExplanation => Result is { EvaluationText.Length: >0 } custom ? custom.EvaluationText : Result?.EvaluationMethod switch
    {
        "final_test" => $"Final evaluation on {Result.TestRows:N0} held-out rows. The selected {Noun} was trained on {Result.DevelopmentRows:N0} development rows.",
        "nested" => $"Nested cross-validation evaluates the whole selection procedure. The chosen {Noun} can differ between outer folds. {Result.Final.Note}".Trim(),
        "holdout" => $"Evaluation of your preselected {Noun} on a held-out validation split.",
        "cross_validation" => $"Cross-validation evaluation of your preselected {Noun}.",
        _ => ""
    };
    public string ResultNotes
    {
        get
        {
            if(Result is null) return "";
            var lines=new List<string> { $"{Result.RowsUsed:N0} of {Result.RowsLoaded:N0} rows used." };
            if(Result.ClassNames.Count>0)
                lines.Add(Result.PositiveClass is not null ? $"Precision, recall and F1 treat ‘{Result.PositiveClass}’ as positive." : "Multiclass precision, recall and F1 use macro averaging.");
            return string.Join("\n",lines.Concat(Result.Notes).Concat(Result.Warnings));
        }
    }

    public async Task InitializeAsync()
    {
        Busy=true;
        cancellation=new();
        try
        {
            var paths=WorkerPaths.FromApplication(AppContext.BaseDirectory);
            AppendLog($"Analysis engine: {paths.Python} (from {paths.Source}).");
            worker=new WorkerClient(paths);
            var catalog=await worker.SendAsync<CatalogResult>(new() { Action="catalog" },Progress(),cancellation.Token);
            // A task that cannot load stays visible, greyed out, with its reason (never hidden silently).
            var unavailable=catalog.Modules.Where(m=>m.UnavailableReason.Length>0).ToList();
            foreach(var module in catalog.Modules.Where(m=>m.InputKind=="tabular" && m.UnavailableReason.Length==0)) Modules.Add(module);
            WorkspaceItem Broken(ModuleDescriptor m) => new(m.Title,"Not available · hover for the reason",null,"Not available: "+m.UnavailableReason);
            foreach(var step in Roadmap)
            {
                var module=Modules.FirstOrDefault(m=>m.Id==step.Id);
                var broken=unavailable.FirstOrDefault(m=>m.Id==step.Id);
                if(module is not null) Workspaces.Add(new(module.Title,step.Subtitle,module));
                else if(broken is not null) Workspaces.Add(Broken(broken));
                else if(step.Id=="vision") Workspaces.Add(new(step.Title,step.Subtitle,null,"Planned for a later release."));
            }
            foreach(var module in Modules.Where(m=>Roadmap.All(r=>r.Id!=m.Id))) Workspaces.Add(new(module.Title,module.Description,module));
            foreach(var broken in unavailable.Where(m=>Roadmap.All(r=>r.Id!=m.Id))) Workspaces.Add(Broken(broken));
            SelectedWorkspace=Workspaces.FirstOrDefault(w=>w.IsAvailable);
            Status=SelectedModule is null ? "No analysis task could be loaded. Open the activity log below to see why."
                 : unavailable.Count>0 ? $"Ready. {unavailable.Count} task(s) could not be loaded; open the activity log below to see why."
                 : "Ready. Load a spreadsheet to begin.";
        }
        catch(OperationCanceledException) { Status="Engine loading cancelled. Restart the application to try again."; }
        catch(Exception ex) { ShowError(ex); }
        finally { cancellation.Dispose(); cancellation=null; Busy=false; }
    }
    public void SelectModule(string id)
    {
        if(Workspaces.FirstOrDefault(w=>w.Module?.Id==id) is { } item) SelectedWorkspace=item;
    }
    public async Task LoadDatasetAsync(string path)
    {
        if(Busy) return;
        if(worker is null || SelectedModule is null)
        {
            AppendLog(Status);
            System.Windows.MessageBox.Show(Status, "Cannot load dataset", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Information);
            return;
        }
        Busy=true; Result=null; Inspection=null; Columns.Clear(); Parameters.Clear(); Target=null; Preview=null; Changed(nameof(Preview)); ResetProgress();
        Dataset=path;
        Status="Reading your data…";
        cancellation=new();
        try
        {
            var info=await worker.SendAsync<InspectionResult>(new() { Action="inspect",ModuleId=SelectedModule.Id,Dataset=path },Progress(),cancellation.Token);
            Inspection=info;
            foreach(var column in info.Columns) Columns.Add(column);
            foreach(var parameter in info.Parameters.Where(p=>p.Options.Count>0))
            {
                var choice=new ParameterChoice(parameter);
                choice.PropertyChanged+=(_,_)=>RefreshState();
                Parameters.Add(choice);
            }
            Target=Columns.FirstOrDefault(c=>c.Name==info.SuggestedTarget && c.Suitable);
            var table=new DataTable();
            for(int i=0;i<info.Columns.Count;i++) table.Columns.Add(new DataColumn($"c{i}") { Caption=info.Columns[i].Name });
            foreach(var row in info.Preview) table.Rows.Add(info.Columns.Select(c=>(object)row.GetValueOrDefault(c.Name,"")).ToArray());
            Preview=table.DefaultView; Changed(nameof(Preview));
            Status=Target is null
                ? $"Data loaded, but no column looks suitable for {SelectedModule.Title.ToLowerInvariant()}. Check the list of columns."
                : $"Data loaded. Check the column to predict and choose your {Labels.ModelNounPlural}.";
        }
        catch(OperationCanceledException) { Status="Loading cancelled."; }
        catch(Exception ex) { ShowError(ex); }
        finally { cancellation.Dispose(); cancellation=null; Busy=false; }
    }
    public async Task RunAsync()
    {
        if(!CanRun || worker is null || SelectedModule is null || Target is null) return;
        if(string.IsNullOrWhiteSpace(OutputRoot) || !Path.IsPathFullyQualified(OutputRoot)) { Status="Choose a full path for your output folder."; return; }
        var settings=new Dictionary<string,object> { ["models"]=Models.Where(m=>m.Selected && m.Available).Select(m=>m.Model.Id).ToArray(),["validation"]=Validation.Id,["selection"]=Selection.Id,["figures"]=CreateFigures };
        foreach(var parameter in Parameters)
            if(parameter.Selected is { } option) settings[parameter.Parameter.Id]=option.Id;
        foreach(var choice in AdvancedChoices)
            if(choice.Selected is { } option) settings[choice.Parameter.Id]=option.Id;
        foreach(var list in AdvancedLists) settings[list.Descriptor.Id]=list.Ticked;
        var request=new WorkerRequest
        {
            Action="run",ModuleId=SelectedModule.Id,Dataset=Dataset,Target=Target.Name,OutputRoot=OutputRoot,
            Settings=settings
        };
        Busy=true; Result=null; activeSession=null; ActivityLog=""; Status="Starting your analysis…"; ResetProgress();
        cancellation=new();
        try
        {
            Result=await worker.SendAsync<AnalysisResult>(request,Progress(),cancellation.Token);
            if(ProgressKnown) ProgressValue=100;
            Status=Result.Warnings.Count>0 ? "Analysis completed. Some optional outputs need attention; see notes." : "Analysis complete. Your reports and model are ready.";
        }
        catch(OperationCanceledException)
        {
            ResetProgress();
            Status="Analysis cancelled. Partial output is not a completed result.";
            if(activeSession is not null)
            {
                try { await File.WriteAllTextAsync(Path.Combine(activeSession,"status.json"),"{\"state\":\"cancelled\"}"); }
                catch(IOException) { }
                catch(UnauthorizedAccessException) { }
            }
        }
        catch(Exception ex) { ShowError(ex); }
        finally { cancellation.Dispose(); cancellation=null; Busy=false; }
    }
    public void Cancel() => cancellation?.Cancel();
    public void RestoreAdvancedDefaults()
    {
        foreach(var choice in AdvancedChoices) choice.Reset();
        foreach(var list in AdvancedLists) list.Reset();
        RefreshState();
    }
    public void SelectModels(bool all)
    {
        foreach(var model in Models) model.Selected=model.Available && (all || model.Model.DefaultSelected);
    }

    private static DataView BuildComparison(AnalysisResult value,ModuleDescriptor module)
    {
        var rows=value.Comparison;
        var metrics=module.Metrics.Where(m=>rows.Any(r=>r.Metric(m.Key) is not null)).ToList();
        var table=new DataTable();
        var noun=module.Labels.ModelNoun;
        table.Columns.Add(new DataColumn("c0") { Caption=char.ToUpper(noun[0])+noun[1..] });
        for(int i=0;i<metrics.Count;i++) table.Columns.Add(new DataColumn($"c{i+1}") { Caption=metrics[i].Label });
        foreach(var row in rows)
        {
            var name=row.Name==value.SelectedModel && !row.Reference ? row.Name+"  ✓ selected" : row.Name;
            table.Rows.Add(new object[] { name }.Concat(metrics.Select(m=>(object)m.Display(row.Metric(m.Key)))).ToArray());
        }
        return table.DefaultView;
    }
    private static DataView BuildTable(ResultTable source)
    {
        var table=new DataTable();
        for(int i=0;i<source.Columns.Count;i++) table.Columns.Add(new DataColumn($"c{i}") { Caption=source.Columns[i] });
        foreach(var row in source.Rows)
            table.Rows.Add(Enumerable.Range(0,source.Columns.Count).Select(i=>(object)(i<row.Count ? row[i] : "")).ToArray());
        return table.DefaultView;
    }
    private static ImageSource? LoadImage(string path)
    {
        try
        {
            if(!File.Exists(path)) return null;
            var image=new BitmapImage();
            image.BeginInit();
            image.CacheOption=BitmapCacheOption.OnLoad;          // releases the file at once
            image.CreateOptions=BitmapCreateOptions.IgnoreImageCache;
            image.DecodePixelWidth=900;
            image.UriSource=new Uri(path);
            image.EndInit();
            image.Freeze();
            return image;
        }
        catch(Exception) { return null; }
    }
    private static readonly string[] StatusPrefixes = ["Training ","Scoring ","Running nested","Drawing figures","Writing the report","Preparing","Measuring"];
    private IProgress<WorkerProgress> Progress() => new Progress<WorkerProgress>(p=>
    {
        if(p.SessionDir is not null) activeSession=p.SessionDir;
        if(p.Fraction is { } share)
        {
            ProgressKnown=true;
            ProgressValue=Math.Max(ProgressValue,Math.Clamp(share*100,0,100));   // never moves backwards
        }
        if(p.Message.Length==0) return;           // a progress-only update (deep-learning epochs)
        AppendLog(p.Message);
        if(StatusPrefixes.Any(prefix=>p.Message.StartsWith(prefix,StringComparison.Ordinal))) Status=p.Message;
    });
    private void AppendLog(string line)
    {
        var text=ActivityLog+line+Environment.NewLine;
        ActivityLog=text.Length>60000 ? text[^60000..] : text;
    }
    private void ShowError(Exception ex)
    {
        ResetProgress();
        Status=ex.Message;
        AppendLog(ex is WorkerException { Detail: not null } engine ? engine.Detail : ex.ToString());
    }
    private void RefreshState() { Changed(nameof(Idle)); Changed(nameof(CanRun)); Changed(nameof(RunHint)); Changed(nameof(PlanText)); }
    private void Changed([CallerMemberName] string? name=null) => PropertyChanged?.Invoke(this,new(name));
    public event PropertyChangedEventHandler? PropertyChanged;
}
