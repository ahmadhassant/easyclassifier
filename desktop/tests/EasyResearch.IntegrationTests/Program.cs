using System.Diagnostics;
using System.Text.Json;
using EasyResearch.Core;

if(args.Length<3) throw new ArgumentException("Usage: integration-tests <python.exe> <project-root> <test-output> [large-dataset.csv]");
var python=Path.GetFullPath(args[0]);
var root=Path.GetFullPath(args[1]);
var output=Path.GetFullPath(args[2]);
Directory.CreateDirectory(output);
var client=new WorkerClient(new(python,Path.Combine(root,"worker","worker.py"),Path.Combine(root,"modules"),Path.Combine(output,"plot-cache")));
void Check(bool condition,string message) { if(!condition) throw new Exception(message); Console.WriteLine("PASS "+message); }
var catalog=await client.SendAsync<CatalogResult>(new() { Action="catalog" });
Check(catalog.Modules.Select(m=>m.Id).SequenceEqual(new[] { "classification","forecasting","regression","signals" }),"module discovery and C# contract deserialization");
Check(catalog.Modules.All(m=>m.UnavailableReason.Length==0),"every module loads with this Python: "+string.Join("; ",catalog.Modules.Select(m=>m.UnavailableReason).Where(r=>r.Length>0)));
var regression=catalog.Modules.Single(m=>m.Id=="regression");
Check(regression.Headline.SequenceEqual(new[] { "r2","rmse","mae" }) && regression.Labels.ModelNoun=="model" && regression.Metrics.Count==6 && regression.Metrics.Any(m=>m.Key=="mhsp"),"module labels and metric descriptors reach C#");
var inspect=await client.SendAsync<InspectionResult>(new() { Action="inspect",ModuleId="classification",Dataset="demo" });
Check(inspect.Rows==150 && inspect.SuggestedTarget=="Species","demo inspection and target suggestion");
Check(inspect.Advice.Count>0 && inspect.Advice.Any(a=>a.Contains("nested cross-validation")),"plain-language advice reaches C#");
Check(File.Exists(Path.Combine(root,"examples","README.txt")) && Directory.GetFiles(Path.Combine(root,"examples"),"*.csv").Length==4,"an example file for every task");
try { await client.SendAsync<InspectionResult>(new() { Action="inspect",ModuleId="classification",Dataset="missing-file.csv" }); throw new Exception("Expected error"); }
catch(WorkerException ex) { Check(ex.Detail is not null,"Python error propagated through C# without false completion"); }
var run=await client.SendAsync<AnalysisResult>(new()
{
 Action="run",ModuleId="classification",Dataset="demo",Target="Species",OutputRoot=output,
 Settings=new() { ["classifiers"]=new[] { "decision_tree","logistic_regression" },["figures"]=true }
});
Check(run.EvaluationMethod=="nested" && run.Final.Key=="nested","nested selection uses the final estimate");
Check(run.Comparison.Count==2 && run.Final.Metrics["accuracy"]>.85,"C# receives real evaluation metrics");
Check(File.Exists(Path.Combine(run.OutputDir,"summary.csv")) && run.Artifacts.Any(a=>a.Name.EndsWith(".png")),"reports and figures are discoverable");
await File.WriteAllTextAsync(Path.Combine(output,"csharp-demo-result.json"),JsonSerializer.Serialize(run,Wire.Json));
var classification=catalog.Modules.Single(m=>m.Id=="classification");
Check(classification.AdvancedChoices.Any(c=>c.Id=="knn_distance" && c.Default=="hassanat") && classification.AdvancedLists.Single(l=>l.Id=="metrics").Default.Count==7
      && catalog.Modules.All(m=>m.AdvancedLists.Any(l=>l.Id=="figure_keys" && l.Default.All(d=>l.Options.Any(o=>o.Id==d)))),"advanced options reach C#");
var advanced=await client.SendAsync<AnalysisResult>(new()
{
 Action="run",ModuleId="classification",Dataset="demo",Target="Species",OutputRoot=output,
 Settings=new() { ["models"]=new[] { "knn" },["knn_distance"]="manhattan",["knn_k"]="7",["metrics"]=new[] { "balanced_accuracy","cohen_kappa" },
                  ["figure_keys"]=new[] { "confusion_matrix","learning_curve" },["figure_format"]="png+svg" }
});
Check(advanced.SelectedModel.Contains("k=7") && advanced.Final.Metrics.Keys.ToHashSet().SetEquals(new[] { "balanced_accuracy","cohen_kappa" })
      && advanced.Figures.Select(f=>f.Key).SequenceEqual(new[] { "confusion_matrix","learning_curve" }) && advanced.Artifacts.Any(a=>a.Name.EndsWith(".svg")),
      "advanced choices sent from C# change the analysis");
if(args.Length==4)
{
 var large=await client.SendAsync<AnalysisResult>(new()
 {
  Action="run",ModuleId="classification",Dataset=Path.GetFullPath(args[3]),Target="Outcome",OutputRoot=output,
  Settings=new() { ["classifiers"]=new[] { "decision_tree","random_forest","logistic_regression" },["figures"]=false }
 });
 Check(large.RowsUsed>2000 && large.EvaluationMethod=="final_test","larger dataset selects the final-test path");
 Check(large.DevelopmentRows+large.TestRows==large.RowsUsed,"held-out and development row counts match");
 await File.WriteAllTextAsync(Path.Combine(output,"csharp-large-result.json"),JsonSerializer.Serialize(large,Wire.Json));
}
var rinspect=await client.SendAsync<InspectionResult>(new() { Action="inspect",ModuleId="regression",Dataset="demo" });
Check(rinspect.SuggestedTarget=="Progression" && !rinspect.Columns.Single(c=>c.Name=="sex").Suitable,"regression inspection marks suitable targets");
var fractions=new System.Collections.Concurrent.ConcurrentQueue<double>();
var reg=await client.SendAsync<AnalysisResult>(new()
{
 Action="run",ModuleId="regression",Dataset="demo",Target="Progression",OutputRoot=output,
 Settings=new() { ["models"]=new[] { "linear_regression","random_forest" },["figures"]=true }
},new Progress<WorkerProgress>(p=>{ if(p.Fraction is { } f) fractions.Enqueue(f); }));
await Task.Delay(200);
Check(fractions.Count>=5 && fractions.All(f=>f>=0 && f<1),"progress fractions reach C#");
Check(reg.EvaluationMethod=="nested" && reg.Final.Metric("r2")>.4 && reg.Baseline is { Reference: true },"regression nested estimate and reference row");
Check(reg.Figures.Count>=4 && reg.Figures.All(f=>File.Exists(f.Path)),"regression figures are listed for the gallery");
Check(regression.Metrics.Single(m=>m.Key=="rmse").Display(reg.Final.Metric("rmse"))!="—","metric formatting");
await File.WriteAllTextAsync(Path.Combine(output,"csharp-regression-result.json"),JsonSerializer.Serialize(reg,Wire.Json));
var forecasting=catalog.Modules.Single(m=>m.Id=="forecasting");
Check(forecasting.ValidationOptions.Count==0 && forecasting.Headline.SequenceEqual(new[] { "mase","mae","mhsp" }) && forecasting.Models.All(m=>m.Id is not "naive" and not "seasonal_naive"),"forecasting module descriptor reaches C#");
var finspect=await client.SendAsync<InspectionResult>(new() { Action="inspect",ModuleId="forecasting",Dataset="demo" });
Check(finspect.SuggestedTarget=="CO2_ppm" && finspect.Parameters.Select(p=>p.Id).SequenceEqual(new[] { "time","horizon","metric" }) && finspect.Parameters.All(p=>p.Options.Any(o=>o.Id==p.Default)),"forecasting choices (date column, horizon, measure) reach C#");
var fc=await client.SendAsync<AnalysisResult>(new()
{
 Action="run",ModuleId="forecasting",Dataset="demo",Target="CO2_ppm",OutputRoot=output,
 Settings=new() { ["models"]=new[] { "ridge","theta" },["time"]="auto",["horizon"]="6",["metric"]="mase",["figures"]=true }
});
Check(fc.EvaluationMethod=="rolling_origin" && fc.EvaluationText.Length>0 && fc.Final.Metric("mase")<1 && fc.Baseline is { Reference: true },"forecast final-period evaluation and reference rule");
Check(fc.Tables.Count==1 && fc.Tables[0].Rows.Count==6 && fc.Tables[0].Rows.All(r=>r.Count==fc.Tables[0].Columns.Count),"forecast table reaches C#");
Check(fc.Figures.Count>=4 && fc.Figures.All(f=>File.Exists(f.Path)),"forecast figures are listed for the gallery");
await File.WriteAllTextAsync(Path.Combine(output,"csharp-forecast-result.json"),JsonSerializer.Serialize(fc,Wire.Json));
var signals=catalog.Modules.Single(m=>m.Id=="signals");
Check(signals.FileFilter.Contains("*.tsv") && signals.Models.All(m=>m.Id!="majority") && regression.FileFilter.Contains("*.xlsx"),"signal module descriptor and file types reach C#");
var sinspect=await client.SendAsync<InspectionResult>(new() { Action="inspect",ModuleId="signals",Dataset="demo" });
Check(sinspect.SuggestedTarget=="label" && sinspect.Parameters.Select(p=>p.Id).SequenceEqual(new[] { "first","last" }),"signal range choices reach C#");
var sig=await client.SendAsync<AnalysisResult>(new()
{
 Action="run",ModuleId="signals",Dataset="demo",Target="label",OutputRoot=output,
 Settings=new() { ["models"]=new[] { "knn" },["first"]="t1",["last"]="t140",["figures"]=false }
});
Check(sig.EvaluationMethod=="cross_validation" && sig.Final.Metric("balanced_accuracy")>.6 && sig.ClassNames.Count==3 && sig.Baseline is { Reference: true },"signal classification result and reference row");
await File.WriteAllTextAsync(Path.Combine(output,"csharp-signals-result.json"),JsonSerializer.Serialize(sig,Wire.Json));
var watch=Stopwatch.StartNew();
using var cancel=new CancellationTokenSource(TimeSpan.FromMilliseconds(150));
try { await client.SendAsync<CatalogResult>(new() { Action="catalog" },cancellation:cancel.Token); throw new Exception("Expected cancellation"); }
catch(OperationCanceledException) { Check(watch.Elapsed<TimeSpan.FromSeconds(10),"cancellation terminates the worker promptly"); }
Console.WriteLine("All C# integration checks passed.");
