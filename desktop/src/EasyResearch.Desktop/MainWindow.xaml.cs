using System.Data;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using EasyResearch.Core;
using Microsoft.Win32;

namespace EasyResearch.Desktop;

public partial class MainWindow : Window
{
    private readonly MainViewModel viewModel;
    public MainWindow(MainViewModel viewModel)
    {
        InitializeComponent();
        this.viewModel=viewModel;
        DataContext=viewModel;
        Closing+=(_,_)=>viewModel.Cancel();
    }
    private async void Browse_Click(object sender,RoutedEventArgs e)
    {
        var dialog=new OpenFileDialog { Title="Choose your dataset",Filter=viewModel.SelectedModule?.FileFilter ?? "CSV and Excel files|*.csv;*.xlsx;*.xlsm|All files|*.*" };
        if(dialog.ShowDialog(this)==true) await viewModel.LoadDatasetAsync(dialog.FileName);
    }
    private async void Demo_Click(object sender,RoutedEventArgs e) => await viewModel.LoadDatasetAsync("demo");
    /// <summary>Opens the example files shipped with the application (one or more per task).</summary>
    private async void Examples_Click(object sender,RoutedEventArgs e)
    {
        var dialog=new OpenFileDialog { Title="Choose an example file",Filter=viewModel.SelectedModule?.FileFilter ?? "CSV and Excel files|*.csv;*.xlsx;*.xlsm|All files|*.*" };
        var examples=Path.Combine(AppContext.BaseDirectory,"examples");
        if(Directory.Exists(examples)) dialog.InitialDirectory=examples;
        if(dialog.ShowDialog(this)==true) await viewModel.LoadDatasetAsync(dialog.FileName);
    }
    private async void Run_Click(object sender,RoutedEventArgs e)
    {
        await viewModel.RunAsync();
        if(viewModel.HasResult) MainTabs.SelectedItem=ResultsTab;
    }
    private void Cancel_Click(object sender,RoutedEventArgs e) => viewModel.Cancel();
    private void Fast_Click(object sender,RoutedEventArgs e) => viewModel.SelectModels(false);
    private void All_Click(object sender,RoutedEventArgs e) => viewModel.SelectModels(true);
    private void RestoreAdvanced_Click(object sender,RoutedEventArgs e) => viewModel.RestoreAdvancedDefaults();
    private void Output_Click(object sender,RoutedEventArgs e)
    {
        var dialog=new OpenFolderDialog { Title="Choose an output folder" };
        if(Directory.Exists(viewModel.OutputRoot)) dialog.InitialDirectory=viewModel.OutputRoot;
        if(dialog.ShowDialog(this)==true) viewModel.OutputRoot=dialog.FolderName;
    }
    private void OpenResults_Click(object sender,RoutedEventArgs e) => Open(viewModel.Result?.OutputDir);
    private void OpenReport_Click(object sender,RoutedEventArgs e)
    {
        if(viewModel.Result is not { } result) return;
        var pdf=Path.Combine(result.OutputDir,"report.pdf");
        Open(File.Exists(pdf) ? pdf : Path.Combine(result.OutputDir,"report.tex"));
    }
    private void Artifact_DoubleClick(object sender,MouseButtonEventArgs e)
    {
        if(ArtifactsList.SelectedItem is Artifact artifact) Open(artifact.Path);
    }
    private void Figure_Click(object sender,MouseButtonEventArgs e)
    {
        if(sender is FrameworkElement { DataContext: FigurePreview figure }) Open(figure.Path);
    }
    /// <summary>
    /// Tables use neutral column ids (c0, c1, …) with the real name as caption, so column names
    /// containing dots, brackets or slashes, or differing only in case, display correctly.
    /// </summary>
    private void Table_AutoGeneratingColumn(object? sender,DataGridAutoGeneratingColumnEventArgs e)
    {
        if(sender is not DataGrid { ItemsSource: DataView { Table: { } table } } grid || !table.Columns.Contains(e.PropertyName)) return;
        var column=table.Columns[e.PropertyName]!;
        e.Column=new DataGridTextColumn
        {
            Header=column.Caption,
            Binding=new Binding($"[{e.PropertyName}]"),
            Width=column.Ordinal==0 && Equals(grid.Tag,"stretch-first") ? new DataGridLength(1,DataGridLengthUnitType.Star) : DataGridLength.Auto,
            MinWidth=column.Ordinal==0 ? 160 : 90
        };
    }
    private void Open(string? path)
    {
        if(string.IsNullOrWhiteSpace(path)) return;
        try { Process.Start(new ProcessStartInfo(path) { UseShellExecute=true }); }
        catch(Exception ex) { MessageBox.Show(this,ex.Message,"Could not open file",MessageBoxButton.OK,MessageBoxImage.Information); }
    }
}
