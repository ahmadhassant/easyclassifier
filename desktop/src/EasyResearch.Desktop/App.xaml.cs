using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace EasyResearch.Desktop;

public partial class App : Application
{
    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        var viewModel = new MainViewModel();
        var window = new MainWindow(viewModel);
        MainWindow = window;
        // --render-preview|--verify-ui <png> [module id]
        if (e.Args.Length is 2 or 3 && (e.Args[0] == "--render-preview" || e.Args[0] == "--verify-ui"))
        {
            window.ShowActivated = false;
            window.Opacity = 0;
            window.Show();
            await viewModel.InitializeAsync();
            if (e.Args.Length == 3) viewModel.SelectModule(e.Args[2]);
            await viewModel.LoadDatasetAsync("demo");
            if (!viewModel.CanRun) throw new InvalidOperationException(viewModel.Status);
            if (e.Args[0] == "--verify-ui")
            {
                viewModel.OutputRoot = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(e.Args[1]))!, "ui-test-results");
                await viewModel.RunAsync();
                if (!viewModel.HasResult) throw new InvalidOperationException(viewModel.ActivityLog);
                window.ResultsTab.IsSelected = true;
            }
            var surface=(FrameworkElement)window.Content;
            surface.Measure(new Size(1250, 860));
            surface.Arrange(new Rect(0, 0, 1250, 860));
            surface.UpdateLayout();
            await Dispatcher.InvokeAsync(() => { }, DispatcherPriority.ContextIdle);
            var image = new RenderTargetBitmap(1250, 860, 96, 96, PixelFormats.Pbgra32);
            image.Render(surface);
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(image));
            using var stream = File.Create(e.Args[1]);
            encoder.Save(stream);
            Shutdown();
            return;
        }
        window.Show();
        await viewModel.InitializeAsync();
    }
}
