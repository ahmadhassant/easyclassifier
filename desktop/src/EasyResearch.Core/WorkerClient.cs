using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace EasyResearch.Core;

public sealed record WorkerPaths(string Python, string Script, string Manifests, string Cache)
{
    /// <summary>Where the Python interpreter came from, for the activity log.</summary>
    public string Source { get; init; } = "";

    /// <summary>
    /// The Python used for analyses, in this order: the portable application's own runtime;
    /// python-path.txt written by build.ps1 next to a development build (the Python its tests passed with);
    /// the EASYRESEARCH_PYTHON environment variable.
    /// </summary>
    public static WorkerPaths FromApplication(string folder)
    {
        var bundled = Path.Combine(folder, "runtime", "python", "python.exe");
        var recordedFile = Path.Combine(folder, "python-path.txt");
        var recorded = File.Exists(recordedFile) ? File.ReadAllText(recordedFile).Trim().Trim('"') : "";
        var configured = Environment.GetEnvironmentVariable("EASYRESEARCH_PYTHON") ?? "";
        var (python, source) = File.Exists(bundled) ? (bundled, "the application's own Python")
            : recorded.Length > 0 ? (recorded, "python-path.txt written by build.ps1")
            : (configured, "the EASYRESEARCH_PYTHON setting");
        if (string.IsNullOrWhiteSpace(python))
            throw new WorkerException("The analysis engine is missing. Use the complete portable application folder, or run desktop\\build.ps1 (development builds remember the Python they were tested with).");
        if (!File.Exists(python))
            throw new WorkerException($"The analysis engine was not found at {python} (from {source}). Run desktop\\build.ps1 again with the Python you want to use.");
        var script = Path.Combine(folder, "worker", "worker.py");
        var manifests = Path.Combine(folder, "modules");
        if (!File.Exists(script) || !Directory.Exists(manifests))
            throw new WorkerException("The application is incomplete: its analysis modules are missing.");
        var cache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "EasyResearchDesktop", "plot-cache");
        return new(python, script, manifests, cache) { Source = source };
    }
}

public interface IWorkerClient
{
    Task<T> SendAsync<T>(WorkerRequest request, IProgress<WorkerProgress>? progress = null, CancellationToken cancellation = default);
}

public sealed class WorkerClient(WorkerPaths paths) : IWorkerClient
{
    public async Task<T> SendAsync<T>(WorkerRequest request, IProgress<WorkerProgress>? progress = null, CancellationToken cancellation = default)
    {
        cancellation.ThrowIfCancellationRequested();
        Directory.CreateDirectory(paths.Cache);
        var start = new ProcessStartInfo(paths.Python)
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
            StandardInputEncoding = new UTF8Encoding(false),
            WorkingDirectory = Path.GetDirectoryName(paths.Script)!
        };
        start.ArgumentList.Add(paths.Script);
        start.ArgumentList.Add("--manifest-dir");
        start.ArgumentList.Add(paths.Manifests);
        start.Environment["PYTHONIOENCODING"] = "utf-8";
        start.Environment["MPLBACKEND"] = "Agg";
        start.Environment["MPLCONFIGDIR"] = paths.Cache;
        start.Environment["PYTHONPATH"] = "";
        start.Environment["OMP_NUM_THREADS"] = "2";
        start.Environment["OPENBLAS_NUM_THREADS"] = "2";
        start.Environment["MKL_NUM_THREADS"] = "2";
        using var process = new Process { StartInfo = start };
        try { process.Start(); }
        catch (Exception ex) { throw new WorkerException("The analysis engine could not start. " + ex.Message); }
        using var registration = cancellation.Register(() => Kill(process));
        var stderr = process.StandardError.ReadToEndAsync();
        try
        {
            await process.StandardInput.WriteLineAsync(JsonSerializer.Serialize(request, Wire.Json));
            process.StandardInput.Close();
            T? completed = default;
            WorkerException? failure = null;
            bool terminal = false;
            while (await process.StandardOutput.ReadLineAsync(cancellation) is { } line)
            {
                cancellation.ThrowIfCancellationRequested();
                if (string.IsNullOrWhiteSpace(line)) continue;
                using var doc = JsonDocument.Parse(line);
                var e = doc.RootElement;
                if (e.GetProperty("protocol_version").GetInt32() != Wire.Version || e.GetProperty("request_id").GetString() != request.RequestId)
                    throw new WorkerException("The analysis engine returned an incompatible response.");
                if (terminal) throw new WorkerException("The analysis engine returned data after its final response.");
                switch (e.GetProperty("type").GetString())
                {
                    case "progress":
                        progress?.Report(new(e.TryGetProperty("message", out var text) ? text.GetString() ?? "" : "",
                                             e.TryGetProperty("session_dir", out var session) ? session.GetString() : null,
                                             e.TryGetProperty("fraction", out var share) && share.ValueKind == JsonValueKind.Number ? share.GetDouble() : null));
                        break;
                    case "completed":
                        completed = e.GetProperty("result").Deserialize<T>(Wire.Json);
                        terminal = true;
                        break;
                    case "error":
                        failure = new(e.GetProperty("message").GetString() ?? "Analysis failed.", e.TryGetProperty("detail", out var detail) ? detail.GetString() : null);
                        terminal = true;
                        break;
                    default: throw new WorkerException("The analysis engine returned an unknown event.");
                }
            }
            await process.WaitForExitAsync(cancellation);
            cancellation.ThrowIfCancellationRequested();
            var errors = await stderr;
            if (failure is not null) throw failure;
            if (process.ExitCode != 0 || !terminal || completed is null)
                throw new WorkerException("The analysis engine stopped before completing the request.", errors);
            return completed;
        }
        catch (Exception) when (cancellation.IsCancellationRequested)
        {
            Kill(process);
            await process.WaitForExitAsync(CancellationToken.None);
            throw new OperationCanceledException(cancellation);
        }
        catch (JsonException ex) { Kill(process); throw new WorkerException("The analysis engine returned an unreadable response.", ex.Message); }
        catch { Kill(process); throw; }
    }

    private static void Kill(Process process)
    {
        try { if (!process.HasExited) process.Kill(entireProcessTree: true); }
        catch (InvalidOperationException) { }
        catch (System.ComponentModel.Win32Exception) { }
    }
}
