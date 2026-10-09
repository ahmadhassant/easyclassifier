<#
Checks a portable EasyResearch Desktop folder on a clean Windows computer
(one that never had Python, Visual Studio or the .NET SDK installed).

Before running:
  1. Make the portable folder with publish.ps1 and copy it to the clean
     computer, ideally into a folder whose name has spaces and Arabic
     letters (for example a folder on the Desktop named in Arabic, with spaces).
  2. Disconnect the computer from the internet (the application must work
     offline).

Run (in Windows PowerShell):
  powershell -ExecutionPolicy Bypass -File verify_portable.ps1 -AppFolder "<the portable folder>"

Every check prints PASS or FAIL with a short reason, and a report is saved
as verify-report.txt in the work folder (in TEMP, inside a folder whose name
has a space and Arabic letters). Add -SkipUi to skip the slower
window checks (about 10 minutes for the four tasks).
#>
param(
    [Parameter(Mandatory = $true)][string]$AppFolder,
    [string]$WorkFolder = '',
    [switch]$SkipUi
)
$ErrorActionPreference = 'Stop'
$AppFolder = [IO.Path]::GetFullPath($AppFolder)
# The words for 'test data' in Arabic, built from character codes so this file reads correctly in any code page.
$arabic = -join ([char[]](0x0628, 0x064A, 0x0627, 0x0646, 0x0627, 0x062A, 0x20, 0x062A, 0x062C, 0x0631, 0x0628, 0x0629))
if (-not $WorkFolder) { $WorkFolder = Join-Path $env:TEMP "EasyResearch verify $arabic" }
New-Item -ItemType Directory -Path $WorkFolder -Force | Out-Null
$script:results = New-Object System.Collections.Generic.List[object]

function Check([string]$Name, [scriptblock]$Test) {
    try {
        $detail = & $Test
        $script:results.Add([pscustomobject]@{ Check = $Name; Result = 'PASS'; Detail = "$detail" })
        Write-Host "PASS  $Name  $detail" -ForegroundColor Green
    } catch {
        $script:results.Add([pscustomobject]@{ Check = $Name; Result = 'FAIL'; Detail = $_.Exception.Message })
        Write-Host "FAIL  $Name  $($_.Exception.Message)" -ForegroundColor Red
    }
}

function Info([string]$Name, [string]$Detail) {
    $script:results.Add([pscustomobject]@{ Check = $Name; Result = 'INFO'; Detail = $Detail })
    Write-Host "INFO  $Name  $Detail" -ForegroundColor Yellow
}

$python = Join-Path $AppFolder 'runtime\python\python.exe'
$worker = Join-Path $AppFolder 'worker\worker.py'
$modules = Join-Path $AppFolder 'modules'
$examples = Join-Path $AppFolder 'examples'

# Sends one request to the analysis engine exactly as the application does (UTF-8 both ways).
function Invoke-Worker([hashtable]$Request) {
    $Request['protocol_version'] = 1
    $Request['request_id'] = [guid]::NewGuid().ToString('N')
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $python
    $start.Arguments = "`"$worker`" --manifest-dir `"$modules`""
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardInput = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.StandardOutputEncoding = [Text.Encoding]::UTF8
    $start.StandardErrorEncoding = [Text.Encoding]::UTF8
    $start.WorkingDirectory = Split-Path $worker
    $start.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
    $start.EnvironmentVariables['PYTHONPATH'] = ''
    $process = [Diagnostics.Process]::Start($start)
    $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes(($Request | ConvertTo-Json -Depth 6 -Compress) + "`n")
    $process.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
    $process.StandardInput.Close()
    $errors = $process.StandardError.ReadToEndAsync()
    $output = $process.StandardOutput.ReadToEnd()
    $process.WaitForExit()
    $last = ($output -split "`n" | Where-Object { $_.Trim() }) | Select-Object -Last 1
    if (-not $last) { throw "The engine gave no answer. $($errors.Result)" }
    $answer = $last | ConvertFrom-Json
    if ($answer.type -ne 'completed') { throw "$($answer.message)" }
    return $answer.result
}

Write-Host "Checking $AppFolder`nWork folder: $WorkFolder`n"

if ($AppFolder -match ' ' -and $AppFolder -match '[^\u0000-\u007F]') {
    Info 'Application path' 'The application folder has spaces and non-English letters, as intended for this test.'
} else {
    Info 'Application path' 'For a complete test, copy the application into a folder whose name has spaces and Arabic letters, and run again.'
}

Check 'Application files' {
    foreach ($path in @((Join-Path $AppFolder 'EasyResearchDesktop.exe'), $python, $worker)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "missing: $path" }
    }
    $m = @(Get-ChildItem -LiteralPath $modules -Filter *.json).Count
    $e = @(Get-ChildItem -LiteralPath $examples -Filter *.csv).Count
    if ($m -lt 4 -or $e -lt 4) { throw "found $m module files and $e example files; expected 4 of each" }
    "$m task modules, $e example files"
}

Check 'Bundled Python and packages' {
    $out = & $python -c "import sys, easyresearch, easyclassifier, sklearn, statsmodels, torch; print(sys.version.split()[0], 'EasyResearch', easyresearch.__version__, 'PyTorch', torch.__version__)" 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "the bundled Python could not load its packages. If the message mentions a DLL (for example c10.dll or WinError 126), the Microsoft Visual C++ runtime is missing. Details: $out"
    }
    $out.Trim()
}

Check 'All four tasks load' {
    $catalog = Invoke-Worker @{ action = 'catalog' }
    $broken = @($catalog.modules | Where-Object { $_.unavailable_reason })
    if ($broken.Count) { throw (($broken | ForEach-Object { "$($_.title): $($_.unavailable_reason)" }) -join '; ') }
    ($catalog.modules | ForEach-Object { $_.title }) -join ', '
}

$runs = @(
    @{ module = 'classification'; file = 'classification_iris.csv'; target = 'Species'; models = @('decision_tree', 'logistic_regression'); extra = @{} },
    @{ module = 'regression'; file = 'regression_diabetes.csv'; target = 'Progression'; models = @('linear_regression', 'ridge'); extra = @{} },
    @{ module = 'forecasting'; file = 'forecasting_co2_monthly.csv'; target = 'CO2_ppm'; models = @('ridge', 'theta', 'mlp'); extra = @{ horizon = '6' } },
    @{ module = 'signals'; file = 'signals_heartbeats_synthetic.csv'; target = 'label'; models = @('knn', 'rocket'); extra = @{} }
)
$dataFolder = Join-Path $WorkFolder "data $arabic"
New-Item -ItemType Directory -Path $dataFolder -Force | Out-Null
foreach ($run in $runs) {
    Check "Run $($run.module) (path with spaces and Arabic letters)" {
        $data = Join-Path $dataFolder ($arabic + ' ' + $run.file)
        Copy-Item -LiteralPath (Join-Path $examples $run.file) -Destination $data -Force
        $inspection = Invoke-Worker @{ action = 'inspect'; module_id = $run.module; dataset = $data }
        if ($inspection.suggested_target -ne $run.target) { throw "suggested '$($inspection.suggested_target)' instead of '$($run.target)'" }
        $settings = @{ models = $run.models; figures = $true }
        foreach ($k in $run.extra.Keys) { $settings[$k] = $run.extra[$k] }
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $result = Invoke-Worker @{ action = 'run'; module_id = $run.module; dataset = $data; target = $run.target;
                                   output_root = (Join-Path $WorkFolder "results $arabic"); settings = $settings }
        $missing = @('report.tex', 'summary.csv', 'results.xlsx', 'trained_model.pkl', 'settings.json', 'versions.txt', 'log.txt') |
            Where-Object { -not (Test-Path -LiteralPath (Join-Path $result.output_dir $_)) }
        if ($missing) { throw "missing results: $($missing -join ', ')" }
        $pdf = if (Test-Path -LiteralPath (Join-Path $result.output_dir 'report.pdf')) { 'PDF report' } else { 'LaTeX report (no LaTeX installed, as expected on a clean computer)' }
        "$($result.selected_model); $pdf; $([int]$watch.Elapsed.TotalSeconds) s"
    }
}

try {
    $null = Invoke-WebRequest -Uri 'https://pypi.org' -UseBasicParsing -TimeoutSec 5
    Info 'Offline' 'The internet was reachable during this test. Disconnect it and run again to confirm offline use.'
} catch {
    Check 'Offline' { 'no internet connection during the test; everything above worked offline' }
}

if (-not $SkipUi) {
    foreach ($module in 'classification', 'regression', 'forecasting', 'signals') {
        Check "Window: $module example runs and shows results" {
            $png = Join-Path $WorkFolder "window-$module.png"
            if (Test-Path -LiteralPath $png) { Remove-Item -LiteralPath $png }
            $p = Start-Process -FilePath (Join-Path $AppFolder 'EasyResearchDesktop.exe') -ArgumentList @('--verify-ui', "`"$png`"", $module) -PassThru -Wait
            if (-not (Test-Path -LiteralPath $png)) { throw "no screenshot was produced (exit code $($p.ExitCode))" }
            "screenshot: $png"
        }
    }
}

$report = Join-Path $WorkFolder 'verify-report.txt'
$header = "EasyResearch Desktop portable check, $(Get-Date -Format 'yyyy-MM-dd HH:mm')`nApplication: $AppFolder`nWindows: $([Environment]::OSVersion.VersionString)`n"
$header + ($script:results | Format-Table -AutoSize -Wrap | Out-String -Width 200) | Set-Content -LiteralPath $report -Encoding UTF8
$failed = @($script:results | Where-Object { $_.Result -eq 'FAIL' }).Count
Write-Host "`n$failed check(s) failed. Report: $report"
if ($failed) { exit 1 }
