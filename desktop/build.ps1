param([string]$Dotnet = 'dotnet', [string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Push-Location $projectRoot
try {
    & $Dotnet build EasyResearchDesktop.sln -c Release --disable-build-servers
    if ($LASTEXITCODE -ne 0) { throw 'The C# build failed.' }
    $env:EASYRESEARCH_PYTHON = (Get-Command $Python).Source
    # Windows PowerShell treats a native program's error output as a failure when
    # it is captured, so the check runs with 'Continue' and reports the real reason.
    $ErrorActionPreference = 'Continue'
    $importCheck = & $Python -c "import easyresearch.regression, easyresearch.timeseries" 2>&1 | Out-String
    $importOk = $LASTEXITCODE -eq 0
    $ErrorActionPreference = 'Stop'
    if (-not $importOk) { throw "EasyResearch cannot be imported by $($env:EASYRESEARCH_PYTHON):`n$importCheck`nInstall it for this Python with: `"$($env:EASYRESEARCH_PYTHON)`" -m pip install -e ..\easyresearch" }
    # Development builds (Release, and Debug for Visual Studio) remember this Python,
    # so the application uses it however it is started.
    foreach ($configuration in 'Release', 'Debug') {
        $buildFolder = Join-Path $projectRoot "src\EasyResearch.Desktop\bin\$configuration\net10.0-windows"
        New-Item -ItemType Directory -Path $buildFolder -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $buildFolder 'python-path.txt') -Value $env:EASYRESEARCH_PYTHON -Encoding utf8
    }
    Write-Host "The application will use $($env:EASYRESEARCH_PYTHON) (recorded in python-path.txt)."
    & $Python -m unittest discover -s tests/python -v
    if ($LASTEXITCODE -ne 0) { throw 'The Python adapter tests failed.' }
    & $Dotnet run --project tests/EasyResearch.IntegrationTests -c Release --no-build -- $env:EASYRESEARCH_PYTHON $projectRoot (Join-Path $projectRoot 'test-results')
    if ($LASTEXITCODE -ne 0) { throw 'The C# integration checks failed.' }
} finally { Pop-Location }
