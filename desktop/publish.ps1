param(
    [string]$Dotnet = 'dotnet',
    [string]$Python = 'python',
    [string]$Destination = (Join-Path $PSScriptRoot 'release\EasyResearchDesktop'),
    [string]$RuntimeArchive = ''
)
$ErrorActionPreference = 'Stop'
$Destination = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
& $Dotnet publish (Join-Path $PSScriptRoot 'src\EasyResearch.Desktop\EasyResearch.Desktop.csproj') -c Release -r win-x64 --self-contained true --disable-build-servers -o $Destination
if ($LASTEXITCODE -ne 0) { throw 'Desktop publishing failed.' }
$runtimeFolder = Join-Path $Destination 'runtime\python'
New-Item -ItemType Directory -Path $runtimeFolder -Force | Out-Null
if (-not $RuntimeArchive) {
    $RuntimeArchive = Join-Path $PSScriptRoot 'release\python-3.12.8-embed-amd64.zip'
    Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip' -OutFile $RuntimeArchive
}
Expand-Archive -LiteralPath $RuntimeArchive -DestinationPath $runtimeFolder -Force
@('python312.zip', '.', 'Lib/site-packages', 'import site') | Set-Content -LiteralPath (Join-Path $runtimeFolder 'python312._pth') -Encoding ascii
& $Python -m pip install --only-binary=:all: --target (Join-Path $runtimeFolder 'Lib\site-packages') -r (Join-Path $PSScriptRoot 'requirements.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Bundled analysis dependencies could not be installed.' }
# The analysis cores come from the EasyResearch source next to this folder,
# so the desktop runs exactly the code the command line runs. Their
# dependencies, including statsmodels and the CPU build of PyTorch used by
# forecasting, are pinned in requirements.lock.txt.
$wheelFolder = Join-Path $PSScriptRoot 'release\wheels'
New-Item -ItemType Directory -Path $wheelFolder -Force | Out-Null
# EasyClassifier is also bundled from the source in this repository (replacing the PyPI copy from the
# lock file), so the portable application always matches the code that was tested.
foreach ($package in @(@{ Name = 'easyresearch'; Source = '..\easyresearch' }, @{ Name = 'easyclassifier'; Source = '..' })) {
    & $Python -m pip wheel --no-deps -w $wheelFolder (Join-Path $PSScriptRoot $package.Source)
    if ($LASTEXITCODE -ne 0) { throw "The $($package.Name) package could not be built." }
    $wheel = Get-ChildItem -LiteralPath $wheelFolder -Filter "$($package.Name)-*.whl" | Sort-Object LastWriteTime | Select-Object -Last 1
    & $Python -m pip install --no-deps --upgrade --force-reinstall --target (Join-Path $runtimeFolder 'Lib\site-packages') $wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "The $($package.Name) package could not be bundled." }
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'PORTABLE-README.txt') -Destination $Destination -Force
Write-Host "Portable application: $Destination\EasyResearchDesktop.exe"
