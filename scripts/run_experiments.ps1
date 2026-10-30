# scripts/run_experiments.ps1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [object[]] $Experiments,
    [string] $Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$stages = @("parse", "dedup", "split", "grid", "select", "audit", "calibrate", "bundle", "dev_eval")
$seenTags = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
$pairs = @()

# Each entry is a hashtable/object with ConfigPath and ResultsDir properties.
# Validate the complete plan before executing its first stage.
foreach ($experiment in $Experiments) {
    $configValue = [string] $experiment.ConfigPath
    $resultsValue = [string] $experiment.ResultsDir
    if ([string]::IsNullOrWhiteSpace($configValue) -or [string]::IsNullOrWhiteSpace($resultsValue)) {
        throw "Each experiment must specify non-empty ConfigPath and ResultsDir values."
    }
    $configPath = if ([System.IO.Path]::IsPathRooted($configValue)) {
        $configValue
    } else {
        Join-Path $projectRoot $configValue
    }
    $resultsPath = if ([System.IO.Path]::IsPathRooted($resultsValue)) {
        $resultsValue
    } else {
        Join-Path $projectRoot $resultsValue
    }
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Config file does not exist: $configPath"
    }
    $resultsPath = [System.IO.Path]::GetFullPath($resultsPath).TrimEnd('\', '/')
    $runTag = Split-Path -Leaf $resultsPath
    if ([string]::IsNullOrWhiteSpace($runTag) -or -not $seenTags.Add($runTag)) {
        throw "Every experiment must use a distinct, non-empty results-directory basename: $runTag"
    }
    $pairs += [PSCustomObject] @{ ConfigPath = $configPath; ResultsDir = $resultsPath }
}
if ($pairs.Count -eq 0) {
    throw "Provide at least one experiment pair."
}

Push-Location $projectRoot
try {
    foreach ($pair in $pairs) {
        foreach ($stage in $stages) {
            Write-Host "[$($pair.ResultsDir)] Stage: $stage"
            & $Python "scripts/run_pipeline.py" --config $pair.ConfigPath --results-dir $pair.ResultsDir --stage $stage
            if ($LASTEXITCODE -ne 0) {
                throw "Stage '$stage' failed with exit code $LASTEXITCODE. No subsequent stages or experiments ran."
            }
        }
    }
} finally {
    Pop-Location
}
