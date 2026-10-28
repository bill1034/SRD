param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    & $Python -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        throw "SRD tests failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
