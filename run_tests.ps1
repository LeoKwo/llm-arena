# Run the full test suite (Python + front-end).
$ErrorActionPreference = "Stop"

$py = "python"
if (Test-Path ".\myenv\Scripts\python.exe") {
  $py = ".\myenv\Scripts\python.exe"
} elseif (Test-Path ".\myenv\bin\python") {
  $py = ".\myenv\bin\python"
}

Write-Host "== Python tests ==" -ForegroundColor Cyan
& $py -m pytest -q

Write-Host "== Front-end tests ==" -ForegroundColor Cyan
node --test web/tests
