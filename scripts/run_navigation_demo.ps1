param(
    [switch]$ConfirmStart
)

$ErrorActionPreference = "Stop"
if (-not $ConfirmStart) {
    throw "Use -ConfirmStart to authorize recorded-EEG-assisted autonomous travel in Webots."
}

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$world = Join-Path $root "simulation\worlds\hospital_rehab.wbt"
$telemetry = Join-Path $root "artifacts\runtime\wheelchair.json"
$webots = Join-Path $env:LOCALAPPDATA "Programs\Webots\msys64\mingw64\bin\webots.exe"

if (-not (Test-Path $python)) {
    throw "Project Python environment is missing: $python"
}
if (Get-NetTCPConnection -LocalPort 5555 -State Listen -ErrorAction SilentlyContinue) {
    throw "Port 5555 is in use. Stop the dashboard or other publisher first."
}

$started = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
if (-not (Get-Process -Name webots,webots-bin -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path $webots)) {
        throw "Webots was not found at $webots. Open the hospital world manually and rerun."
    }
    Start-Process -FilePath $webots -ArgumentList @($world) -WorkingDirectory $root | Out-Null
}

Write-Host "Waiting for Webots. Use real-time playback; reload the world to start at the dock."
$deadline = (Get-Date).AddSeconds(60)
$ready = $false
while ((Get-Date) -lt $deadline) {
    if (Test-Path $telemetry) {
        try {
            $state = Get-Content $telemetry -Raw | ConvertFrom-Json
        } catch [System.IO.IOException] {
            Start-Sleep -Milliseconds 200
            continue
        }
        $now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
        if ($state.engine -eq "Webots" -and $state.timestamp -gt $started -and
            ($now - $state.timestamp) -lt 0.75) {
            if ($state.emergency_stop) {
                throw "Emergency stop is latched. Reset it explicitly before starting."
            }
            $distanceFromDock = [Math]::Sqrt(
                [Math]::Pow($state.position[0] + 7, 2) + [Math]::Pow($state.position[1], 2))
            if ($distanceFromDock -gt 0.3) {
                throw "Chair is not at the dock. Reload the world in Webots, then rerun this launcher."
            }
            $ready = $true
            break
        }
    }
    Start-Sleep -Milliseconds 200
}
if (-not $ready) {
    throw "No fresh Webots telemetry. Check real-time playback and the Python controller console."
}

Write-Host "Recorded EEG requests the trip; autonomous planning handles the route."
Write-Host "Start screen recording now. Keep Webots and this terminal visible."
for ($seconds = 10; $seconds -gt 0; $seconds--) {
    Write-Host "EEG replay starts in $seconds seconds (Ctrl+C to cancel)."
    Start-Sleep -Seconds 1
}

Push-Location $root
try {
    & $python -m src.bci.navigation_demo --confirm-start
    if ($LASTEXITCODE -ne 0) {
        throw "Navigation stopped with an error. See the trip status above."
    }
} finally {
    Pop-Location
}
