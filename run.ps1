# run.ps1 — мини-чат с RAG + источниками + памятью задачи (день 25).
#
#   .\run.ps1                             # проверка на 2 длинных сценариях (12 и 13 реплик)
#   .\run.ps1 -Mode chat                  # мини-чат; сессия сохраняется между запусками
#   .\run.ps1 -Mode chat -Session mcp     # отдельная сессия
#
# Ключ AITunnel спросит со скрытым вводом. Модель по умолчанию — gpt-5.6-luna.
# Если Windows блокирует запуск:  powershell -ExecutionPolicy Bypass -File .\run.ps1

param(
    [ValidateSet("scenarios", "chat", "calibrate")]
    [string]$Mode = "scenarios",
    [string]$Model = "gpt-5.6-luna",
    [ValidateSet("aitunnel", "local")]
    [string]$Backend = "aitunnel",
    [string]$Session = "default"
)

Set-Location $PSScriptRoot

function Find-Python {
    foreach ($candidate in @("py", "python",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe")) {
        try {
            $output = & $candidate --version 2>&1
            if ($output -match "Python \d") { return $candidate }
        }
        catch { }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host "Python не найден. Установите Python 3 с python.org или из Microsoft Store." -ForegroundColor Red
    exit 1
}
Write-Host "Использую Python: $py" -ForegroundColor Green

Write-Host "Устанавливаю зависимости..." -ForegroundColor Cyan
& $py -m pip install -r requirements.txt --quiet --disable-pip-version-check

if (-not ($Backend -eq "local" -and $Mode -eq "calibrate") -and -not $env:AITUNNEL_API_KEY) {
    $secureKey = Read-Host "Введите ваш AITUNNEL_API_KEY" -AsSecureString
    $env:AITUNNEL_API_KEY = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    )
}

Write-Host "`n=== Подбор порогов (фильтр и «не знаю») ===" -ForegroundColor Cyan
& $py .\calibrate.py --backend $Backend; if ($LASTEXITCODE -ne 0) { exit 1 }

switch ($Mode) {
    "calibrate" { }
    "scenarios" {
        Write-Host "`n=== Два длинных сценария ===" -ForegroundColor Cyan
        & $py .\run_scenarios.py --backend $Backend --model $Model
    }
    "chat" { & $py .\chat.py --backend $Backend --model $Model --session $Session }
}
