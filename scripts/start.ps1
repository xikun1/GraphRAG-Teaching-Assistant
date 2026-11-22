[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location $ProjectRoot

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Python = "python"
} else {
    throw "未找到可用 Python。请先运行 setup.bat。"
}

& $Python (Join-Path $ProjectRoot "scripts\verify_project.py")
if ($LASTEXITCODE -ne 0) { throw "项目数据校验失败，服务未启动。" }
& $Python (Join-Path $ProjectRoot "main.py")