[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Invoke-SystemPython([string[]]$Arguments) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python @Arguments
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 @Arguments
    } else {
        throw "未找到 Python。请先安装 Python 3.9 或更高版本，并勾选 Add Python to PATH。"
    }
    if ($LASTEXITCODE -ne 0) { throw "Python 命令执行失败，退出码：$LASTEXITCODE" }
}

Set-Location $ProjectRoot
if (-not (Test-Path $VenvPython)) {
    Write-Host "[1/4] 创建虚拟环境 .venv ..."
    Invoke-SystemPython @("-m", "venv", $VenvDir)
}

Write-Host "[2/4] 安装依赖 ..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip 升级失败" }
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }

$EnvFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $EnvFile)) {
    Copy-Item (Join-Path $ProjectRoot ".env.example") $EnvFile
    Write-Host "[3/4] 已生成 .env。需要 AI 问答时，请填写 DASHSCOPE_API_KEY。"
} else {
    Write-Host "[3/4] 保留现有 .env。"
}

Write-Host "[4/4] 校验项目数据 ..."
& $VenvPython (Join-Path $ProjectRoot "scripts\verify_project.py")
if ($LASTEXITCODE -ne 0) { throw "项目校验失败" }
Write-Host "安装完成。运行 start.bat 启动 GraphRAG。" -ForegroundColor Green