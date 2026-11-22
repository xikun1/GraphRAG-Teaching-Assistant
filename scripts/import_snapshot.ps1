[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$SourceGraph = Join-Path $SourceRoot "examples\example_output\example_mmkg.graphml"
$SourceWorking = Join-Path $SourceRoot "examples\example_working"
$SourceImageData = Join-Path $SourceWorking "kv_store_image_data.json"
$SourceChunks = Join-Path $SourceWorking "kv_store_text_chunks.json"
$SourceImages = Join-Path $SourceWorking "images"

foreach ($required in @($SourceGraph, $SourceImageData, $SourceChunks, $SourceImages)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "训练产物不存在：$required" }
}

$DataRoot = Join-Path $ProjectRoot "data"
$GraphDir = Join-Path $DataRoot "graph"
$MetadataDir = Join-Path $DataRoot "metadata"
$ImageDir = Join-Path $DataRoot "images"
New-Item -ItemType Directory -Force -Path $GraphDir, $MetadataDir, $ImageDir | Out-Null

$GraphTarget = Join-Path $GraphDir "example_mmkg.graphml"
$ImageDataTarget = Join-Path $MetadataDir "kv_store_image_data.json"
$ChunksTarget = Join-Path $MetadataDir "kv_store_text_chunks.json"
Copy-Item -LiteralPath $SourceGraph -Destination $GraphTarget -Force
Copy-Item -LiteralPath $SourceImageData -Destination $ImageDataTarget -Force
Copy-Item -LiteralPath $SourceChunks -Destination $ChunksTarget -Force

# Convert training-workspace image references into portable project-relative paths.
foreach ($target in @($GraphTarget, $ImageDataTarget)) {
    $text = [System.IO.File]::ReadAllText($target)
    $text = $text.Replace("examples/example_working\images\", "data/images/")
    $text = $text.Replace("examples/example_working/images/", "data/images/")
    [System.IO.File]::WriteAllText($target, $text, [System.Text.UTF8Encoding]::new($false))
}

$ResolvedImageDir = [System.IO.Path]::GetFullPath($ImageDir)
$ResolvedDataRoot = [System.IO.Path]::GetFullPath($DataRoot)
if (-not $ResolvedImageDir.StartsWith($ResolvedDataRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "图片目标目录不在项目 data 内，已拒绝清理。"
}
Get-ChildItem -LiteralPath $ImageDir -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|webp|gif)$' } |
    Remove-Item -Force
Copy-Item -Path (Join-Path $SourceImages '*') -Destination $ImageDir -Force

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
& $Python (Join-Path $ProjectRoot "scripts\verify_project.py")
if ($LASTEXITCODE -ne 0) { throw "快照已复制，但完整性校验失败。" }
Write-Host "训练快照已导入 GraphRAG；运行时不再依赖源工程。" -ForegroundColor Green