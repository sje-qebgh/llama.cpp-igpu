# mtl-prof.ps1 - per-op profiles (Meteor Lake or Lunar Lake; -SkipLong for part A only) for
#   A) token generation bandwidth per quant type and matrix shape (tg at depth 0, several models)
#   B) what grows with context: tg and prefill at depth 0 and 16384, flash attention on (q8_0 cache) and off (f16 cache)
#
# Uses GGML_VK_PERF_LOGGER=1 (per-op GPU times; the t/s printed in these runs is lower than normal, ignore it).
# Plug in the AC adapter and use the best performance power mode. The script does a warm-up run first.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\mtl-prof.ps1
# Send back the whole output folder as a zip.

param(
    [string]$ForkDir  = "D:\a\igpu-latest",
    [string]$ModelDir = "D:\LLM\models",
    [string]$Main     = "Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
    [string[]]$TgModels = @("Qwen3.6-35B-A3B-UD-IQ3_S.gguf", "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
                            "Qwen3-Coder-30B-A3B-Instruct-UD-Q2_K_XL.gguf",
                            "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf", "gpt-oss-20b-UD-Q4_K_XL.gguf"),
    [switch]$SkipLong
)

$ErrorActionPreference = "Continue"
$Stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$OutDir = Join-Path (Get-Location) "prof-$Stamp"
New-Item -ItemType Directory -Path $OutDir | Out-Null
$Info = Join-Path $OutDir "info.md"
function Log([string]$s) { Write-Host $s; Add-Content -Path $Info -Value $s -Encoding UTF8 }
function Clear-VkEnv { Get-ChildItem Env: | Where-Object { $_.Name -like "GGML_VK_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" } }
function Find-Bin([string]$dir, [string]$exe) {
    foreach ($d in @("", "build\bin\Release", "Build\bin\Release", "build\bin")) { $p = Join-Path $dir (Join-Path $d $exe); if (Test-Path $p) { return $p } }
    return $null
}

Clear-VkEnv
$Bench = Find-Bin $ForkDir "llama-bench.exe"
if (-not $Bench) { Write-Host "llama-bench.exe not found under $ForkDir"; exit 1 }
$mainPath = Join-Path $ModelDir $Main
if (-not (Test-Path $mainPath)) { Write-Host "$mainPath not found"; exit 1 }

Log "# Meteor Lake profiles $Stamp"
$gpu = (Get-CimInstance Win32_VideoController | Select-Object -First 1)
Log "- GPU: $($gpu.Name), driver $($gpu.DriverVersion)"
Log "- power plan: $((powercfg /getactivescheme) -replace '.*\((.*)\).*','$1'), on AC: $((Get-CimInstance -ClassName BatteryStatus -Namespace root/wmi -ErrorAction SilentlyContinue | Select-Object -First 1).PowerOnline)"
Log "- build: $((& $Bench --version 2>&1 | Out-String) -replace '(?s).*(version: [^\r\n]*).*', '$1')"
Log ""

# warm-up (no profiling), also gives a normal tg/pp number for reference
Log "## Warm-up / reference (no profiler)"
for ($i = 0; $i -lt 2; $i++) {
    $out = & $Bench -m $mainPath -ngl 99 -p 512 -n 128 -r 1 2>&1 | Out-String
    foreach ($m in [regex]::Matches($out, '\|\s+(pp512|tg128)\s+\|\s+([\d\.]+)')) { Log "- run $($i+1) $($m.Groups[1].Value): $($m.Groups[2].Value) t/s" }
}
Log ""

function Prof([string]$name, [string]$model, [string[]]$extra) {
    Clear-VkEnv
    $env:GGML_VK_PERF_LOGGER = "1"
    $args_ = @("-m", $model, "-ngl", "99", "-r", "1") + $extra
    $out = & $Bench @args_ 2>&1 | Out-String
    Clear-VkEnv
    $file = Join-Path $OutDir "$name.txt"
    ("args: " + ($extra -join " ") + "`r`n" + $out) | Set-Content -Encoding UTF8 $file
    Write-Host "  wrote $name.txt"
}

# A) token generation per model, depth 0
Log "## A) tg profiles at depth 0"
foreach ($n in $TgModels) {
    $p = Join-Path $ModelDir $n
    if (-not (Test-Path $p)) { Log "- $n not found, skipped"; continue }
    Prof ("tg-d0-" + ($n -replace '\.gguf$', '' -replace '[^\w\-\.]', '_')) $p @("-p", "0", "-n", "64")
    Log "- $n"
}
Log ""

# B) context growth, main model
if ($SkipLong) { Log "Done. Output folder: $OutDir"; exit 0 }
Log "## B) $Main at depth 0 / 16384"
$fa1 = @("-fa", "1", "-ctk", "q8_0", "-ctv", "q8_0")
$fa0 = @("-fa", "0")
Prof "pp-d16384-fa1q8" $mainPath (@("-p", "512", "-n", "0", "-d", "16384") + $fa1)
Prof "pp-d16384-fa0f16" $mainPath (@("-p", "512", "-n", "0", "-d", "16384") + $fa0)
Prof "tg-d16384-fa1q8" $mainPath (@("-p", "0", "-n", "32", "-d", "16384") + $fa1)
Prof "tg-d16384-fa0f16" $mainPath (@("-p", "0", "-n", "32", "-d", "16384") + $fa0)
Log "- pp512 and tg32 at depth 16384, -fa 1 q8_0/q8_0 and -fa 0 f16"
Log ""
Log "Done. Output folder: $OutDir"
