# bench-compare.ps1 - upstream llama.cpp vs llama.cpp-igpu, alternating runs (Windows, Vulkan)
#
# Same script for Meteor Lake and Lunar Lake. Missing models are skipped.
#   - every model: pp512 and tg128 at depth 0, llama-bench defaults
#   - $LongModels: pp512 and tg128 at depth 16384 with -fa 1 -ctk q8_0 -ctv q8_0
# Each round runs upstream, then the fork. The first run is a warm-up and is not counted.
# Plug in the AC adapter and keep the power mode the same for the whole run.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\bench-compare.ps1
#   powershell -ExecutionPolicy Bypass -File .\bench-compare.ps1 -Upstream D:\a\llama.cpp -Fork D:\a\igpu-latest
# Send back summary.md and results.csv from the output folder (or the whole folder as a zip).

param(
    [string]$Upstream = "D:\a\llama.cpp",
    [string]$Fork     = "D:\a\igpu-latest",
    [string]$ModelDir = "D:\LLM\models",
    [string[]]$Models = @("Qwen3.6-35B-A3B-UD-IQ3_S.gguf", "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
                          "Qwen3-Coder-30B-A3B-Instruct-UD-Q2_K_XL.gguf",
                          "gemma-4-26B-A4B-it-UD-IQ4_NL.gguf", "gpt-oss-20b-UD-Q4_K_XL.gguf"),
    [string[]]$LongModels = @("Qwen3.6-35B-A3B-UD-IQ3_S.gguf"),
    [int]$Rounds     = 3,
    [int]$LongRounds = 2,
    [int]$Depth      = 16384
)

$ErrorActionPreference = "Continue"
$Stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$OutDir = Join-Path (Get-Location) "bench-compare-$Stamp"
New-Item -ItemType Directory -Path $OutDir | Out-Null
$Summary = Join-Path $OutDir "summary.md"
$Csv     = Join-Path $OutDir "results.csv"
function Log([string]$s) { Write-Host $s; Add-Content -Path $Summary -Value $s -Encoding UTF8 }
function Clear-VkEnv { Get-ChildItem Env: | Where-Object { $_.Name -like "GGML_VK_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" } }
function Find-Bin([string]$dir, [string]$exe) {
    foreach ($d in @("", "build\bin\Release", "Build\bin\Release", "build\bin")) { $p = Join-Path $dir (Join-Path $d $exe); if (Test-Path $p) { return $p } }
    return $null
}
function Mean($v) { if ($v.Count) { ($v | Measure-Object -Average).Average } else { $null } }
function Fmt($v) { if ($v.Count) { "{0:N1} ({1:N1}-{2:N1})" -f (Mean $v), (($v | Measure-Object -Minimum).Minimum), (($v | Measure-Object -Maximum).Maximum) } else { "-" } }
function Power-Mode {
    $map = @{ "ded574b5-45a0-4f42-8737-46345c09c238" = "best performance"; "00000000-0000-0000-0000-000000000000" = "balanced";
              "961cc777-2547-4f9d-8174-7d86181b8a7a" = "best power efficiency"; "3af9b8d9-7c97-431d-ad78-34a8bfea439f" = "better battery" }
    try {
        $k = Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes" -ErrorAction Stop
        $ac = [string]$k.ActiveOverlayAcPowerScheme
        if ($map.ContainsKey($ac)) { return $map[$ac] } elseif ($ac) { return $ac }
    } catch {}
    return "unknown"
}

Clear-VkEnv
$Bin = [ordered]@{ "upstream" = (Find-Bin $Upstream "llama-bench.exe"); "fork" = (Find-Bin $Fork "llama-bench.exe") }
foreach ($k in $Bin.Keys) { if (-not $Bin[$k]) { Write-Host "llama-bench.exe not found for $k"; exit 1 } }

Log "# upstream vs llama.cpp-igpu $Stamp"
$gpu = (Get-CimInstance Win32_VideoController | Select-Object -First 1)
Log "- GPU: $($gpu.Name), driver $($gpu.DriverVersion)"
Log "- CPU: $((Get-CimInstance Win32_Processor | Select-Object -First 1).Name)"
Log "- power plan: $((powercfg /getactivescheme) -replace '.*\((.*)\).*','$1'), power mode (AC): $(Power-Mode), on AC: $((Get-CimInstance -ClassName BatteryStatus -Namespace root/wmi -ErrorAction SilentlyContinue | Select-Object -First 1).PowerOnline)"
foreach ($k in $Bin.Keys) {
    $v = (& $Bin[$k] --version 2>&1 | Out-String)
    $m = [regex]::Match($v, 'version: [^\r\n]*')
    Log "- $k : $($Bin[$k]) ($($m.Value))"
}
Log "- rounds: $Rounds at depth 0, $LongRounds at depth $Depth; each round runs upstream then the fork"
Log ""
"machine,model,test,build,round,tps" | Set-Content -Encoding UTF8 $Csv
$machine = $gpu.Name -replace ',', ' '

function Run([string]$build, [string]$model, [string[]]$extra, [string]$tag, [int]$round, [hashtable]$store) {
    Clear-VkEnv
    $args_ = @("-m", $model, "-ngl", "99", "-r", "1") + $extra
    $out = & $Bin[$build] @args_ 2>&1 | Out-String
    $out | Add-Content -Encoding UTF8 (Join-Path $OutDir ("raw-{0}-{1}-{2}.log" -f ((Split-Path $model -Leaf) -replace '\.gguf$', '' -replace '[^\w\-\.]', '_'), $tag, $build))
    foreach ($m in [regex]::Matches($out, '\|\s+((?:pp|tg)\d+(?: @ d\d+)?)\s+\|\s+([\d\.]+)')) {
        $test = $m.Groups[1].Value; $tps = [double]$m.Groups[2].Value
        if ($round -ge 0) {
            $key = "$test|$build"
            if (-not $store.ContainsKey($key)) { $store[$key] = @() }
            $store[$key] += $tps
            "$machine,$((Split-Path $model -Leaf)),$test,$build,$round,$tps" | Add-Content -Encoding UTF8 $Csv
        }
        Write-Host ("  {0} {1} {2}: {3}" -f (Split-Path $model -Leaf), $build, $test, $tps)
    }
}

$found = @()
foreach ($n in $Models) { $p = Join-Path $ModelDir $n; if (Test-Path $p) { $found += $n } else { Log "- $n not found, skipped" } }
if (-not $found.Count) { Log "No models found under $ModelDir"; exit 1 }

# warm-up, not counted
Write-Host "warm-up"
$dummy = @{}
foreach ($k in $Bin.Keys) { Run $k (Join-Path $ModelDir $found[0]) @("-p", "512", "-n", "64") "warmup" -1 $dummy }

Log "## Depth 0 (llama-bench defaults), t/s mean (min-max)"
Log ""
Log "| model | test | upstream | this fork | fork / upstream |"
Log "|---|---|---|---|---|"
foreach ($n in $found) {
    $p = Join-Path $ModelDir $n
    $st = @{}
    for ($r = 0; $r -lt $Rounds; $r++) { foreach ($k in $Bin.Keys) { Run $k $p @("-p", "512", "-n", "128") "d0" $r $st } }
    foreach ($t in @("pp512", "tg128")) {
        $u = $st["$t|upstream"]; $f = $st["$t|fork"]
        $ratio = if ((Mean $u) -and (Mean $f)) { "{0:N2}x" -f ((Mean $f) / (Mean $u)) } else { "-" }
        Log "| $n | $t | $(Fmt $u) | $(Fmt $f) | $ratio |"
    }
}
Log ""

$long = @(); foreach ($n in $LongModels) { if ($found -contains $n) { $long += $n } }
if ($long.Count) {
    Log "## Depth $Depth, -fa 1 -ctk q8_0 -ctv q8_0, t/s mean (min-max)"
    Log ""
    Log "| model | test | upstream | this fork | fork / upstream |"
    Log "|---|---|---|---|---|"
    foreach ($n in $long) {
        $p = Join-Path $ModelDir $n
        $st = @{}
        for ($r = 0; $r -lt $LongRounds; $r++) {
            foreach ($k in $Bin.Keys) { Run $k $p @("-p", "512", "-n", "128", "-d", "$Depth", "-fa", "1", "-ctk", "q8_0", "-ctv", "q8_0") "d$Depth" $r $st }
        }
        foreach ($t in @("pp512 @ d$Depth", "tg128 @ d$Depth")) {
            $u = $st["$t|upstream"]; $f = $st["$t|fork"]
            $ratio = if ((Mean $u) -and (Mean $f)) { "{0:N2}x" -f ((Mean $f) / (Mean $u)) } else { "-" }
            Log "| $n | $t | $(Fmt $u) | $(Fmt $f) | $ratio |"
        }
    }
    Log ""
}
Clear-VkEnv
Log "Done. Output folder: $OutDir"
