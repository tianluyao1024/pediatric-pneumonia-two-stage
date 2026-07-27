param(
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

# The raw Kaggle archive/extraction must be present under data/raw/.
& $Python -m src.stage1_compare_select_stable_v3 --project-root $ProjectRoot
& $Python -m src.stage2_compare_select_nextgen --project-root $ProjectRoot
& $Python -m src.stage2_ssl_pretrain_groupfold --project-root $ProjectRoot
& $Python -m src.stage2_oof_fixed_stack --project-root $ProjectRoot
& $Python -m src.evaluate_end_to_end_cascade --project-root $ProjectRoot
