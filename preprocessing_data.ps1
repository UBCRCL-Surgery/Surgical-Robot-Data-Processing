param(
    [Parameter(Mandatory=$true)]
    [string]$config
)
Write-Host "Processing $config"
python ./sync_all.py --config $config
Write-Host "Synced $config"
python ./trim_left_video.py --config $config
Write-Host "Trimmed $config"