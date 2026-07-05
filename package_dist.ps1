$dist = "C:\Users\user\Desktop\spectrum-dashboard\dist\SpectrumDashboard"
$env_src = "C:\Users\user\Desktop\spectrum-dashboard\.env"
$zip_out = "C:\Users\user\Desktop\SpectrumDashboard_설치본.zip"

# .env 복사
Copy-Item $env_src -Destination "$dist\.env" -Force
Write-Output ".env 복사 완료"

# zip 생성
if (Test-Path $zip_out) { Remove-Item $zip_out -Force }
Compress-Archive -Path "$dist\*" -DestinationPath $zip_out -Force
$size = [math]::Round((Get-Item $zip_out).Length / 1MB, 1)
Write-Output "zip 생성 완료: $zip_out ($size MB)"
