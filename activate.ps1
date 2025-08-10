# PowerShell script to activate the virtual environment
Write-Host "Activating Clash Royale Troop Annotation Tool virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Virtual environment activated! You can now run:" -ForegroundColor Yellow
Write-Host "  python main.py" -ForegroundColor Cyan
Write-Host "  python test_basic.py" -ForegroundColor Cyan
Write-Host "" 