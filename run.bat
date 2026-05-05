@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Lancement du pipeline invoice-processor...
echo (Appuyez sur Ctrl+C pour interrompre, les resultats sont sauves au fur et a mesure)
python main.py
pause
