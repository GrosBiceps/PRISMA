@echo off
REM ============================================================
REM  build.bat — Compile FlowSOM Analyzer Pro en .exe portable
REM  Usage : double-clic ou appel depuis un terminal
REM ============================================================

cd /d "%~dp0"

echo.
echo ========================================================
echo  FlowSOM Analyzer Pro — Build PyInstaller
echo ========================================================
echo.

REM Nettoyage manuel des anciens artefacts
REM (on NE passe PAS --clean a PyInstaller : bug Windows FileNotFoundError)
echo [1/3] Nettoyage...
if exist dist\FlowSOMAnalyzer rmdir /s /q dist\FlowSOMAnalyzer
if exist build\flowsom_gui rmdir /s /q build\flowsom_gui

REM Recrée le dossier build AVANT d'appeler PyInstaller (requis sur Windows)
mkdir build\flowsom_gui
echo       OK.
echo.

REM Lancement du build (log dans build_log.txt)
echo [2/3] Compilation en cours (peut prendre 10-20 min)...
echo       Les logs sont ecrits dans build_log.txt
python -m PyInstaller flowsom_gui.spec -y > build_log.txt 2>&1

REM Vérification du résultat
echo.
if exist dist\FlowSOMAnalyzer\FlowSOMAnalyzer.exe (
    echo [3/3] BUILD REUSSI !
    echo.
    echo  Executable : dist\FlowSOMAnalyzer\FlowSOMAnalyzer.exe
    echo.
    echo  Pour distribuer : zippez le dossier dist\FlowSOMAnalyzer\ entier.
) else (
    echo [3/3] ERREUR : l'executable n'a pas ete cree.
    echo  Consultez build_log.txt pour les details.
    echo.
    echo --- Dernières lignes du log ---
    powershell -command "Get-Content build_log.txt -Tail 30"
)

echo.
pause
