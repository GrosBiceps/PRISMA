@echo off
REM ============================================================
REM  build.bat — Compile PRISMA en .exe portable
REM  Usage : double-clic ou appel depuis un terminal
REM ============================================================

cd /d "%~dp0"

echo.
echo ========================================================
echo  PRISMA — Build PyInstaller
echo ========================================================
echo.

REM Nettoyage manuel des anciens artefacts
REM (on NE passe PAS --clean a PyInstaller : bug Windows FileNotFoundError)
echo [1/3] Nettoyage...
if exist dist\PRISMA rmdir /s /q dist\PRISMA
if exist build\flowsom_gui rmdir /s /q build\flowsom_gui

REM Recrée le dossier build AVANT d'appeler PyInstaller (requis sur Windows)
mkdir build\flowsom_gui
echo       OK.
echo.

REM Lancement du build (log dans build_log.txt)
echo [2/3] Compilation en cours (peut prendre 10-20 min)...
echo       Les logs sont ecrits dans build_log.txt

REM Horodatage de début
for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
    set /a _START_S = %%a*3600 + %%b*60 + %%c
    set _TIME_START=%%a:%%b:%%c
)

python -m PyInstaller flowsom_gui.spec -y > build_log.txt 2>&1

REM Horodatage de fin + calcul de la durée
for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
    set /a _END_S = %%a*3600 + %%b*60 + %%c
    set _TIME_END=%%a:%%b:%%c
)
set /a _ELAPSED = _END_S - _START_S
set /a _MIN = _ELAPSED / 60
set /a _SEC = _ELAPSED %% 60

REM Vérification du résultat
echo.
if exist dist\PRISMA\PRISMA.exe (
    echo [3/3] BUILD REUSSI !
    echo.
    echo  Executable : dist\PRISMA\PRISMA.exe
    echo.
    echo  Debut    : %_TIME_START%
    echo  Fin      : %_TIME_END%
    echo  Duree    : %_MIN% min %_SEC% sec
    echo.
    echo  Pour distribuer : zippez le dossier dist\PRISMA\ entier.
) else (
    echo [3/3] ERREUR : l'executable n'a pas ete cree.
    echo  Duree    : %_MIN% min %_SEC% sec
    echo  Consultez build_log.txt pour les details.
    echo.
    echo --- Dernières lignes du log ---
    powershell -command "Get-Content build_log.txt -Tail 30"
)

echo.
pause
