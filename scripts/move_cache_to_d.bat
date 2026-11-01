@echo off
rem ======================================================================
rem  One-time helper: move DATASETS this project already downloaded from
rem  C:\Users\<you>\.cache  to  D:\datasets\cache  (the folder that
rem  original_lfcbm\data_utils.py now uses).
rem
rem  MODEL WEIGHTS ARE NOT MOVED: CLIP (.cache\clip), ResNet18-CUB (.torch\models)
rem  and torchvision weights stay on C: on purpose.
rem
rem  Run from cmd.exe with NO training running:
rem    scripts\move_cache_to_d.bat preview     (lists what would move, changes nothing)
rem    scripts\move_cache_to_d.bat             (moves it)
rem
rem  Uses robocopy /MOVE: files are copied, verified, then removed from C:.
rem  Exit codes 0-7 mean success, 8 or more means something failed.
rem  NOTE: written without access to a Windows machine -- run "preview" first.
rem ======================================================================
setlocal
set "SRC=%USERPROFILE%\.cache"
set "DST=D:\datasets\cache"
set "LIST="
if /i "%~1"=="preview" set "LIST=/L"

echo Source: %SRC%
echo Target: %DST%
if defined LIST (echo MODE: PREVIEW - nothing will be moved) else (echo MODE: MOVE)
echo.
if not exist "%DST%" mkdir "%DST%"

rem --- CIFAR-10 / CIFAR-100 ---
if exist "%SRC%\cifar-10-batches-py" robocopy "%SRC%\cifar-10-batches-py" "%DST%\cifar-10-batches-py" /E /MOVE /R:1 /W:1 /NP %LIST%
if exist "%SRC%\cifar-100-python"    robocopy "%SRC%\cifar-100-python"    "%DST%\cifar-100-python"    /E /MOVE /R:1 /W:1 /NP %LIST%
robocopy "%SRC%" "%DST%" cifar-10-python.tar.gz cifar-100-python.tar.gz /MOV /R:1 /W:1 /NP %LIST%

rem --- Places365: extracted images, validation images, archives, list files ---
if exist "%SRC%\data_256_standard" robocopy "%SRC%\data_256_standard" "%DST%\data_256_standard" /E /MOVE /R:1 /W:1 /NP %LIST%
if exist "%SRC%\val_256"           robocopy "%SRC%\val_256"           "%DST%\val_256"           /E /MOVE /R:1 /W:1 /NP %LIST%
robocopy "%SRC%" "%DST%" *places365* categories_places365.txt val_256.tar /MOV /R:1 /W:1 /NP %LIST%

echo.
echo Done. Contents of %DST%:
dir /b "%DST%" 2>nul
endlocal
