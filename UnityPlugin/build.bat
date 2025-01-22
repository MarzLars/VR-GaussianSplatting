if "%1"=="--skip" (
    echo Skip Configuration
) else (
    rmdir /S /Q build
    cmake -G "Visual Studio 17" . -B build
)
cmake --build build --config Release -j 4

@echo off
set SOURCE=.\build\gaussiansplatting.dll
set DESTINATION=..\GaussianSplattingVRViewer\Assets\GaussianSplattingPlugin\Plugins\gaussiansplatting.dll

if exist "%SOURCE%" (
    echo Copying gaussiansplatting.dll...
    copy /Y "%SOURCE%" "%DESTINATION%"
    echo Copy Done
) else (
    echo Compilation Failed.
)