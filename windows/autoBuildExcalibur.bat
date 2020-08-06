set MSBuild="C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\MSBuild\15.0\Bin\MSBuild.exe"

%MSBuild% .\Excalibur.sln /t:Rebuild /p:Platform=x64 /p:Configuration=SharpC
%MSBuild% .\Excalibur.sln /t:Rebuild /p:Platform=x64 /p:Configuration=CplusC
%MSBuild% .\Excalibur.sln /t:Rebuild /p:Platform=x64 /p:Configuration=SharpG
%MSBuild% .\Excalibur.sln /t:Rebuild /p:Platform=x64 /p:Configuration=CplusG

xcopy ..\Build\x64\CplusC\Gaius.lib ..\Build\x64\SDK\Windows\CPU_ONLY\ /y
xcopy ..\Build\x64\CplusC\Gaius.dll ..\Build\x64\SDK\Windows\CPU_ONLY\ /y
xcopy ..\Build\x64\CplusC\Cassius.lib ..\Build\x64\SDK\Windows\CPU_ONLY\ /y
xcopy ..\Build\x64\CplusC\Cassius.dll ..\Build\x64\SDK\Windows\CPU_ONLY\ /y
xcopy ..\Build\x64\CplusC\Longinus.lib ..\Build\x64\SDK\Windows\CPU_ONLY\ /y
xcopy ..\Build\x64\CplusC\Longinus.dll ..\Build\x64\SDK\Windows\CPU_ONLY\ /y
xcopy ..\Build\x64\CplusC\Irisviel.lib ..\Build\x64\SDK\Windows\CPU_ONLY\ /y
xcopy ..\Build\x64\CplusC\Irisviel.dll ..\Build\x64\SDK\Windows\CPU_ONLY\ /y
xcopy ..\Build\x64\SharpC\Gaiunia.dll ..\Build\x64\SDK\Windows\CPU_ONLY\ /y
xcopy ..\Build\x64\SharpC\Cassiunia.dll ..\Build\x64\SDK\Windows\CPU_ONLY\ /y
xcopy ..\Build\x64\SharpC\Longinucia.dll ..\Build\x64\SDK\Windows\CPU_ONLY\ /y

xcopy ..\Build\x64\CplusG\Gaius.lib ..\Build\x64\SDK\Windows\CPU_GPU\ /y
xcopy ..\Build\x64\CplusG\Gaius.dll ..\Build\x64\SDK\Windows\CPU_GPU\ /y
xcopy ..\Build\x64\CplusG\Cassius.lib ..\Build\x64\SDK\Windows\CPU_GPU\ /y
xcopy ..\Build\x64\CplusG\Cassius.dll ..\Build\x64\SDK\Windows\CPU_GPU\ /y
xcopy ..\Build\x64\CplusG\Longinus.lib ..\Build\x64\SDK\Windows\CPU_GPU\ /y
xcopy ..\Build\x64\CplusG\Longinus.dll ..\Build\x64\SDK\Windows\CPU_GPU\ /y
xcopy ..\Build\x64\CplusG\Irisviel.lib ..\Build\x64\SDK\Windows\CPU_GPU\ /y
xcopy ..\Build\x64\CplusG\Irisviel.dll ..\Build\x64\SDK\Windows\CPU_GPU\ /y
xcopy ..\Build\x64\SharpG\Gaiunia.dll ..\Build\x64\SDK\Windows\CPU_GPU\ /y
xcopy ..\Build\x64\SharpG\Cassiunia.dll ..\Build\x64\SDK\Windows\CPU_GPU\ /y
xcopy ..\Build\x64\SharpG\Longinucia.dll ..\Build\x64\SDK\Windows\CPU_GPU\ /y

xcopy ..\include\Cassius\CassiusFeature.hpp ..\Build\x64\SDK\Windows\include\Cassius\ /y
xcopy ..\include\Damocles\damocles.hpp ..\Build\x64\SDK\Windows\include\Damocles\ /y
xcopy ..\include\Damocles\vdamocles.hpp ..\Build\x64\SDK\Windows\include\Damocles\ /y
xcopy ..\include\Gaius\GaiusFeature.hpp ..\Build\x64\SDK\Windows\include\Gaius\ /y
xcopy ..\include\Irisviel\IrisvielSearch.hpp ..\Build\x64\SDK\Windows\include\Irisviel\ /y
xcopy ..\include\Longinus\BaseLonginusCascade.hpp ..\Build\x64\SDK\Windows\include\Longinus\ /y
xcopy ..\include\Longinus\common.hpp ..\Build\x64\SDK\Windows\include\Longinus\ /y
xcopy ..\include\Longinus\LonginusDetector.hpp ..\Build\x64\SDK\Windows\include\Longinus\ /y
xcopy ..\include\Longinus\match_recter.hpp ..\Build\x64\SDK\Windows\include\Longinus\ /y
xcopy ..\include\Longinus\matcher.hpp ..\Build\x64\SDK\Windows\include\Longinus\ /y
xcopy ..\include\Selene\selene.hpp ..\Build\x64\SDK\Windows\include\Selene\ /y
pause