@echo off
xcopy /v /y "Resources\\fonts\\dejavu-fonts-ttf-2.37\\ttf\\DejaVuSansMono.ttf" "Build\Debug\Fonts\"
xcopy /v /y "Resources\\fonts\\fontawesome-webfont.ttf" "Build\Debug\Fonts\"
xcopy /v /y "Resources\\fonts\\Roboto\\Roboto-Italic.ttf" "Build\Debug\Fonts\"
xcopy /v /y "Resources\\fonts\\Roboto\\Roboto-Thin.ttf" "Build\Debug\Fonts\"
xcopy /v /y "Resources\\fonts\\Roboto\\Roboto-Bold.ttf" "Build\Debug\Fonts\"
xcopy /v /y "Resources\\fonts\\Roboto\\Roboto-BoldItalic.ttf" "Build\Debug\Fonts\"
xcopy /v /y "../../Build\\Source\\LTSimulationEngineRuntime.dll" "Build\Debug\"
