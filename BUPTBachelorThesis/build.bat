@echo off
REM 设置干净的 PATH，排除 adb.exe 错误路径
set PATH=D:\desktop\app\installation\MiKTeX\miktex\bin\x64;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0

REM 编译 LaTeX 文档
xelatex -interaction=nonstopmode main.tex

REM 检查编译结果
if exist main.pdf (
    echo.
    echo ================================
    echo 编译成功！PDF 已生成。
    echo ================================
    start main.pdf
) else (
    echo.
    echo ================================
    echo 编译失败！请检查错误日志。
    echo ================================
)
