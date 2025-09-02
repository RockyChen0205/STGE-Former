@echo off
setlocal enabledelayedexpansion

:: 获取当前日期和时间
set "date=%date%"
set "time=%time%"
:: data= yyyy/mm/dd
set "date=!date:~0,4!%date:~5,2%!date:~8,2!"
set "time=!time:~0,2!%time:~3,2%!time:~6,2!"

:: 构建文件名
set "filename=!date!_!time!.log"

:: 运行命令并将输出重定向到文件


python .\eeg_slice.py >> %filename% 2>&1

endlocal