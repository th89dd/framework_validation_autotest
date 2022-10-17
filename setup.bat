::  :version: 1.0
::  :copyright: 2018, Tim Häberlein, TU Dresden, FZM

@echo off
set venv_name=venv
set my_dir=%cd%
set cur_dir=%cd%

:: print actual python path and ask if its right
FOR /f %%i IN ('python -c "import os, sys; print(os.path.dirname(sys.executable))"') DO set python_dir=%%i
::set python_dir=%python -c "import os, sys; print(os.path.dirname(sys.executable))"%
@echo python dir is set to %python_dir%
set /p input=change path? (y/n):

if /i '%input%' == 'y' (
    goto :CHANGE1
) else (
    goto :NEXT1
)
:CHANGE1
set /p python_dir=set python dir:

:NEXT1
:: print actual path and ask for destiny path
@echo venv destiny dir is set to %my_dir%
set /p input=change path? (y/n):

if /i '%input%' == 'y' (
    goto :CHANGE2
) else (
    goto :NEXT2
)

:CHANGE2
set /p my_dir=set venv destiny dir:
mkdir %my_dir%

:NEXT2
:: install a venv
@%python_dir%\python.exe -m venv %my_dir%\%venv_name%

cd %my_dir%
cd %venv_name%\Scripts
@pip.exe install -r %cur_dir%\requirements.txt

cd %cur_dir%

pause