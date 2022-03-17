@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%1" == "" goto install

goto %1


:format
isort pynlin
black -t py36 pynlin
docformatter --in-place --recursive pynlin
goto end

:lint
flake8 pynlin
pylint pynlin
goto end


:install
pip install -r requirements.txt
pip install -e .
goto end


:test
mkdir tests\reports
python -m pytest

:end
popd
