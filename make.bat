@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%1" == "" goto install

goto %1


:format
isort pynlin tests scripts
black -t py36 pynlin tests scripts
docformatter --in-place --recursive pynlin tests scripts
goto end

:lint
flake8 pynlin tests scripts
pylint pynlin tests scripts
goto end


:install
pip install -e .[dev]
goto end


:test
mkdir tests\reports
python -m pytest

:end
popd
