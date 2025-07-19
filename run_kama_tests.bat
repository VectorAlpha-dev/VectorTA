@echo off
echo Running KAMA Python tests...
call .venv\Scripts\activate
cd tests\python
python -m pytest test_kama.py -v
pause