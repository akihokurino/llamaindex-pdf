SHELL := /bin/bash

venv:
	python3 -m venv venv

vendor:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt

setup:
	source venv/bin/activate && python index.py -s ./source -d ./index -f True

query:
	source venv/bin/activate && python query.py

chat:
	source venv/bin/activate && python chat.py