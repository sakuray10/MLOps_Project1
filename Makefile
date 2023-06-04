install:
	pip install --upgrade pip &&\
		pip install 'git+https://github.com/facebookresearch/detectron2.git' #install detectron2 &&\
		pip install -r requirements.txt


lint:
	pylint --disable=R,C main.py

test:
	python -m pytest -vv --cov=main test_main.py

run:
	python main.py