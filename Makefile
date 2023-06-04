install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt 
		# &&\
		# pip install 'git+https://github.com/facebookresearch/detectron2.git' #install detectron2

lint:
	pylint --disable=R,C hello.py

test:
	python -m pytest -vv --cov=hello test_hello.py

run:
	python main.py