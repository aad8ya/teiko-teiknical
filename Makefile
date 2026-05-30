.PHONY: setup pipeline dashboard

setup:
	pip install -r requirements.txt

pipeline:
	python load_data.py && python run_pipeline.py

dashboard:
	streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
