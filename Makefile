install:
	python -m pip install -r requirements.txt
	python -m pip install -r apps/api/requirements.txt
api:
	uvicorn apps.api.main:app --reload --port 8000
web:
	cd apps/web && npm install && npm run dev
