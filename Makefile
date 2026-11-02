.PHONY: app debug down test acceptance lint

app:
	docker compose --profile app up --build

debug:
	docker compose --profile debug up --build

down:
	docker compose down --remove-orphans

test:
	pytest tests/ -v

acceptance:
	pytest -m acceptance -v

lint:
	ruff check . && ty check
