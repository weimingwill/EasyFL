protobuf:
	python -m grpc_tools.protoc -I./protos \
		--python_out=. \
		--grpc_python_out=. \
		protos/easyfl/pb/*.proto

base_image:
	docker build -f docker/base.Dockerfile -t easyfl:base .

image:
	docker build -f docker/client.Dockerfile -t easyfl-client .
	docker build -f docker/server.Dockerfile -t easyfl-server .
	docker build -f docker/tracker.Dockerfile -t easyfl-tracker .
	docker build -f docker/run.Dockerfile -t easyfl-run .
