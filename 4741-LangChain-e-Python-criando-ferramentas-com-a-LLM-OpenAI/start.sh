#!/bin/bash
docker build -t app_langchain .
docker run -d --name app_langchain_container -v "$(pwd)":/app app_langchain
echo "Container app_langchain_container rodando em background."
echo "Para executar o script, use: docker exec -it app_langchain_container python main.py"
