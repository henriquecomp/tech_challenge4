# Tech Challenge Fase 4 - Previsão de Ações (PETR4) com LSTM

Projeto de Machine Learning Engineering para previsão de preços de fechamento das ações da Petrobras (PETR4.SA) utilizando Redes Neurais Recorrentes (LSTM). O sistema inclui pipeline de retreino contínuo, monitoramento via MLflow e API RESTful.

## Funcionalidades

* **API REST (FastAPI):** Endpoints para previsão em tempo real e sob demanda.
* **Modelo LSTM:** Rede neural treinada em 5 anos de dados históricos.
* **MLflow:** Rastreamento de experimentos, métricas (LOSS, RMSE, MAE, MAPE) e artefatos.
* **Monitoramento:** Logs de tempo de resposta e endpoint `/metrics`.
* **Automação:** Agendamento de retreino diário automático.

## Como Executar

### Pré-requisitos
* Docker e Docker Compose instalados.

### Passo a Passo
1.  Clone o repositório.

2.  Suba os containers:
    ```bash
    docker-compose up --build
    ```

3.  Acesse os serviços:
    * **API (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
    * **MLflow UI:** [http://localhost:5001](http://localhost:5001)

## Endpoints Principais

* `POST /predict-custom`: Previsão baseada em dados enviados pelo usuário (JSON).
* `GET /predict-live`: Previsão baseada nos últimos 60 dias do Yahoo Finance.
* `POST /train`: Dispara o pipeline de retreino imediatamente.
* `GET /metrics`: Visualiza os últimos 50 logs de performance da API.

## Pipeline de Retreino Contínuo (Continuous Training)

O projeto implementa uma arquitetura de **Retreino Contínuo** para garantir que o modelo LSTM se mantenha atualizado com as tendências mais recentes do mercado financeiro. O processo ocorre da seguinte forma:

### 1. Gatilho de Execução
O retreino pode ser iniciado de duas formas:
* **Manual (On-Demand):** Através de uma requisição `POST` para o endpoint `/train`.
* **Automático (Agendado):** Configurado via `APScheduler` para rodar diariamente (ex: às 19:30), garantindo que o fechamento do dia seja incorporado.

### 2. O Fluxo de Treinamento
Ao ser acionado, o sistema executa os seguintes passos em segundo plano (`BackgroundTasks`), sem interromper a disponibilidade da API:

1.  **Coleta de Dados Frescos:** O script conecta-se à API do **Yahoo Finance** e baixa os últimos 5 anos de dados históricos da `PETR4.SA`.
2.  **Pré-processamento:** Os dados são normalizados (MinMaxScaler) e janelados (60 dias) para o formato esperado pela LSTM.
3.  **Treinamento:** Uma nova instância da rede neural é treinada do zero com os dados atualizados.
4.  **Avaliação e Log (MLflow):**
    * Métricas de erro (LOSS, RMSE, MAE, MAPE) são calculadas.
    * Gráficos comparativos (Real vs Previsto) são gerados.
    * Tudo é registrado no servidor **MLflow** para auditoria.
5.  **Persistência:** O novo modelo (`.pth`) e os escaladores (`.joblib`) são salvos no disco.

### 3. Hot Swapping (Atualização a Quente)
Após o sucesso do treinamento, a API recarrega automaticamente o modelo na memória RAM.
* **Resultado:** A próxima requisição para `/predict` já utilizará a versão mais recente e inteligente do modelo, sem necessidade de reiniciar o servidor ou causar *downtime*.

## Links de Produção
O projeto está deployado e acessível publicamente:

* **API (Swagger):** [https://tech_challenge.smarth.my/docs](https://tech_challenge.smarth.my/docs)
* **MLflow Dashboard:** [https://mlflow.smarth.my/](https://mlflow.smarth.my/)

## Autores
* Henrique Fávaro Tâmbalo (RM362398)
* Willian do Prado Vieira (RM360949)