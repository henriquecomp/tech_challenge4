import time
import logging
import torch
import joblib
import numpy as np
import yfinance as yf
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import timedelta, datetime
from typing import List
from ContinuousTraining import RNN, ContinuousTraining
from collections import deque
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api_monitoring")

scheduler = AsyncIOScheduler()

# API 
app = FastAPI(
    title="API Previsão PETR4 - Tech Challenge Fase 4",
    description="API para prever preços de ações usando LSTM com suporte a Retreino Contínuo"
)

# Lista de tamanho fixo para armazenar os últimos 50 logs na memória
logs_recentes = deque(maxlen=50)

# CPU ou GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gerenciamento de Estado Global
modelo = None
scaler_preco = None
scaler_volume = None

def carregar_modelo():
    """
    Carrega ou recarrega o modelo e scalers na memória da API.
    Chamada na inicialização e após cada retreino bem-sucedido.
    """
    global modelo, scaler_preco, scaler_volume
    try:
        # Usa a arquitetura
        novo_modelo = RNN(input_size=2, hidden_size=100, output_size=1)
        
        # Carrega os pesos salvos
        novo_modelo.load_state_dict(torch.load('modelo_petr4.pth', map_location=device))
        novo_modelo.to(device)
        novo_modelo.eval()  # Modo de avaliação
        
        # Atualiza as variáveis globais com o modelo
        modelo = novo_modelo
        scaler_preco = joblib.load('scaler_preco.joblib')
        scaler_volume = joblib.load('scaler_volume.joblib')
        
        logger.info("Modelo LSTM e Scalers carregados/atualizados com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo (pode não existir ainda): {e}")


# Carrega na inicialização
carregar_modelo()

# Middleware de monitoramento
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Cria a string de log formatada
    log_entry = (
        f"Path: {request.url.path} | "
        f"Method: {request.method} | "
        f"Status: {response.status_code} | "
        f"Tempo: {process_time:.4f}s"
    )
    
    # Loga no console (para o Docker/Administrador ver)
    logger.info(log_entry)
    
    # Salva na lista em memória. Não grava do metrics
    if request.url.path in("/predict-live", "/predict-custom") :
        logs_recentes.append(log_entry)
    
    return response

# Model de entrada 
class StockInput(BaseModel):
    precos: List[float]
    volumes: List[float]

# Calcula proxima data util
def proxima_data_util(data_inicial):
    """Calcula a próxima data útil (pula sábado e domingo)."""
    data_proxima = data_inicial + timedelta(days=1)
    while data_proxima.weekday() >= 5: 
        data_proxima += timedelta(days=1)
    return data_proxima.date()

# Realiza a inferencia
def realizar_previsao(dados_completos):
    """Função central de inferência."""
    if modelo is None:
        raise RuntimeError("Modelo não está carregado na memória.")

    try:
        X = np.array(dados_completos)
        X = np.reshape(X, (1, X.shape[0], X.shape[1]))
        
        X_tensor = torch.from_numpy(X).float().to(device)
        
        with torch.no_grad():
            pred = modelo(X_tensor)
        
        pred = pred.cpu().numpy()
        preco_final = scaler_preco.inverse_transform(pred)
        
        return round(float(preco_final[0][0]), 2)
    except Exception as e:
        logger.error(f"Erro na inferência: {e}")
        raise e

# Tarefa de Background (Retreino) 
def tarefa_retreino_background():
    """
    Executa o pipeline de treinamento em background.
    Se funcionar, atualiza o modelo da API "a quente".
    """
    logger.info("Iniciando tarefa de retreino...")
    trainer = ContinuousTraining() 
    
    # Executa o treino
    sucesso = trainer.train(epochs=150)
    
    if sucesso:
        logger.info("Treino finalizado com sucesso. Atualizando API...")
        carregar_modelo()
    else:
        logger.error("Falha no treinamento. O modelo antigo será mantido.")

# Endpoints

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "model_loaded": modelo is not None
    }

@app.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    """
    Gatilho para iniciar o retreino do modelo (Continuous Training).
    """
    background_tasks.add_task(tarefa_retreino_background)
    return {
        "status": "accepted",
        "message": "Treinamento iniciado em segundo plano. Acompanhe os logs."
    }

@app.get("/predict-live")
def predict_live():
    """Busca dados do Yahoo Finance e faz a previsão."""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não treinado/carregado.")

    try:
        # Baixa dados para garantir 60 dias úteis
        dados = yf.download("PETR4.SA", period="90d")
        
        if len(dados) < 60:
            raise HTTPException(status_code=503, detail="Dados insuficientes")
            
        dados = dados.tail(60)
        
        data_close = dados["Close"].values.reshape(-1, 1)
        data_volume = dados["Volume"].values.reshape(-1, 1)
        
        # Usa os scalers globais carregados
        dados_scaled = scaler_preco.transform(data_close)
        volume_scaled = scaler_volume.transform(data_volume)
        
        dados_completos = np.hstack((dados_scaled, volume_scaled))
        
        preco_previsto = realizar_previsao(dados_completos)
        dia_previsto = proxima_data_util(datetime.now())
        
        # Pega o último preço real para comparação
        ultimo_preco_real = float(data_close[-1][0])

        return {
            "tipo": "live",
            "ativo": "PETR4.SA",
            "ultimo_preco_real": round(ultimo_preco_real, 2),
            "data_previsao": dia_previsto,
            "preco_previsto": preco_previsto,
            "unidade": "BRL"
        }

    except Exception as e:
        logger.error(f"Erro no endpoint /predict-live: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-custom")
def predict_custom(input_data: StockInput):
    """Recebe dados manuais do usuário para previsão."""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não treinado/carregado.")

    if len(input_data.precos) != 60 or len(input_data.volumes) != 60:
        raise HTTPException(
            status_code=400, 
            detail=f"Requer 60 dias. Recebido: {len(input_data.precos)} preços."
        )

    try:
        data_close = np.array(input_data.precos).reshape(-1, 1)
        data_volume = np.array(input_data.volumes).reshape(-1, 1)
        
        dados_scaled = scaler_preco.transform(data_close)
        volume_scaled = scaler_volume.transform(data_volume)
        
        dados_completos = np.hstack((dados_scaled, volume_scaled))
        
        preco_previsto = realizar_previsao(dados_completos)
        
        return {
            "tipo": "custom",
            "mensagem": "Previsão com dados do usuário",
            "ultimo_preco_enviado": round(float(data_close[-1][0]), 2),
            "preco_previsto": preco_previsto,
            "unidade": "BRL"
        }

    except Exception as e:
        logger.error(f"Erro no endpoint /predict-custom: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno no processamento.")
    
@app.get("/metrics")
def get_metrics():
    """
    Retorna os últimos 50 logs de requisição armazenados em memória.
    Útil para monitoramento rápido sem acesso ao terminal.
    """
    return {
        "total_registros": len(logs_recentes),
        "logs": list(logs_recentes) # Converte o deque para lista padrão para o JSON
    }


@app.on_event("startup")
def iniciar_agendador():
    try:
        # Agenda o treino para todo dia as 19:30
        fuso_brasil = timezone('America/Sao_Paulo')
        trigger = CronTrigger(hour=19, minute=30,timezone=fuso_brasil)
        scheduler.add_job(tarefa_retreino_background, trigger)
        scheduler.start()
        logger.info("Agendador de retreino iniciado (Diariamente às 18:00).")
    except Exception as e:
        logger.error(f"Erro ao iniciar agendador: {e}")