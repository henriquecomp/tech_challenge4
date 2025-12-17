import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import joblib
import logging
import mlflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mlflow.models.signature import infer_signature

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("training_pipeline")

# Definição da Arquitetura da Rede Neural
class RNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=100, output_size=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        ultima_saida = lstm_out[:, -1, :]
        previsao = self.linear(ultima_saida)
        return previsao

# Classe de Treinamento Contínuo
class ContinuousTraining:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.janela = 60
        self.ticker = "PETR4.SA"

    def _get_data(self):
        """Baixa dados do Yahoo Finance"""
        logger.info(f"Baixando dados atualizados para {self.ticker}...")
        # Baixa 5 anos para garantir massa de dados suficiente para treino
        dados = yf.download(self.ticker, period="5y")
        if len(dados) == 0:
            raise ValueError("Não foi possível baixar dados do Yahoo Finance (DataFrame vazio).")
        return dados

    def _preprocess(self, dados):
        """Prepara os dados para o formato LSTM (Sliding Window)"""
        logger.info("Pré-processando e normalizando dados...")
        
        data_close = dados["Close"].values.reshape(-1, 1)
        data_volume = dados["Volume"].values.reshape(-1, 1)

        # Cria escaladores 
        scaler_preco = MinMaxScaler()
        scaler_volume = MinMaxScaler()

        dados_scaled = scaler_preco.fit_transform(data_close)
        volume_scaled = scaler_volume.fit_transform(data_volume)
        
        # Junta Preço e Volume
        dados_completos = np.hstack((dados_scaled, volume_scaled))
        
        X, y = [], []
        for i in range(self.janela, len(dados_completos)):
            X.append(dados_completos[i-self.janela:i, :]) 
            y.append(dados_completos[i, 0])                
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 2))
        
        return X, y, scaler_preco, scaler_volume

    def train(self, epochs=150, lr=0.001):
        """
        Executa o pipeline completo:
        Coleta -> Processamento -> Treino -> Avaliação -> Registro no MLflow -> Salvamento
        """
        try:
            # Obtenção e Tratamento dos Dados
            raw_data = self._get_data()
            X, y, scaler_p, scaler_v = self._preprocess(raw_data)

            # Envia para o dispositivo
            X_tensor = torch.from_numpy(X).float().to(self.device)
            y_tensor = torch.from_numpy(y).float().to(self.device)

            # Definição de Hiperparâmetros e Modelo
            hidden_size = 100
            model = RNN(input_size=2, hidden_size=hidden_size, output_size=1).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Configuração do MLflow
            # Pega a URI do ambiente (Docker) ou usa local se não existir
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("Continuous_Training_PETR4")
            
            logger.info("Iniciando treinamento...")

            with mlflow.start_run():
                # LOG DE PARÂMETROS
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("window_size", self.janela)
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("hidden_size", hidden_size)
                mlflow.log_param("optimizer", "Adam")
                mlflow.log_param("device", str(self.device))

                model.train()
                
                # LOOP DE TREINAMENTO
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    target = y_tensor.view(-1, 1)
                    
                    # Loss de Treino (MSE nos dados normalizados 0-1)
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()

                    # CÁLCULO DE MÉTRICAS REAIS (R$) A CADA ÉPOCA 
                    # Desnormaliza para valores reais
                    pred_np = outputs.cpu().detach().numpy()
                    target_np = target.cpu().detach().numpy()
                    
                    pred_real = scaler_p.inverse_transform(pred_np)
                    target_real = scaler_p.inverse_transform(target_np)
                    
                    # Calcula metricas
                    mae = np.mean(np.abs(target_real - pred_real))
                    rmse = np.sqrt(np.mean((target_real - pred_real)**2))
                    # Adiciona 1e-8 para evitar divisão por zero
                    mape = np.mean(np.abs((target_real - pred_real) / (target_real + 1e-8))) * 100

                    # 3. Log no MLflow
                    mlflow.log_metric("loss_scaled", loss.item(), step=epoch)
                    mlflow.log_metric("mae_real", mae, step=epoch)
                    mlflow.log_metric("rmse_real", rmse, step=epoch)
                    mlflow.log_metric("mape_percent", mape, step=epoch)
                    
                    # Log no Console (apenas a cada 10 para não poluir)
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch [{epoch+1}/{epochs}] | MAPE: {mape:.2f}% | MAE: R${mae:.2f} | RMSE: R${rmse:.2f}")

                # GERAÇÃO DE GRÁFICO COMPARATIVO
                logger.info("Gerando gráfico de performance...")
                model.eval()
                with torch.no_grad():
                    full_pred = model(X_tensor).cpu().numpy()
                
                # Desnormaliza dados completos
                full_pred_real = scaler_p.inverse_transform(full_pred)
                full_target_real = scaler_p.inverse_transform(y_tensor.cpu().numpy().reshape(-1, 1))

                # Cria figura
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(full_target_real, label='Preço Real', color='black', linewidth=1)
                ax.plot(full_pred_real, label='Previsão LSTM', color='orange', alpha=0.8, linewidth=1)
                ax.set_title(f"Resultado do Treino - PETR4 (RMSE: R${rmse:.2f})")
                ax.set_xlabel("Dias (Janela de Treino)")
                ax.set_ylabel("Preço (R$)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Salva figura no MLflow
                mlflow.log_figure(fig, "graficos/comparativo_treino.png")
                plt.close(fig) # Libera memória

                # SALVAMENTO E REGISTRO DO MODELO 
                logger.info("Salvando artefatos locais e registrando modelo...")
                
                # Salva localmente para a API usar imediatamente
                torch.save(model.state_dict(), 'modelo_petr4.pth')
                joblib.dump(scaler_p, 'scaler_preco.joblib')
                joblib.dump(scaler_v, 'scaler_volume.joblib')
                
                # Cria Assinatura do Modelo (Schema de entrada/saída)
                # Pega um exemplo de entrada (1ª linha) e saída
                input_sample = X[0:1] 
                output_sample = full_pred[0:1]
                signature = infer_signature(input_sample, output_sample)

                # Log do Modelo no MLflow
                mlflow.pytorch.log_model(
                    model, 
                    "lstm_model", 
                    signature=signature,
                    registered_model_name="PETR4_LSTM_Predictor"
                )
                
            logger.info("Pipeline de treinamento concluído com sucesso!")
            return True

        except Exception as e:
            logger.error(f"FATAL: Falha no pipeline de treinamento: {str(e)}")
            return False