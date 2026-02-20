import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Ibovespa Predictor - Tech Challenge Fase 4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .prediction-low {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Carregar modelo e scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('ibovespa_model.pkl')
    scaler = joblib.load('ibovespa_scaler.pkl')
    return model, scaler

@st.cache_data
def load_validation_data():
    return pd.read_csv('validation_data.csv')

@st.cache_data
def load_model_info():
    with open('model_info.json', 'r') as f:
        return json.load(f)

# Carregar dados
model, scaler = load_model_and_scaler()
validation_data = load_validation_data()
model_info = load_model_info()

# Título principal
st.title("📈 Ibovespa Predictor - Tech Challenge Fase 4")
st.markdown("**Deploy e Monitoramento do Modelo Preditivo com Streamlit**")
st.markdown("---")

# Sidebar para navegação
st.sidebar.title("Menu de Navegação")
page = st.sidebar.radio("Selecione uma página:", 
                        ["🏠 Início", "🔮 Previsões", "📊 Painel de Métricas", "📋 Histórico"])

# ============================================================================
# PÁGINA 1: INÍCIO
# ============================================================================
if page == "🏠 Início":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Bem-vindo ao Ibovespa Predictor!")
        st.markdown("""
        Esta aplicação utiliza um modelo de **Machine Learning** (Random Forest Classifier) 
        treinado com dados históricos do Ibovespa para prever a direção do preço de fechamento 
        no dia seguinte.
        
        ### 📌 Características da Aplicação:
        
        - **Modelo Treinado**: Random Forest com 200 árvores e profundidade máxima de 13
        - **Período de Treino**: Abril de 2005 a Março de 2025 (4.953 amostras)
        - **Período de Validação**: Abril de 2025 (19 amostras)
        - **Features Utilizadas**: Abertura, Máxima, Mínima, Último, High_Low, Open_Close
        - **Dados Reais**: Históricos completos do Ibovespa de 2005 a 2025
        
        ### 🎯 Como Usar:
        
        1. **Previsões**: Insira os dados do dia e obtenha a previsão em tempo real
        2. **Painel de Métricas**: Visualize a performance do modelo
        3. **Histórico**: Consulte o histórico de previsões e logs de uso
        """)
    
    with col2:
        st.info(f"""
        ### 📊 Estatísticas Rápidas
        
        **Acurácia de Validação**: {model_info['accuracy_val']:.2%}
        
        **Precisão**: {model_info['precision']:.2%}
        
        **Recall**: {model_info['recall']:.2%}
        
        **F1-Score**: {model_info['f1_score']:.2%}
        """)

# ============================================================================
# PÁGINA 2: PREVISÕES
# ============================================================================
elif page == "🔮 Previsões":
    st.header("🔮 Faça uma Previsão")
    st.markdown("Insira os dados do dia e obtenha a previsão para o fechamento do dia seguinte.")
    
    # Criar colunas para entrada de dados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        abertura = st.number_input("Abertura", value=130000.0, step=100.0, format="%.2f")
        maxima = st.number_input("Máxima", value=131000.0, step=100.0, format="%.2f")
    
    with col2:
        minima = st.number_input("Mínima", value=129000.0, step=100.0, format="%.2f")
        ultimo = st.number_input("Último (Fechamento)", value=130500.0, step=100.0, format="%.2f")
    
    with col3:
        st.write("")  # Espaço vazio
        st.write("")
        fazer_previsao = st.button("🚀 Fazer Previsão", use_container_width=True)
    
    # Realizar previsão
    if fazer_previsao:
        # Calcular features
        high_low = maxima - minima
        open_close = ultimo - abertura
        
        # Preparar dados para o modelo
        features = np.array([[abertura, maxima, minima, ultimo, high_low, open_close]])
        features_scaled = scaler.transform(features)
        
        # Fazer previsão
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Exibir resultado
        st.markdown("---")
        
        if prediction == 1:
            st.markdown("""
            <div class="prediction-high">
            <h3>📈 PREVISÃO: ALTA</h3>
            <p>O modelo prevê que o Ibovespa <strong>fechará em alta</strong> no próximo dia.</p>
            </div>
            """, unsafe_allow_html=True)
            confianca = prediction_proba[1]
        else:
            st.markdown("""
            <div class="prediction-low">
            <h3>📉 PREVISÃO: BAIXA</h3>
            <p>O modelo prevê que o Ibovespa <strong>fechará em baixa</strong> no próximo dia.</p>
            </div>
            """, unsafe_allow_html=True)
            confianca = prediction_proba[0]
        
        # Exibir confiança
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confiança da Previsão", f"{confianca*100:.2f}%")
        with col2:
            st.metric("Probabilidade Alternativa", f"{(1-confianca)*100:.2f}%")
        
        # Visualizar confiança
        fig = go.Figure(data=[
            go.Bar(x=['Alta', 'Baixa'], y=[prediction_proba[1]*100, prediction_proba[0]*100],
                   marker_color=['#28a745', '#dc3545'])
        ])
        fig.update_layout(
            title="Distribuição de Probabilidades",
            yaxis_title="Probabilidade (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Salvar previsão em log
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'abertura': abertura,
            'maxima': maxima,
            'minima': minima,
            'ultimo': ultimo,
            'high_low': high_low,
            'open_close': open_close,
            'predicao': 'Alta' if prediction == 1 else 'Baixa',
            'confianca': float(confianca)
        }
        
        # Salvar em arquivo JSON
        log_file = 'prediction_logs.json'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        st.success("✅ Previsão salva no histórico!")

# ============================================================================
# PÁGINA 3: PAINEL DE MÉTRICAS
# ============================================================================
elif page == "📊 Painel de Métricas":
    st.header("📊 Painel de Monitoramento - Análise de Performance")
    st.markdown("Visualize as principais métricas de validação e performance do modelo.")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <h4>Acurácia</h4>
        <h2>{model_info['accuracy_val']:.2%}</h2>
        <p>Proporção de previsões corretas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <h4>Precisão</h4>
        <h2>{model_info['precision']:.2%}</h2>
        <p>Acertos entre previsões positivas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <h4>Recall</h4>
        <h2>{model_info['recall']:.2%}</h2>
        <p>Cobertura de casos positivos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
        <h4>F1-Score</h4>
        <h2>{model_info['f1_score']:.2%}</h2>
        <p>Média harmônica de precisão e recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráfico de série temporal de validação
    st.subheader("📈 Série Temporal - Período de Validação (Abril 2025)")
    
    validation_data['Data'] = pd.to_datetime(validation_data['Data'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=validation_data['Data'],
        y=validation_data['Último'],
        mode='lines+markers',
        name='Preço de Fechamento',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Evolução do Preço de Fechamento - Abril 2025",
        xaxis_title="Data",
        yaxis_title="Preço (Último)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas descritivas
    st.subheader("📊 Estatísticas Descritivas - Dados de Validação")
    
    stats = validation_data[['Abertura', 'Máxima', 'Mínima', 'Último', 'High_Low', 'Open_Close']].describe()
    st.dataframe(stats, use_container_width=True)
    
    # Matriz de correlação
    st.subheader("🔗 Matriz de Correlação - Features")
    
    correlation_matrix = validation_data[['Abertura', 'Máxima', 'Mínima', 'Último', 'High_Low', 'Open_Close']].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Correlação entre Features",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Informações do modelo
    st.subheader("🤖 Configuração do Modelo")
    
    model_config = {
        'Algoritmo': 'Random Forest Classifier',
        'Número de Árvores': 200,
        'Profundidade Máxima': 13,
        'Random State': 42,
        'Amostras de Treino': f"{model_info['train_samples']:,}",
        'Amostras de Validação': f"{model_info['val_samples']:,}",
        'Total de Amostras': f"{model_info['total_samples']:,}",
        'Features': 6
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        for key, value in list(model_config.items())[:3]:
            st.write(f"**{key}**: {value}")
    
    with col2:
        for key, value in list(model_config.items())[3:6]:
            st.write(f"**{key}**: {value}")
    
    with col3:
        for key, value in list(model_config.items())[6:]:
            st.write(f"**{key}**: {value}")
    
    # Período de dados
    st.subheader("📅 Período de Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Período de Treino:**")
        st.write(f"Início: {model_info['train_start_date']}")
        st.write(f"Fim: {model_info['train_end_date']}")
    
    with col2:
        st.write(f"**Período de Validação:**")
        st.write(f"Início: {model_info['val_start_date']}")
        st.write(f"Fim: {model_info['val_end_date']}")

# ============================================================================
# PÁGINA 4: HISTÓRICO
# ============================================================================
elif page == "📋 Histórico":
    st.header("📋 Histórico de Previsões")
    st.markdown("Visualize o histórico de todas as previsões realizadas.")
    
    log_file = 'prediction_logs.json'
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        if logs:
            # Converter para DataFrame
            logs_df = pd.DataFrame(logs)
            logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
            logs_df = logs_df.sort_values('timestamp', ascending=False)
            
            # Exibir estatísticas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Previsões", len(logs_df))
            
            with col2:
                altas = (logs_df['predicao'] == 'Alta').sum()
                st.metric("Previsões de Alta", altas)
            
            with col3:
                baixas = (logs_df['predicao'] == 'Baixa').sum()
                st.metric("Previsões de Baixa", baixas)
            
            st.markdown("---")
            
            # Tabela de previsões
            st.subheader("Detalhes das Previsões")
            
            display_df = logs_df[['timestamp', 'abertura', 'maxima', 'minima', 'ultimo', 'predicao', 'confianca']].copy()
            display_df.columns = ['Data/Hora', 'Abertura', 'Máxima', 'Mínima', 'Último', 'Previsão', 'Confiança (%)']
            display_df['Confiança (%)'] = display_df['Confiança (%)'].apply(lambda x: f"{x*100:.2f}%")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Gráfico de confiança ao longo do tempo
            st.subheader("📈 Evolução da Confiança das Previsões")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=logs_df['timestamp'],
                y=logs_df['confianca']*100,
                mode='lines+markers',
                name='Confiança',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Confiança das Previsões ao Longo do Tempo",
                xaxis_title="Data/Hora",
                yaxis_title="Confiança (%)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Exportar dados
            st.subheader("📥 Exportar Dados")
            
            csv = logs_df.to_csv(index=False)
            st.download_button(
                label="Baixar histórico em CSV",
                data=csv,
                file_name=f"ibovespa_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        else:
            st.info("📭 Nenhuma previsão realizada ainda. Acesse a página de Previsões para começar!")
    
    else:
        st.info("📭 Nenhuma previsão realizada ainda. Acesse a página de Previsões para começar!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>Ibovespa Predictor - Tech Challenge Fase 4 | Desenvolvido com Streamlit</p>
    <p>Modelo: Random Forest Classifier | Dados Reais: Abril 2005 - Abril 2025</p>
</div>
""", unsafe_allow_html=True)
