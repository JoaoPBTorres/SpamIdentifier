from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re

app = Flask(__name__)
CORS(app)

modelo = joblib.load('melhor_modelo_spam.pkl')
scaler = joblib.load('scaler_spam.pkl')

FEATURES_PALAVRAS = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet", "order", "mail", "receive", 
    "will", "people", "report", "addresses", "free", "business", "email", "you", "credit", "your", 
    "font", "000", "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857", "data", 
    "415", "85", "technology", "1999", "parts", "pm", "direct", "cs", "meeting", "original", "project", 
    "re", "edu", "table", "conference", ";", "(", "[", "!", "$", "#"
]

def extrair_caracteristicas(texto):
    texto_lower = texto.lower()
    vetor = []
    
    # Calcula a frequência percentual de cada palavra/caractere no texto
    palavras_totais = len(re.findall(r'\w+', texto_lower)) if len(re.findall(r'\w+', texto_lower)) > 0 else 1
    
    for feat in FEATURES_PALAVRAS:
        if feat in [";", "(", "[", "!", "$", "#"]:
            qtd = texto_lower.count(feat)
            porcentagem = (qtd / palavras_totais) * 100
        else:
            qtd = len(re.findall(r'\b' + feat + r'\b', texto_lower))
            porcentagem = (qtd / palavras_totais) * 100
        vetor.append(porcentagem)
        
    while len(vetor) < 57:
        vetor.append(0.0)
        
    return np.array(vetor).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    dados = request.get_json()
    texto_email = dados.get('texto', '')
    
    if not texto_email.strip():
        return jsonify({'error': 'Texto vazio'}), 400
        
    features = extrair_caracteristicas(texto_email)
    
    features_scaled = scaler.transform(features)
    
    predicao = modelo.predict(features_scaled)[0]
    probabilidade = modelo.predict_proba(features_scaled)[0][1]
    
    return jsonify({
        'is_spam': int(predicao),
        'probabilidade': float(probabilidade * 100)
    })

if __name__ == '__main__':
    print("Servidor do Modelo rodando...")
    app.run(port=5000, debug=True)