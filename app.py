import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io

# Configuration de la page
st.set_page_config(
    page_title="🌱 PlantDoc AI - Détection de Maladies",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design ultra-professionnel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .content-wrapper {
        background: white;
        margin: 2rem;
        border-radius: 24px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #2E8B57 0%, #90EE90 100%);
        padding: 3rem 2rem;
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
        opacity: 0.3;
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        font-weight: 300;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    
    .hero-description {
        font-size: 1.1rem;
        opacity: 0.8;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .main-content {
        padding: 3rem 2rem;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .upload-area {
        border: 3px dashed #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(145deg, #f8fafc, #f1f5f9);
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .upload-area:hover {
        border-color: #2E8B57;
        background: linear-gradient(145deg, #f0fff4, #e6fffa);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46, 139, 87, 0.15);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .prediction-content {
        position: relative;
        z-index: 1;
    }
    
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    
    .prediction-result {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .confidence-score {
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .confidence-high { color: #48bb78; }
    .confidence-medium { color: #ed8936; }
    .confidence-low { color: #f56565; }
    
    .status-card {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 6px solid;
    }
    
    .status-success {
        background: linear-gradient(135deg, #f0fff4, #c6f6d5);
        border-color: #48bb78;
        color: #22543d;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fffbf0, #feebc8);
        border-color: #ed8936;
        color: #744210;
    }
    
    .status-info {
        background: linear-gradient(135deg, #f0f8ff, #bee3f8);
        border-color: #4299e1;
        color: #2a4365;
    }
    
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        background: white;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .image-info {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 12px;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #4a5568;
    }
    
    .sidebar-card {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .metric-item {
        text-align: center;
        padding: 1rem;
        background: rgba(46, 139, 87, 0.1);
        border-radius: 12px;
        border: 2px solid rgba(46, 139, 87, 0.2);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2E8B57;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #4a5568;
        margin-top: 0.25rem;
    }
    
    .instructions-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .instruction-card {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .instruction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }
    
    .instruction-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .instruction-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    /* Customisation des composants Streamlit */
    .stFileUploader > div > div {
        border: none !important;
        background: transparent !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #2E8B57, #90EE90) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2E8B57, #90EE90);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(46, 139, 87, 0.3);
    }
    
    .stSpinner > div {
        border-color: #2E8B57 transparent transparent transparent !important;
    }
    
    /* Animation de chargement */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# Configuration du modèle
IMG_SIZE = (64, 64)

# Liste des classes
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

@st.cache_resource
def load_model():
    """Charger le modèle pré-entraîné"""
    try:
        model = tf.keras.models.load_model('plant_disease_model.h5')
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

def preprocess_image(image):
    """Préprocesser l'image pour la prédiction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_disease(model, image):
    """Faire une prédiction sur l'image"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return predictions[0]

def format_class_name(class_name):
    """Formatter le nom de la classe pour l'affichage"""
    parts = class_name.split('___')
    plant = parts[0].replace('_', ' ').title()
    condition = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Unknown"
    return f"{plant} - {condition}"

def get_confidence_color(confidence):
    """Retourner la classe CSS selon le niveau de confiance"""
    if confidence > 0.8:
        return "confidence-high"
    elif confidence > 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_prediction_chart(predictions, class_names, top_n=5):
    """Créer un graphique des prédictions"""
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    top_predictions = predictions[top_indices]
    top_classes = [format_class_name(class_names[i]) for i in top_indices]
    
    # Couleurs personnalisées pour le graphique
    colors = ['#2E8B57', '#32CD32', '#90EE90', '#98FB98', '#F0FFF0']
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_classes,
            x=top_predictions,
            orientation='h',
            marker=dict(
                color=colors[:len(top_predictions)],
                line=dict(color='white', width=2)
            ),
            text=[f'{p:.1%}' for p in top_predictions],
            textposition='inside',
            textfont=dict(color='white', size=12, family='Inter')
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=f"🎯 Top {top_n} Prédictions",
            font=dict(size=18, family='Inter', color='#2d3748'),
            x=0.5
        ),
        xaxis=dict(
            title="Probabilité",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            title_font=dict(family='Inter')
        ),
        yaxis=dict(
            title="",
            title_font=dict(family='Inter')
        ),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#2d3748')
    )
    
    return fig

def main():
    # Wrapper principal avec design moderne
    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)
    
    # Section héro
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <div class="hero-title">🌱 PlantDoc AI</div>
            <div class="hero-subtitle">Détection Intelligente des Maladies des Plantes</div>
            <div class="hero-description">
                Utilisez l'intelligence artificielle pour diagnostiquer instantanément les maladies de vos plantes
                grâce à une simple photo de feuille
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Charger le modèle
    model = load_model()
    
    if model is None:
        st.markdown("""
        <div class="main-content">
            <div class="status-card status-warning">
                <h3>❌ Modèle non disponible</h3>
                <p>Impossible de charger le modèle. Vérifiez que le fichier 'plant_disease_model.h5' existe dans le répertoire.</p>
                <p><strong>💡 Conseil :</strong> Assurez-vous que votre modèle .h5 est dans le même dossier que cette application.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-title">🤖 Informations du Modèle</div>
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Classes</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{}x{}</div>
                    <div class="metric-label">Résolution</div>
                </div>
            </div>
        </div>
        """.format(len(CLASS_NAMES), IMG_SIZE[0], IMG_SIZE[1]), unsafe_allow_html=True)
        
        
        confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 
                                        help="Seuil minimum pour considérer une prédiction comme fiable")
    
    # Contenu principal
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-title">📤 Télécharger une Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choisissez une image...",
            type=['png', 'jpg', 'jpeg'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="", use_container_width=True)
            
            st.markdown(f"""
            <div class="image-info">
                <strong>📋 Informations de l'image :</strong><br>
                • Taille originale : {image.size[0]} × {image.size[1]} pixels<br>
                • Mode couleur : {image.mode}<br>
                • Taille du fichier : {len(uploaded_file.getvalue())/1024:.1f} KB
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<div class="section-title">🔍 Résultats de l\'Analyse</div>', unsafe_allow_html=True)
            
            with st.spinner("🔬 Analyse en cours..."):
                predictions = predict_disease(model, image)
                
                best_idx = np.argmax(predictions)
                best_confidence = predictions[best_idx]
                best_class = format_class_name(CLASS_NAMES[best_idx])
                
                confidence_class = get_confidence_color(best_confidence)
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-content">
                        <div class="prediction-title"> Diagnostic Principal</div>
                        <div class="prediction-result">{best_class}</div>
                        <div class="confidence-score {confidence_class}">{best_confidence:.1%}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Interprétation avec design moderne
                if best_confidence > confidence_threshold:
                    if "healthy" in CLASS_NAMES[best_idx].lower():
                        st.markdown("""
                        <div class="status-card status-success">
                            <h4>Excellente nouvelle !</h4>
                            <p>Votre plante semble être en parfaite santé. Continuez vos bons soins !</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="status-card status-warning">
                            <h4>Attention requise</h4>
                            <p>Une maladie potentielle a été détectée. Nous recommandons de consulter un expert pour confirmation et traitement.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="status-card status-info">
                        <h4>ℹRésultat incertain</h4>
                        <p>L'analyse n'est pas concluante. Essayez avec une image plus nette et mieux éclairée.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()