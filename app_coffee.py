import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuración de página
st.set_page_config(page_title="Dashboard: Consumer Behavior - Coffee", layout="wide")

# Carga de datos con caché para optmizar el rendimiento (Mejor práctica en Streamlit)
@st.cache_data
def load_data():
    df = pd.read_csv("data_coffee.csv", encoding="latin-1")
    
    # Mapeo de variables clave para que los gráficos sean más legibles
    brand_map = {1: 'Starbucks', 2: 'Dunkin', 3: 'Peets', 4: 'Caribou'}
    gender_map = {1: 'Mujer', 2: 'Hombre'}
    income_map = {1: '< $30k', 2: '$30k - $70k', 3: '> $70k'}
    snob_map = {1: 'Principiante', 2: 'Ignorante', 3: 'Conocedor', 4: 'Experto'}
    work_map = {1: 'Jornada Completa', 2: 'Media Jornada', 3: 'Desempleado'}

    df['Marca_Elegida'] = df['CHOSEN'].map(brand_map)
    df['Genero'] = df['GENDER'].map(gender_map)
    df['Ingresos'] = df['INCOME'].map(income_map)
    df['Nivel_Snob'] = df['SNOB'].map(snob_map)
    df['Situacion_Laboral'] = df['WORK'].map(work_map)
    
    return df

df = load_data()

st.title("Análisis del Comportamiento del Consumidor: Marcas de Café")
st.markdown("---")
st.markdown("""
Este panel permite explorar el conjunto de datos de decisiones de elección de cafeterías mediante
análisis estadísticos, tablas cruzadas (crosstabs) y regresiones logísticas.
""")

# ---------- 1. BARRA LATERAL (SIDEBAR) ----------
st.sidebar.header("Filtros Globales")
st.sidebar.markdown("Filtra los datos para explorar segmentos específicos:")
selected_gender = st.sidebar.multiselect("Filtrar por Género", options=df['Genero'].dropna().unique(), default=df['Genero'].dropna().unique())
selected_income = st.sidebar.multiselect("Filtrar por Nivel de Ingresos", options=df['Ingresos'].dropna().unique(), default=df['Ingresos'].dropna().unique())

# Aplicar filtros
df_filtered = df[(df['Genero'].isin(selected_gender)) & (df['Ingresos'].isin(selected_income))]

# ---------- 2. KPIs y METRICAS PRINCIPALES ----------
st.header("1. Visión General de la Muestra")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Observaciones", len(df_filtered))
col2.metric("Edad Promedio", f"{df_filtered['AGE'].mean():.1f} años")
col3.metric("Marca Más Popular", df_filtered['Marca_Elegida'].mode()[0])
col4.metric("Nivel Snob Más Frecuente", df_filtered['Nivel_Snob'].mode()[0])

st.markdown("<br>", unsafe_allow_html=True)

# ---------- 3. ANALISIS EXPLORATORIO (CROSSTABS Y GRAFICOS) ----------
st.header("2. Cruce de Variables y Análisis Exploratorio")

tab1, tab2, tab3, tab4 = st.tabs(["Frecuencias por Marca", "Análisis Sociodemográfico (Crosstab)", "Métricas Adicionales", "Perfil de Marcas (Radar)"])

with tab1:
    st.subheader("Distribución de la Marca Elegida")
    brand_counts = df_filtered['Marca_Elegida'].value_counts().reset_index()
    brand_counts.columns = ['Marca_Elegida', 'Cantidad']
    
    fig_brand = px.bar(brand_counts, x='Marca_Elegida', y='Cantidad', color='Marca_Elegida',
                       title="Recuento de Elección de Marcas",
                       labels={'Cantidad': 'Número de Personas', 'Marca_Elegida': 'Marca'},
                       color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_brand.update_layout(xaxis_title="Marca", yaxis_title="Cantidad de Personas Elegidas")
    st.plotly_chart(fig_brand)

with tab2:
    st.subheader("Cruce de Variables Interactivos (Cross Tables)")
    st.markdown("Selecciona dos variables categóricas para observar cómo se distribuyen y se relacionan entre ellas.")
    
    col_x, col_y = st.columns(2)
    with col_x:
        var_x = st.selectbox("Selecciona Variable 1 (Eje X)", ['Marca_Elegida', 'Genero', 'Ingresos', 'Nivel_Snob'])
    with col_y:
        var_color = st.selectbox("Selecciona Variable 2 (Asignación por Color)", ['Genero', 'Ingresos', 'Marca_Elegida', 'Nivel_Snob'])
    
    # Crear crosstab y visualizar como heatmap o gráfico de barras apiladas
    crosstab_data = pd.crosstab(df_filtered[var_x], df_filtered[var_color])
    st.dataframe(crosstab_data.style.background_gradient(cmap='Blues'))
    
    # Gráfico de barras normalizado apilado (100%) para ver proporciones
    crosstab_norm = pd.crosstab(df_filtered[var_x], df_filtered[var_color], normalize='index') * 100
    fig_cross = px.bar(crosstab_norm, barmode='stack', 
                       title=f"Distribución Relativa: {var_x} vs {var_color}",
                       labels={'value': 'Porcentaje (%)', var_x: var_x},
                       color_discrete_sequence=px.colors.qualitative.Set2)
    fig_cross.update_layout(yaxis_title="Porcentaje (%)")
    st.plotly_chart(fig_cross)

with tab3:
    st.subheader("Distribución de Situación Laboral por Marca")
    fig_work = px.histogram(df_filtered, x='Marca_Elegida', color='Situacion_Laboral', barmode='group',
                            title="Desglose Laboral según Marca",
                            labels={'Marca_Elegida': 'Marca', 'count': 'Frecuencia'},
                            color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_work)

    st.subheader("Distribución de la Edad según Marca Elegida")
    fig_box = px.box(df_filtered, x='Marca_Elegida', y='AGE', color='Marca_Elegida', 
                     title="Boxplot de Edad por Marca Elegida",
                     labels={'AGE': 'Edad (años)', 'Marca_Elegida': 'Marca'},
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_box)

with tab4:
    st.subheader("Perfil Comparativo de Atributos (Radar Plot)")
    st.markdown("Comparación eficiente de la percepción promedio de cada atributo entre las distintas marcas.")
    
    # Lista de atributos base y sufijos de marca
    attributes = ['TASTE', 'AMB', 'SPEED', 'MENU', 'APP', 'ADV']
    brand_suffixes = ['STA', 'DUN', 'PEET', 'CAR']
    brand_names = ['Starbucks', 'Dunkin', 'Peets', 'Caribou']
    
    # Pre-calcular promedios vectorizados para eficiencia
    radar_data = []
    for suffix, b_name in zip(brand_suffixes, brand_names):
        for attr in attributes:
            col = f"{attr}_{suffix}"
            if col in df_filtered.columns:
                radar_data.append({'Marca': b_name, 'Atributo': attr, 'Puntuación Promedio': df_filtered[col].mean()})
                
    df_radar = pd.DataFrame(radar_data)
    
    if not df_radar.empty:
        # line_polar es la forma óptima de hacer Radar Plots en plotly express
        fig_radar = px.line_polar(df_radar, r='Puntuación Promedio', theta='Atributo', color='Marca',
                                  line_close=True, markers=True,
                                  title="Comparación Perceptual Promedio por Marca",
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
        
        # Ajustar el eje radial para que vaya de 1 a 5 (escalas de Likert habituales)
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No hay columnas de atributos perceptuales disponibles en el dataset.")

# ---------- 4. REGRESIÓN LOGÍSTICA Y EXPLICACIÓN ----------
st.markdown("---")
st.header("3. Análisis de Elección Discreta (Regresión Logística)")
st.markdown("""
A continuación, ajustamos un modelo Logit Binario para entender qué factores influyen en **la probabilidad de elegir Starbucks (frente a las otras opciones)**.
La variable dependiente es `1` si eligieron Starbucks y `0` en caso contrario. Esto replica conceptualmente los análisis del modelo de elección en la Tarea 1 Parte 2.
""")

df_reg = df_filtered.copy()

# Tratamiento de datos antes de la regresión
df_reg['Elige_Starbucks'] = (df_reg['CHOSEN'] == 1).astype(int)
df_reg = df_reg.dropna(subset=['Elige_Starbucks', 'PRICE_STA', 'TASTE_STA', 'AMB_STA', 'SNOB'])

# Construimos la ecuación del modelo
model_formula = 'Elige_Starbucks ~ PRICE_STA + TASTE_STA + AMB_STA + SNOB'

try:
    # Ajustamos modelo logit
    model = smf.logit(formula=model_formula, data=df_reg).fit(disp=0)
    
    col_reg1, col_reg2 = st.columns([1, 1])
    
    with col_reg1:
        st.subheader("Resultados Estadísticos del Modelo Logit")
        st.code(model.summary().as_text())
        
    with col_reg2:
        st.subheader("Interpretaciones Prácticas Paso a Paso")
        st.markdown(f"**Pseudo R-cuadrado ($Pseudo R^2$):** {model.prsquared:.4f}. Representa la mejora del modelo respecto a un modelo nulo aleatorio en la predicción de la elección.")
        
        st.markdown("### Coeficientes e Interpretación:")
        st.markdown("- **Intercepto:** Base de elección de la alternativa, asumiendo resto 0.")
        
        pvalues = model.pvalues
        coefs = model.params
        
        st.markdown("- **Precio de Starbucks (PRICE_STA):**")
        if pvalues['PRICE_STA'] < 0.05:
            direction = "aumenta" if coefs['PRICE_STA'] > 0 else "disminuye"
            st.success(f"Significativo (p-valor < 0.05). Un aumento en el precio **{direction}** la probabilidad de elegir Starbucks.")
        else:
            st.info(f"El precio no fue estadísticamente significativo (p-valor = {pvalues['PRICE_STA']:.3f}).")

        st.markdown("- **Sabor percibido (TASTE_STA):**")
        if pvalues['TASTE_STA'] < 0.05:
            direction = "aumenta" if coefs['TASTE_STA'] > 0 else "disminuye"
            st.success(f"Muy significativo. Una mejor percepción del sabor **{direction}** fuertemente las chances de comprar Starbucks.")
        else:
            st.info("La oferta de sabor no probó ser estadísticamente significativa.")
            
        st.markdown("- **Ambiente (AMB_STA):**")
        if pvalues['AMB_STA'] < 0.05:
            direction = "positiva" if coefs['AMB_STA'] > 0 else "negativa"
            st.success(f"Asociación **{direction}** y significativa. A mejor ambiente, mayor probabilidad de elegirla.")
        else:
            st.info("El ambiente no es determinante para elegir Starbucks en esta muestra.")

        st.markdown("""
        **Conclusión sobre el modelo de elección:** 
        La regresión Logit muestra qué atributos inclinan verdaderamente la balanza hacia la decisión de compra, permitiendo replicar la esencia de los modelos econométricos presentados en R utilizando Python.
        """)

except Exception as e:
    st.error(f"Se produjo un error al calcular la regresión. Esto puede suceder si la matriz no se puede invertir u otros problemas con los datos: {e}")

st.markdown("---")
st.info("""
**Nota metodológica:** 
Los resultados del modelo Logit indican asociación estadística y no necesariamente relaciones causales directas. Para establecer causalidad, se requeriría un diseño experimental.
""")


