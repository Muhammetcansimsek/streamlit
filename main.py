import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import pingouin

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.feather'):
            df = pd.read_feather(file)
        else:
            st.error("Lüftfen csv veya feather formatında bir dosya yükleyin.")
            return None
        return df
        
    except Exception as e:
        st.error(f"Dosya yükleme hatası: {str(e)}")
        return None

def calculate_p_value(t_stat, df):
    return t.cdf(t_stat, df=df)

def perform_ttest(data, col1, col2, alpha=0.5):
    try:
        data['diff'] = data[col1] - data[col2]

        x_bar_diff = data['diff'].mean()
        s_diff = data['diff'].std()
        n_diff = len(data['diff'])
    
        t_stat = (x_bar_diff - 0) / np.sqrt(s_diff**2 / n_diff)

        degrees_of_freedom = n_diff - 1 # Paired test (df = n - 1)

        p_value = calculate_p_value(t_stat, degrees_of_freedom)

        pingouin.ttest(x=data[col1],
                       y=data[col2],
                       paired=True,
                       alternative='less')
        return {
            't_stat': t_stat,
            'p_value': p_value,
            'df': degrees_of_freedom,
            'mean_diff': x_bar_diff,
            'std_diff': s_diff
        }
    except Exception as e:
        st.error(f"Test error: {str(e)}")
        return None

def main():
    st.title("İstatistiksel Analiz Uygulaması")

    significance_levels = {
        "%1": 0.01,
        "%5": 0.05,
        "%10": 0.1
    }

    uploaded_file = st.file_uploader("Veri dosyasını yükleyin (.csv ve .feather uzantılı olmalı)", type=['csv', 'feather'])
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("Veri seti önizleme:")
            st.dataframe(df.head())

            col1 = st.selectbox("Birinci sütunu seçin:", df.columns)
            col2 = st.selectbox("İkinci sütunu seçin:", df.columns)

            alpha = st.selectbox(
                "Anlamlılık düzeyi (α) seçin:",
                options = list(significance_levels.keys()),
                index=1
            )
            alpha_value = significance_levels[alpha]

            if st.button("Analizi Başlat"):
                if not (pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2])):
                    st.warning("Seçilen sütunlar sayısal değil! Lütfen sayısal sütunlar girin.")
                else:
                    results = perform_ttest(df, col1, col2, alpha=alpha_value)

                    if results is not None:
                        st.write(f"T-istatistiği: {results['t_stat']:.4f}")
                        st.write(f"Serbestlik Derecesi: {results['df']}")
                        st.write(f"P-değeri: {results['p_value']}")

                        if results['p_value'] <= alpha_value:
                            st.write(f"P-değeri ({results['p_value']}) ≤ α ({alpha_value}) olduğundan sıfır hipotezi reddedilir.")
                            st.write("Gruplar arasında istatistiksel olarak anlamlı bir fark vardır.")

                        else:
                            st.write(f"P-değeri ({results['p_value']}) > α ({alpha_value}) olduğundan sıfır hipotezi reddedilemez.")
                            st.write("Gruplar arasında istatistiksel olarak anlamlı bir fark yoktur.")

                        fig, ax = plt.subplots()
                        ax.hist(df[col1], alpha=0.5, label=col1)
                        ax.hist(df[col2], alpha=0.5, label=col2)
                        ax.legend()
                        ax.set_title("Veri Dağılımı")
                        st.pyplot(fig)

if __name__ == "__main__":
    main()
        
        