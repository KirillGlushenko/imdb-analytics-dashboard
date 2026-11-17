
from pathlib import Path


import re
from collections import Counter
import plotly.express as px
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="IMDb Analytics Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 IMDb Analytics Dashboard")
st.write("Интерактивный анализ отзывов IMDb")

st.divider()


@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "imdb_clean.csv"
    return pd.read_csv(data_path)


try:
    df = load_data()
except FileNotFoundError:
    st.error("Файл imdb_clean.csv не найден в папке проекта.")
    st.stop()

# Рассчитываем реальные показатели
total_reviews = len(df)
positive_reviews = (df["sentiment"] == "positive").sum()
negative_reviews = (df["sentiment"] == "negative").sum()

positive_percent = positive_reviews / total_reviews * 100
negative_percent = negative_reviews / total_reviews * 100

st.header("📊 Общая статистика")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Всего отзывов", f"{total_reviews:,}")

with col2:
    st.metric(
        "Положительных",
        f"{positive_reviews:,}",
        f"{positive_percent:.1f}%"
    )

with col3:
    st.metric(
        "Отрицательных",
        f"{negative_reviews:,}",
        f"{negative_percent:.1f}%"
    )

st.divider()

st.subheader("📋 Пример данных")
st.dataframe(
    df.head(10),
    use_container_width=True
)

st.success("Данные IMDb успешно загружены!")

st.divider()
st.header("📈 Визуализация данных")

col1, col2 = st.columns(2)

with col1:
    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .reset_index()
    )
    sentiment_counts.columns = ["sentiment", "count"]

    fig_pie = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        title="Распределение тональности",
        hole=0.4
    )

    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_hist = px.histogram(
        df,
        x="review_length",
        nbins=50,
        title="Распределение длины отзывов",
        labels={"review_length": "Длина отзыва (символы)"}
    )

    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()
st.header("🔎 Поиск и фильтрация отзывов")

with st.sidebar:
    st.header("🎛️ Фильтры")

    sentiment_filter = st.selectbox(
        "Тональность отзыва",
        ["Все", "Положительные", "Отрицательные"]
    )

    search_query = st.text_input(
        "Поиск по тексту отзыва",
        placeholder="Введите слово или фразу..."
    )

filtered_df = df.copy()

if sentiment_filter == "Положительные":
    filtered_df = filtered_df[
        filtered_df["sentiment"] == "positive"
    ]

elif sentiment_filter == "Отрицательные":
    filtered_df = filtered_df[
        filtered_df["sentiment"] == "negative"
    ]

if search_query.strip():
    filtered_df = filtered_df[
        filtered_df["review"].str.contains(
            search_query.strip(),
            case=False,
            na=False,
            regex=False
        )
    ]

st.metric("Найдено отзывов", len(filtered_df))

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)


st.divider()
st.header("📝 Анализ популярных слов")

stop_words = {
    "one", "all", "who", "about", "out",
    "there", "very", "what", "more",
    "when", "some", "just", "can",
    "would", "could", "should", "were",
    "been", "than", "then", "them",
    "only", "also", "even", "into",
    "the", "and", "a", "to", "of", "in", "is", "it",
    "this", "that", "was", "for", "with", "as",
    "on", "but", "are", "be", "have", "has",
    "at", "by", "an", "from", "not", "or",
    "i", "you", "he", "she", "they", "we",
    "my", "me", "his", "her", "their", "its"
}


@st.cache_data
def get_top_words(texts, top_n=15):
    text = " ".join(texts.astype(str)).lower()
    words = re.findall(r"\b[a-z]{3,}\b", text)

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    return Counter(filtered_words).most_common(top_n)

sentiment_words = st.selectbox(
    "Выберите категорию для анализа слов",
    ["positive", "negative"]
)

texts = df[df["sentiment"] == sentiment_words]["review"]

top_words = get_top_words(texts)

words_df = pd.DataFrame(
    top_words,
    columns=["Слово", "Частота"]
)

fig_words = px.bar(
    words_df,
    x="Частота",
    y="Слово",
    orientation="h",
    title=f"Топ-15 слов в {sentiment_words} отзывах"
)

fig_words.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(fig_words, use_container_width=True)

st.divider()
st.header("⚖️ Сравнение лексики отзывов")

compare_words = [
    "good", "bad", "great", "worst",
    "excellent", "boring", "amazing", "terrible"
]

@st.cache_data
def compare_word_frequencies(data, words):
    results = []

    for sentiment in ["positive", "negative"]:
        texts = data.loc[
            data["sentiment"] == sentiment, "review"
        ].astype(str).str.lower()

        for word in words:
            count = texts.str.count(
                rf"\b{re.escape(word)}\b"
            ).sum()

            results.append({
                "Слово": word,
                "Тональность": sentiment,
                "Частота": int(count)
            })

    return pd.DataFrame(results)


comparison_df = compare_word_frequencies(
    df, tuple(compare_words)
)

fig_comparison = px.bar(
    comparison_df,
    x="Слово",
    y="Частота",
    color="Тональность",
    barmode="group",
    title="Частота слов в положительных и отрицательных отзывах",
    labels={"Частота": "Количество упоминаний"}
)

st.plotly_chart(fig_comparison, use_container_width=True)

st.divider()
st.header("📥 Экспорт данных")

st.write(
    "Вы можете скачать отзывы, выбранные с помощью "
    "фильтров и поиска, в формате CSV."
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="📥 Скачать отфильтрованные отзывы",
    data=csv_data,
    file_name="imdb_filtered_reviews.csv",
    mime="text/csv"
)

st.divider()
st.header("💡 Аналитические выводы")

average_length = df["review_length"].mean()
median_length = df["review_length"].median()

st.write(
    f"В датасете содержится **{total_reviews:,} отзывов**. "
    f"Из них **{positive_percent:.1f}% положительных** "
    f"и **{negative_percent:.1f}% отрицательных**."
)

st.write(
    f"Средняя длина отзыва составляет "
    f"**{average_length:.0f} символов**, "
    f"а медианная — **{median_length:.0f} символов**."
)

if positive_percent > negative_percent:
    st.info(
        "Положительные отзывы немного преобладают "
        "над отрицательными."
    )
elif negative_percent > positive_percent:
    st.info(
        "Отрицательные отзывы немного преобладают "
        "над положительными."
    )
else:
    st.info("Количество положительных и отрицательных отзывов одинаковое.")
