import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer)
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Social Trend Analyzer",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Social Media Trend Analyzer")
st.markdown("Analyze trending topics, hashtags, "
            "sentiment and engagement patterns.")
st.markdown("---")

analyzer = SentimentIntensityAnalyzer()

# Generate social media dataset
@st.cache_data
def generate_data():
    np.random.seed(42)

    topics = {
        'Cricket':     ['#INDvAUS', '#ViratKohli',
                        '#IPL2024', '#T20WorldCup',
                        '#RohitSharma', '#TeamIndia'],
        'Technology':  ['#AI', '#ChatGPT',
                        '#Python', '#MachineLearning',
                        '#TechNews', '#Startup'],
        'Bollywood':   ['#SRK', '#Pathaan',
                        '#BollywoodNews', '#NewMovie',
                        '#BoxOffice', '#OTT'],
        'Politics':    ['#India', '#Budget2024',
                        '#Elections', '#BJP',
                        '#Congress', '#Modi'],
        'Finance':     ['#Sensex', '#Nifty',
                        '#StockMarket', '#Crypto',
                        '#MutualFunds', '#Investment'],
        'Education':   ['#UPSC', '#JEE',
                        '#NEET', '#StudyMotivation',
                        '#OnlineLearning', '#IIT'],
        'Environment': ['#ClimateChange',
                        '#GoGreen', '#SaveEarth',
                        '#Sustainability',
                        '#RenewableEnergy',
                        '#PlantTrees'],
        'Health':      ['#COVID', '#MentalHealth',
                        '#Fitness', '#Yoga',
                        '#Nutrition', '#Wellness']
    }

    sample_texts = {
        'Cricket': [
            "Kohli is absolutely BRILLIANT today! "
            "What a century! #ViratKohli #Cricket",
            "India wins! Amazing performance by "
            "the whole team #INDvAUS #TeamIndia",
            "IPL auction is so exciting this year "
            "record breaking bids #IPL2024",
            "Rohit Sharma playing like a legend "
            "today! #RohitSharma #T20WorldCup",
            "Disappointed with India's performance "
            "today. Need to do better #Cricket"
        ],
        'Technology': [
            "AI is transforming every industry. "
            "The future is here! #AI #Technology",
            "Just learned Python and built my "
            "first ML model! #Python #MachineLearning",
            "ChatGPT is insane. Can't believe "
            "how smart it is #ChatGPT #AI",
            "Indian startup ecosystem is booming "
            "with AI companies #Startup #India",
            "Machine learning jobs are everywhere "
            "right now #MachineLearning #TechJobs"
        ],
        'Finance': [
            "Sensex at all time high! My portfolio "
            "is up 20% #Sensex #StockMarket",
            "Crypto crashing again. Be careful "
            "with investments #Crypto #Bitcoin",
            "SIP in mutual funds is the best "
            "strategy for long term #MutualFunds",
            "Market is volatile. Don't panic "
            "sell #Nifty #Investment",
            "Budget 2024 great for the middle "
            "class! #Budget2024 #India"
        ],
        'Education': [
            "JEE results out! Congratulations "
            "to all students #JEE #IIT",
            "UPSC preparation is a marathon "
            "not a sprint. Stay consistent! #UPSC",
            "Online learning has made education "
            "so accessible #OnlineLearning #Education",
            "NEET results anxious waiting. "
            "Best of luck everyone #NEET",
            "Study motivation thread! Small steps "
            "lead to big success #StudyMotivation"
        ]
    }

    platforms = ['Twitter', 'Instagram',
                 'LinkedIn', 'YouTube',
                 'Facebook', 'Koo']
    platform_weights = [0.35, 0.25, 0.20,
                        0.08, 0.07, 0.05]

    rows = []
    for _ in range(1500):
        topic    = np.random.choice(
            list(topics.keys()))
        hashtag  = np.random.choice(
            topics[topic])
        platform = np.random.choice(
            platforms, p=platform_weights)

        sample_topic = np.random.choice(
            list(sample_texts.keys()))
        text = np.random.choice(
            sample_texts[sample_topic])

        # Add random hashtag to text
        text = text + f" {hashtag}"

        sentiment = analyzer.polarity_scores(
            text)['compound']
        if sentiment >= 0.05:
            sent_label = 'Positive'
        elif sentiment <= -0.05:
            sent_label = 'Negative'
        else:
            sent_label = 'Neutral'

        hour   = np.random.randint(0, 24)
        likes  = int(np.random.exponential(500))
        retweets = int(likes *
                       np.random.uniform(0.05,0.3))
        replies  = int(likes *
                       np.random.uniform(0.02,0.15))
        impressions = likes * \
            np.random.randint(5, 30)
        engagement  = (likes + retweets +
                       replies) / \
                      max(impressions, 1) * 100

        day = np.random.choice(
            ['Mon', 'Tue', 'Wed', 'Thu',
             'Fri', 'Sat', 'Sun'])

        rows.append({
            'topic':       topic,
            'hashtag':     hashtag,
            'platform':    platform,
            'text':        text,
            'sentiment':   sent_label,
            'sent_score':  round(sentiment, 3),
            'hour':        hour,
            'day':         day,
            'likes':       likes,
            'retweets':    retweets,
            'replies':     replies,
            'impressions': impressions,
            'engagement':  round(engagement, 3)
        })

    return pd.DataFrame(rows)

df = generate_data()

# Sidebar
st.sidebar.header("🔍 Filters")
topic_filter = st.sidebar.multiselect(
    "Topic:",
    df['topic'].unique(),
    default=df['topic'].unique()
)
platform_filter = st.sidebar.multiselect(
    "Platform:",
    df['platform'].unique(),
    default=df['platform'].unique()
)
sentiment_filter = st.sidebar.multiselect(
    "Sentiment:",
    df['sentiment'].unique(),
    default=df['sentiment'].unique()
)

filtered = df[
    (df['topic'].isin(topic_filter)) &
    (df['platform'].isin(platform_filter)) &
    (df['sentiment'].isin(sentiment_filter))
].copy()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 Trending",
    "😊 Sentiment",
    "📊 Engagement",
    "⏰ Timing",
    "☁️ Word Cloud"
])

# Tab 1 — Trending
with tab1:
    st.markdown("### 🔥 Trending Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Posts",
              f"{len(filtered):,}")
    c2.metric("Unique Hashtags",
              filtered['hashtag'].nunique())
    c3.metric("Total Likes",
              f"{filtered['likes'].sum():,}")
    c4.metric("Avg Engagement",
              f"{filtered['engagement'].mean():.2f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Top hashtags
        tag_counts = filtered[
            'hashtag'].value_counts().head(15)
        fig = px.bar(
            x=tag_counts.values,
            y=tag_counts.index,
            orientation='h',
            title='🔥 Top Trending Hashtags',
            color=tag_counts.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            height=450,
            template='plotly_white',
            xaxis_title='Post Count'
        )
        st.plotly_chart(fig,
                        use_container_width=True)

    with col2:
        # Topic distribution
        topic_counts = filtered[
            'topic'].value_counts()
        fig2 = px.pie(
            values=topic_counts.values,
            names=topic_counts.index,
            title='Posts by Topic',
            color_discrete_sequence=
                px.colors.qualitative.Set2
        )
        fig2.update_layout(height=450)
        st.plotly_chart(fig2,
                        use_container_width=True)

    # Platform breakdown
    st.markdown("#### 📱 Platform Performance")
    plat_stats = filtered.groupby(
        'platform').agg(
        posts=('text', 'count'),
        avg_likes=('likes', 'mean'),
        avg_engagement=('engagement', 'mean')
    ).reset_index().sort_values(
        'posts', ascending=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name='Posts',
        x=plat_stats['platform'],
        y=plat_stats['posts'],
        marker_color='#3498db'
    ))
    fig3.add_trace(go.Scatter(
        name='Avg Likes',
        x=plat_stats['platform'],
        y=plat_stats['avg_likes'],
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8)
    ))
    fig3.update_layout(
        title='Posts and Avg Likes by Platform',
        yaxis=dict(title='Posts'),
        yaxis2=dict(title='Avg Likes',
                    overlaying='y',
                    side='right'),
        height=350,
        template='plotly_white'
    )
    st.plotly_chart(fig3,
                    use_container_width=True)

# Tab 2 — Sentiment
with tab2:
    st.markdown("### 😊 Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        sent_counts = filtered[
            'sentiment'].value_counts()
        sent_colors = {
            'Positive': '#2ecc71',
            'Neutral':  '#f39c12',
            'Negative': '#e74c3c'
        }
        fig4 = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            title='Overall Sentiment',
            color=sent_counts.index,
            color_discrete_map=sent_colors
        )
        fig4.update_layout(height=350)
        st.plotly_chart(fig4,
                        use_container_width=True)

    with col2:
        topic_sent = filtered.groupby(
            ['topic', 'sentiment']
        ).size().reset_index(name='count')

        fig5 = px.bar(
            topic_sent,
            x='topic', y='count',
            color='sentiment',
            title='Sentiment by Topic',
            color_discrete_map=sent_colors,
            barmode='stack'
        )
        fig5.update_layout(
            height=350,
            template='plotly_white',
            xaxis_title='Topic',
            yaxis_title='Posts'
        )
        fig5.update_xaxes(tickangle=30)
        st.plotly_chart(fig5,
                        use_container_width=True)

    # Sentiment score distribution
    fig6 = px.histogram(
        filtered,
        x='sent_score',
        color='sentiment',
        title='Sentiment Score Distribution',
        nbins=30,
        color_discrete_map=sent_colors,
        barmode='overlay'
    )
    fig6.update_layout(
        height=300,
        template='plotly_white',
        xaxis_title='VADER Compound Score'
    )
    st.plotly_chart(fig6,
                    use_container_width=True)

    # Platform sentiment
    st.markdown(
        "#### 📱 Sentiment by Platform")
    plat_sent = filtered.groupby(
        ['platform', 'sentiment']
    ).size().unstack(fill_value=0)

    fig7 = px.imshow(
        plat_sent,
        title='Sentiment Heatmap: Platform × Sentiment',
        color_continuous_scale='RdYlGn',
        labels=dict(color='Posts')
    )
    fig7.update_layout(
        height=300,
        template='plotly_white'
    )
    st.plotly_chart(fig7,
                    use_container_width=True)

# Tab 3 — Engagement
with tab3:
    st.markdown("### 📊 Engagement Analytics")

    col1, col2 = st.columns(2)

    with col1:
        topic_eng = filtered.groupby(
            'topic').agg(
            avg_likes=('likes', 'mean'),
            avg_retweets=('retweets', 'mean'),
            avg_replies=('replies', 'mean')
        ).reset_index()

        fig8 = go.Figure()
        metrics = ['avg_likes', 'avg_retweets',
                   'avg_replies']
        colors  = ['#f39c12', '#3498db',
                   '#2ecc71']
        names   = ['Likes', 'Retweets',
                   'Replies']
        for m, c, n in zip(
            metrics, colors, names
        ):
            fig8.add_trace(go.Bar(
                name=n,
                x=topic_eng['topic'],
                y=topic_eng[m],
                marker_color=c
            ))
        fig8.update_layout(
            title='Avg Engagement by Topic',
            barmode='group',
            height=400,
            template='plotly_white',
            xaxis_title='Topic'
        )
        fig8.update_xaxes(tickangle=30)
        st.plotly_chart(fig8,
                        use_container_width=True)

    with col2:
        # Top performing posts
        top_posts = filtered.nlargest(
            10, 'likes')[
            ['hashtag', 'platform',
             'sentiment', 'likes',
             'retweets']
        ]
        st.markdown("#### 🏆 Top Posts by Likes")
        st.dataframe(top_posts,
                     use_container_width=True,
                     hide_index=True)

    # Engagement vs sentiment
    fig9 = px.box(
        filtered,
        x='sentiment',
        y='likes',
        color='sentiment',
        title='Like Distribution by Sentiment',
        color_discrete_map={
            'Positive': '#2ecc71',
            'Neutral':  '#f39c12',
            'Negative': '#e74c3c'
        }
    )
    fig9.update_layout(
        height=350,
        template='plotly_white',
        yaxis_title='Likes'
    )
    st.plotly_chart(fig9,
                    use_container_width=True)

# Tab 4 — Timing
with tab4:
    st.markdown("### ⏰ Posting Patterns")

    col1, col2 = st.columns(2)

    with col1:
        # Hourly activity
        hourly = filtered.groupby(
            'hour').agg(
            posts=('text', 'count'),
            avg_likes=('likes', 'mean')
        ).reset_index()

        fig10 = go.Figure()
        fig10.add_trace(go.Bar(
            x=hourly['hour'],
            y=hourly['posts'],
            name='Posts',
            marker_color='#3498db',
            opacity=0.7
        ))
        fig10.add_trace(go.Scatter(
            x=hourly['hour'],
            y=hourly['avg_likes'],
            name='Avg Likes',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='#e74c3c',
                      width=2)
        ))
        fig10.update_layout(
            title='Posting Activity by Hour',
            xaxis_title='Hour of Day',
            yaxis=dict(title='Posts'),
            yaxis2=dict(
                title='Avg Likes',
                overlaying='y',
                side='right'),
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig10,
                        use_container_width=True)

    with col2:
        # Day of week
        day_order = ['Mon', 'Tue', 'Wed',
                     'Thu', 'Fri', 'Sat', 'Sun']
        day_stats = filtered.groupby(
            'day').agg(
            posts=('text', 'count'),
            avg_likes=('likes', 'mean')
        ).reindex(day_order)

        fig11 = px.bar(
            x=day_stats.index,
            y=day_stats['posts'],
            title='Posts by Day of Week',
            color=day_stats['posts'],
            color_continuous_scale='Purples'
        )
        fig11.update_layout(
            height=400,
            template='plotly_white',
            yaxis_title='Post Count'
        )
        st.plotly_chart(fig11,
                        use_container_width=True)

    # Best time to post
    best_hour = hourly.loc[
        hourly['avg_likes'].idxmax(), 'hour']
    best_day  = day_stats[
        'avg_likes'].idxmax()

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Hour to Post",
              f"{best_hour}:00 hrs")
    c2.metric("Best Day to Post",
              best_day)
    c3.metric("Peak Engagement",
              f"{hourly['avg_likes'].max():.0f} likes")

# Tab 5 — Word Cloud
with tab5:
    st.markdown("### ☁️ Trending Word Cloud")

    selected_topic = st.selectbox(
        "Select topic for word cloud:",
        ['All Topics'] +
        df['topic'].unique().tolist()
    )

    if selected_topic == 'All Topics':
        texts = ' '.join(
            filtered['text'].tolist())
    else:
        texts = ' '.join(
            filtered[
                filtered['topic'] ==
                selected_topic
            ]['text'].tolist()
        )

    # Clean text
    texts = re.sub(r'http\S+', '', texts)
    texts = re.sub(r'[^a-zA-Z#\s]', '', texts)

    col1, col2 = st.columns(2)
    with col1:
        colormap = st.selectbox(
            "Color scheme:",
            ['viridis', 'plasma',
             'RdYlGn', 'Blues', 'Reds']
        )
    with col2:
        max_words = st.slider(
            "Max words:", 30, 200, 100)

    if st.button("☁️ Generate Word Cloud",
                 type="primary"):
        if len(texts.split()) > 5:
            wc = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap=colormap,
                max_words=max_words,
                collocations=False
            ).generate(texts)

            fig12, ax = plt.subplots(
                figsize=(12, 5))
            ax.imshow(wc,
                      interpolation='bilinear')
            ax.axis('off')
            ax.set_title(
                f'Word Cloud — '
                f'{selected_topic}',
                fontsize=14)
            plt.tight_layout()
            st.pyplot(fig12)

            # Top words
            words = [
                w for w in texts.split()
                if len(w) > 3
            ]
            top_words = Counter(
                words).most_common(10)
            st.markdown(
                "#### 🔤 Top Words")
            words_df = pd.DataFrame(
                top_words,
                columns=['Word', 'Count'])
            st.dataframe(
                words_df,
                use_container_width=True,
                hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Social Media Trend Analyzer | "
    "1500 posts analyzed across 6 platforms"
)