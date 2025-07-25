import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import re
import io
import docx
import PyPDF2
from collections import Counter
import heapq
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?]', '', text)
        return text.strip()
    
    def extractive_summary_frequency(self, text, num_sentences=3):
        """Frequency-based extractive summarization"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Tokenize and remove stopwords
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Calculate word frequencies
        word_freq = Counter(words)
        max_freq = max(word_freq.values())
        
        # Normalize frequencies
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Score sentences
        sentence_scores = {}
        for sentence in sentences:
            words_in_sentence = word_tokenize(sentence.lower())
            words_in_sentence = [word for word in words_in_sentence if word.isalnum()]
            
            score = 0
            word_count = 0
            for word in words_in_sentence:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
        
        # Get top sentences
        summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        
        # Sort by original order
        summary = []
        for sentence in sentences:
            if sentence in summary_sentences:
                summary.append(sentence)
        
        return ' '.join(summary)
    
    def extractive_summary_textrank(self, text, num_sentences=3):
        """TextRank-based extractive summarization"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create graph and apply PageRank
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Get top sentences
            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
            summary_sentences = [sentence for score, sentence in ranked_sentences[:num_sentences]]
            
            # Sort by original order
            summary = []
            for sentence in sentences:
                if sentence in summary_sentences:
                    summary.append(sentence)
            
            return ' '.join(summary)
        except:
            # Fallback to frequency-based if TextRank fails
            return self.extractive_summary_frequency(text, num_sentences)
    
    def abstractive_summary_simple(self, text, num_sentences=3):
        """Simple abstractive summarization using key phrases"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Extract key phrases using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20)
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            key_phrases = [feature_names[i] for i in mean_scores.argsort()[-10:][::-1]]
            
            # Create abstractive summary by combining key information
            summary_parts = []
            for phrase in key_phrases[:5]:
                for sentence in sentences:
                    if phrase.lower() in sentence.lower() and sentence not in summary_parts:
                        summary_parts.append(sentence)
                        break
                if len(summary_parts) >= num_sentences:
                    break
            
            return ' '.join(summary_parts[:num_sentences])
        except:
            # Fallback to extractive if abstractive fails
            return self.extractive_summary_frequency(text, num_sentences)

def extract_text_from_file(uploaded_file):
    """Extract text from various file formats"""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        else:
            st.error("Unsupported file format!")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def analyze_text(text):
    """Analyze text and return statistics"""
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    stats = {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
        'reading_time_minutes': len(words) / 200  # Assuming 200 words per minute
    }
    
    return stats

def create_word_frequency_chart(text):
    """Create word frequency visualization"""
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 3]
    
    word_freq = Counter(words).most_common(15)
    
    if word_freq:
        df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
        fig = px.bar(df, x='Frequency', y='Word', orientation='h',
                     title='Top 15 Most Frequent Words',
                     color='Frequency',
                     color_continuous_scale='viridis')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig
    return None

def main():
    st.set_page_config(
        page_title="AI Text Summarizer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🤖 AI Text Summarizer")
    st.markdown("Upload documents or paste text to generate intelligent summaries using various NLP techniques.")
    
    # Initialize summarizer
    summarizer = TextSummarizer()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    summary_method = st.sidebar.selectbox(
        "Choose Summarization Method:",
        ["Frequency-based Extractive", "TextRank Extractive", "Simple Abstractive"]
    )
    
    num_sentences = st.sidebar.slider(
        "Number of sentences in summary:",
        min_value=1,
        max_value=10,
        value=3
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # File upload
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files to summarize:",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True
        )
        
        # Text input
        st.subheader("Or Paste Text Directly")
        text_input = st.text_area(
            "Enter text to summarize:",
            height=200,
            placeholder="Paste your text here..."
        )
        
        # Process input
        all_text = ""
        
        if uploaded_files:
            st.subheader("📁 Uploaded Files")
            for uploaded_file in uploaded_files:
                with st.expander(f"📄 {uploaded_file.name}"):
                    file_text = extract_text_from_file(uploaded_file)
                    if file_text:
                        all_text += file_text + "\n\n"
                        st.text_area(
                            f"Content of {uploaded_file.name}:",
                            file_text[:500] + "..." if len(file_text) > 500 else file_text,
                            height=150,
                            disabled=True
                        )
        
        if text_input:
            all_text += text_input
    
    with col2:
        st.header("📊 Results")
        
        if all_text.strip():
            # Preprocess text
            processed_text = summarizer.preprocess_text(all_text)
            
            # Generate summary
            if summary_method == "Frequency-based Extractive":
                summary = summarizer.extractive_summary_frequency(processed_text, num_sentences)
            elif summary_method == "TextRank Extractive":
                summary = summarizer.extractive_summary_textrank(processed_text, num_sentences)
            else:  # Simple Abstractive
                summary = summarizer.abstractive_summary_simple(processed_text, num_sentences)
            
            # Display summary
            st.subheader("📋 Generated Summary")
            st.markdown(f"**Method:** {summary_method}")
            st.text_area(
                "Summary:",
                summary,
                height=200,
                disabled=True
            )
            
            # Download summary
            st.download_button(
                label="💾 Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
            
            # Text analysis
            st.subheader("📈 Text Analysis")
            
            original_stats = analyze_text(processed_text)
            summary_stats = analyze_text(summary)
            
            # Create metrics
            col_metrics1, col_metrics2 = st.columns(2)
            
            with col_metrics1:
                st.markdown("**Original Text:**")
                st.metric("Words", f"{original_stats['word_count']:,}")
                st.metric("Sentences", original_stats['sentence_count'])
                st.metric("Reading Time", f"{original_stats['reading_time_minutes']:.1f} min")
            
            with col_metrics2:
                st.markdown("**Summary:**")
                st.metric("Words", f"{summary_stats['word_count']:,}")
                st.metric("Sentences", summary_stats['sentence_count'])
                st.metric("Compression Ratio", 
                         f"{(1 - summary_stats['word_count']/original_stats['word_count'])*100:.1f}%")
            
            # Visualization
            st.subheader("📊 Word Frequency Analysis")
            chart = create_word_frequency_chart(processed_text)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Comparison
            if st.checkbox("Show Detailed Comparison"):
                st.subheader("🔍 Detailed Comparison")
                
                comparison_df = pd.DataFrame({
                    'Metric': ['Characters', 'Words', 'Sentences', 'Avg Words/Sentence', 'Reading Time (min)'],
                    'Original': [
                        f"{original_stats['character_count']:,}",
                        f"{original_stats['word_count']:,}",
                        original_stats['sentence_count'],
                        f"{original_stats['avg_words_per_sentence']:.1f}",
                        f"{original_stats['reading_time_minutes']:.1f}"
                    ],
                    'Summary': [
                        f"{summary_stats['character_count']:,}",
                        f"{summary_stats['word_count']:,}",
                        summary_stats['sentence_count'],
                        f"{summary_stats['avg_words_per_sentence']:.1f}",
                        f"{summary_stats['reading_time_minutes']:.1f}"
                    ]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
        
        else:
            st.info("👆 Please upload files or enter text to generate a summary.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit • NLP • Machine Learning</p>
            <p><strong>Supported formats:</strong> TXT, PDF, DOCX</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
