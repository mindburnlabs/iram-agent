"""
Instagram Research Agent MCP (IRAM) - Analysis Module

This module handles content analysis including NLP for text and computer vision for media.
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

# NLP imports - make them optional for minimal deployment
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: Transformers not available: {e}. NLP will use basic fallback.")
    TRANSFORMERS_AVAILABLE = False
    pipeline, AutoTokenizer, AutoModel, torch = None, None, None, None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: SentenceTransformers not available: {e}. Embeddings disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Try to import BERTopic, fallback if not available
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: BERTopic not available: {e}. Topic modeling will use fallback.")
    BERTOPIC_AVAILABLE = False
    BERTopic = None

# Computer vision imports - disabled for minimal deployment
# import cv2
# from PIL import Image

# Data processing
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


class ContentAnalyzer:
    """Main content analysis class for Instagram content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the content analyzer."""
        self.config = config or {}
        
        # Initialize NLP models
        self._init_nlp_models()
        
        # Initialize CV models
        self._init_cv_models()
        
        logger.info("Content analyzer initialized")
    
    def _init_nlp_models(self):
        """Initialize NLP models."""
        try:
            # Sentiment analysis
            if TRANSFORMERS_AVAILABLE:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
            else:
                self.sentiment_analyzer = None
            
            # Embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            else:
                self.sentence_model = None
            
            # Topic modeling
            self.topic_model = None  # Initialized on first use
            
            logger.info(f"NLP models initialized - Transformers: {TRANSFORMERS_AVAILABLE}, Embeddings: {SENTENCE_TRANSFORMERS_AVAILABLE}")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
            # Fallback models or disable features
            self.sentiment_analyzer = None
            self.sentence_model = None
    
    def _init_cv_models(self):
        """Initialize computer vision models."""
        try:
            # Object detection would be initialized here
            # For now, we'll use basic image processing
            self.cv_models_loaded = True
            logger.info("CV models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize CV models: {e}")
            self.cv_models_loaded = False
    
    def analyze_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of text content."""
        try:
            if not self.sentiment_analyzer:
                return self._analyze_sentiment_fallback(data)
            
            texts = self._extract_texts(data)
            if not texts:
                return {"error": "No text content found"}
            
            results = []
            sentiment_scores = {"positive": [], "negative": [], "neutral": []}
            
            for text in texts:
                if len(text.strip()) < 3:  # Skip very short texts
                    continue
                    
                # Analyze sentiment
                scores = self.sentiment_analyzer(text[:512])  # Truncate to model limit
                
                # Convert to standardized format
                sentiment_result = {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": scores[0]['label'].lower(),
                    "confidence": scores[0]['score'],
                    "all_scores": {score['label'].lower(): score['score'] for score in scores}
                }
                
                results.append(sentiment_result)
                
                # Aggregate scores
                for score in scores:
                    label = score['label'].lower()
                    if label in sentiment_scores:
                        sentiment_scores[label].append(score['score'])
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(sentiment_scores)
            
            return {
                "analysis_type": "sentiment",
                "individual_results": results,
                "overall_sentiment": overall_sentiment,
                "total_texts": len(results),
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_texts(self, data: Dict[str, Any]) -> List[str]:
        """Extract text content from various data structures."""
        texts = []
        
        try:
            # Handle different data structures
            if isinstance(data, dict):
                # Profile data
                if "biography" in data:
                    texts.append(data["biography"])
                
                # Posts data
                if "posts" in data:
                    for post in data["posts"]:
                        if "caption" in post and post["caption"]:
                            texts.append(post["caption"])
                
                # Direct text
                if "text" in data:
                    texts.append(data["text"])
                
                # Caption
                if "caption" in data:
                    texts.append(data["caption"])
            
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        texts.extend(self._extract_texts(item))
                    elif isinstance(item, str):
                        texts.append(item)
            
            elif isinstance(data, str):
                texts.append(data)
            
            # Clean texts
            cleaned_texts = []
            for text in texts:
                if text and isinstance(text, str):
                    cleaned = text.strip()
                    if len(cleaned) > 0:
                        cleaned_texts.append(cleaned)
            
            return cleaned_texts
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return []
    
    def _calculate_overall_sentiment(self, sentiment_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate overall sentiment from individual scores."""
        try:
            overall = {}
            
            for sentiment, scores in sentiment_scores.items():
                if scores:
                    overall[sentiment] = {
                        "average": np.mean(scores),
                        "count": len(scores)
                    }
                else:
                    overall[sentiment] = {"average": 0.0, "count": 0}
            
            # Determine dominant sentiment
            if overall:
                dominant = max(overall.keys(), key=lambda k: overall[k]["average"])
                overall["dominant_sentiment"] = dominant
                overall["confidence"] = overall[dominant]["average"]
            
            return overall
            
        except Exception as e:
            logger.error(f"Overall sentiment calculation failed: {e}")
            return {}
    
    def _analyze_sentiment_fallback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic sentiment analysis using simple keyword matching."""
        try:
            texts = self._extract_texts(data)
            if not texts:
                return {"error": "No text content found"}
            
            positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'like', 'happy', 'excellent', 'wonderful']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'horrible', 'disgusting']
            
            results = []
            overall_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            for text in texts:
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count:
                    sentiment = 'positive'
                    confidence = min(0.8, 0.5 + positive_count * 0.1)
                    overall_scores['positive'] += 1
                elif negative_count > positive_count:
                    sentiment = 'negative'
                    confidence = min(0.8, 0.5 + negative_count * 0.1)
                    overall_scores['negative'] += 1
                else:
                    sentiment = 'neutral'
                    confidence = 0.6
                    overall_scores['neutral'] += 1
                
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "method": "keyword_fallback"
                })
            
            total_texts = len(results)
            overall_sentiment = {
                "positive": {"average": overall_scores['positive'] / total_texts, "count": overall_scores['positive']},
                "negative": {"average": overall_scores['negative'] / total_texts, "count": overall_scores['negative']},
                "neutral": {"average": overall_scores['neutral'] / total_texts, "count": overall_scores['neutral']}
            }
            
            dominant = max(overall_sentiment.keys(), key=lambda k: overall_sentiment[k]["average"])
            overall_sentiment["dominant_sentiment"] = dominant
            overall_sentiment["confidence"] = overall_sentiment[dominant]["average"]
            
            return {
                "analysis_type": "sentiment_fallback",
                "individual_results": results,
                "overall_sentiment": overall_sentiment,
                "total_texts": len(results),
                "note": "Using fallback sentiment analysis (Transformers unavailable)",
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fallback sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    def extract_topics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract topics from text content using BERTopic or fallback method."""
        try:
            texts = self._extract_texts(data)
            if len(texts) < 3:  # Need minimum texts for topic modeling
                return {"error": "Insufficient text data for topic modeling"}
            
            if not BERTOPIC_AVAILABLE:
                # Fallback to simple keyword-based topic extraction
                return self._extract_topics_fallback(texts)
            
            # Initialize topic model if needed
            if not self.topic_model:
                self.topic_model = BERTopic(
                    embedding_model=self.sentence_model,
                    min_topic_size=2,
                    calculate_probabilities=True
                )
            
            # Fit the model and extract topics
            topics, probabilities = self.topic_model.fit_transform(texts)
            
            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            
            # Extract keywords for each topic
            topic_keywords = {}
            for topic_id in topic_info['Topic']:
                if topic_id != -1:  # Skip outlier topic
                    keywords = self.topic_model.get_topic(topic_id)
                    topic_keywords[topic_id] = [word for word, _ in keywords[:10]]
            
            return {
                "analysis_type": "topics",
                "topics": topic_keywords,
                "topic_distribution": topics.tolist(),
                "topic_info": topic_info.to_dict('records'),
                "total_texts": len(texts),
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return {"error": str(e)}
    
    def _extract_topics_fallback(self, texts: List[str]) -> Dict[str, Any]:
        """Fallback topic extraction using simple keyword frequency."""
        try:
            from collections import Counter
            import string
            
            # Simple text processing
            all_words = []
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            for text in texts:
                # Clean and tokenize
                text = text.lower().translate(str.maketrans('', '', string.punctuation))
                words = [word for word in text.split() if len(word) > 3 and word not in stop_words]
                all_words.extend(words)
            
            # Find most common words as "topics"
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(20)
            
            # Create simple topic groupings
            topics = {
                0: [word for word, count in top_words[:5]],
                1: [word for word, count in top_words[5:10]],
                2: [word for word, count in top_words[10:15]]
            }
            
            return {
                "analysis_type": "topics_fallback",
                "topics": topics,
                "note": "Using fallback topic extraction (BERTopic unavailable)",
                "top_keywords": dict(top_words),
                "total_texts": len(texts),
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fallback topic extraction failed: {e}")
            return {"error": str(e)}
    
    def analyze_engagement(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement patterns in the data."""
        try:
            engagement_data = []
            
            # Extract engagement metrics
            if isinstance(data, dict):
                if "posts" in data:
                    for post in data["posts"]:
                        engagement_data.append(self._extract_post_engagement(post))
                elif "like_count" in data or "comment_count" in data:
                    engagement_data.append(self._extract_post_engagement(data))
            
            if not engagement_data:
                return {"error": "No engagement data found"}
            
            # Calculate engagement statistics
            df = pd.DataFrame(engagement_data)
            
            stats = {
                "total_posts": len(engagement_data),
                "avg_likes": df['likes'].mean() if 'likes' in df.columns else 0,
                "avg_comments": df['comments'].mean() if 'comments' in df.columns else 0,
                "total_engagement": df['total_engagement'].sum() if 'total_engagement' in df.columns else 0,
                "engagement_rate": df['engagement_rate'].mean() if 'engagement_rate' in df.columns else 0,
                "top_posts": df.nlargest(3, 'total_engagement').to_dict('records') if 'total_engagement' in df.columns else []
            }
            
            # Engagement trends over time
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # Calculate rolling averages
                df['rolling_engagement'] = df['total_engagement'].rolling(window=7, min_periods=1).mean()
                
                stats["engagement_trend"] = {
                    "trend": "increasing" if df['rolling_engagement'].iloc[-1] > df['rolling_engagement'].iloc[0] else "decreasing",
                    "recent_avg": df['rolling_engagement'].tail(7).mean(),
                    "overall_avg": df['total_engagement'].mean()
                }
            
            return {
                "analysis_type": "engagement",
                "statistics": stats,
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Engagement analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_post_engagement(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Extract engagement metrics from a single post."""
        try:
            likes = post.get("like_count", 0) or 0
            comments = post.get("comment_count", 0) or 0
            
            # Calculate total engagement
            total_engagement = likes + comments
            
            # Estimate engagement rate (simplified)
            # In real scenario, you'd need follower count
            follower_estimate = 10000  # Placeholder
            engagement_rate = (total_engagement / follower_estimate) * 100 if follower_estimate > 0 else 0
            
            return {
                "post_id": post.get("id", ""),
                "likes": likes,
                "comments": comments,
                "total_engagement": total_engagement,
                "engagement_rate": engagement_rate,
                "timestamp": post.get("taken_at", "")
            }
            
        except Exception as e:
            logger.error(f"Post engagement extraction failed: {e}")
            return {"likes": 0, "comments": 0, "total_engagement": 0, "engagement_rate": 0}
    
    def analyze_hashtags(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze hashtag usage patterns."""
        try:
            texts = self._extract_texts(data)
            
            # Extract hashtags from texts
            all_hashtags = []
            hashtag_pattern = r'#\w+'
            
            for text in texts:
                hashtags = re.findall(hashtag_pattern, text, re.IGNORECASE)
                all_hashtags.extend([tag.lower() for tag in hashtags])
            
            if not all_hashtags:
                return {"error": "No hashtags found"}
            
            # Count hashtag frequency
            hashtag_counts = {}
            for hashtag in all_hashtags:
                hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + 1
            
            # Sort by frequency
            sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "analysis_type": "hashtags",
                "total_hashtags": len(all_hashtags),
                "unique_hashtags": len(hashtag_counts),
                "top_hashtags": sorted_hashtags[:20],
                "hashtag_distribution": dict(sorted_hashtags),
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Hashtag analysis failed: {e}")
            return {"error": str(e)}
    
    def comprehensive_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis combining multiple methods."""
        try:
            results = {
                "analysis_type": "comprehensive",
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
            # Sentiment analysis
            sentiment_result = self.analyze_sentiment(data)
            if "error" not in sentiment_result:
                results["sentiment_analysis"] = sentiment_result
            
            # Topic extraction
            topics_result = self.extract_topics(data)
            if "error" not in topics_result:
                results["topic_analysis"] = topics_result
            
            # Engagement analysis
            engagement_result = self.analyze_engagement(data)
            if "error" not in engagement_result:
                results["engagement_analysis"] = engagement_result
            
            # Hashtag analysis
            hashtag_result = self.analyze_hashtags(data)
            if "error" not in hashtag_result:
                results["hashtag_analysis"] = hashtag_result
            
            # Generate insights
            insights = self._generate_insights(results)
            results["insights"] = insights
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from analysis results."""
        insights = []
        
        try:
            # Sentiment insights
            if "sentiment_analysis" in analysis_results:
                sentiment = analysis_results["sentiment_analysis"]
                if "overall_sentiment" in sentiment:
                    dominant = sentiment["overall_sentiment"].get("dominant_sentiment", "")
                    confidence = sentiment["overall_sentiment"].get("confidence", 0)
                    
                    if dominant == "positive" and confidence > 0.7:
                        insights.append("Content shows strong positive sentiment - continue current strategy")
                    elif dominant == "negative" and confidence > 0.7:
                        insights.append("Content shows negative sentiment - consider adjusting messaging")
            
            # Engagement insights
            if "engagement_analysis" in analysis_results:
                engagement = analysis_results["engagement_analysis"]
                if "statistics" in engagement:
                    stats = engagement["statistics"]
                    avg_engagement = stats.get("total_engagement", 0)
                    
                    if avg_engagement > 1000:
                        insights.append("High engagement levels - content resonating well with audience")
                    elif avg_engagement < 100:
                        insights.append("Low engagement - consider improving content quality or posting times")
            
            # Hashtag insights
            if "hashtag_analysis" in analysis_results:
                hashtags = analysis_results["hashtag_analysis"]
                unique_ratio = hashtags.get("unique_hashtags", 0) / max(hashtags.get("total_hashtags", 1), 1)
                
                if unique_ratio > 0.8:
                    insights.append("High hashtag diversity - good for reach")
                elif unique_ratio < 0.3:
                    insights.append("Limited hashtag variety - consider expanding hashtag strategy")
            
            return insights if insights else ["Analysis completed - review detailed results for specific insights"]
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return ["Unable to generate insights"]