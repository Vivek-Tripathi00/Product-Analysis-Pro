import os
import re
import time
import sqlite3
import atexit
from datetime import datetime
from functools import lru_cache


import requests
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from flask_mail import Mail, Message
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback-secret-key-for-dev')

# Configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'false').lower() == 'true'

# Initialize extensions
mail = Mail(app)

# NLTK Downloads
nltk.download("vader_lexicon", quiet=True)
nltk.download('stopwords', quiet=True)

# Database Setup
def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect('price_alerts.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS price_alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT NOT NULL,
                  product_url TEXT NOT NULL,
                  target_price REAL NOT NULL,
                  current_price REAL,
                  last_checked TEXT,
                  is_active INTEGER DEFAULT 1)''')
    conn.commit()
    conn.close()

init_db()

# Initialize Scheduler
scheduler = BackgroundScheduler(daemon=True)

# Price Tracking Functions
def track_price(url):
    """Track product price from e-commerce URLs."""
    try:
        html_content = get_page_content(url, 'generic')
        if not html_content:
            return None
            
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Amazon price extraction
        price_span = (soup.find("span", class_="a-offscreen") or 
                     soup.find("span", class_="a-price-whole"))
        
        if price_span:
            price_text = price_span.get_text(strip=True)
            price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
            if price_match:
                return float(price_match.group())
        
        # Flipkart price extraction
        price_div = (soup.find("div", class_="Nx9bqj") or 
                    soup.find("div", class_="_30jeq3") or
                    soup.find("div", class_="_16Jk6d") or
                    soup.find("div", class_="_1vC4OE"))
        
        if price_div:
            price_text = price_div.get_text(strip=True)
            price_match = re.search(r'[\d,]+\.?\d*', price_text.replace('₹', '').replace(',', ''))
            if price_match:
                return float(price_match.group())
            
        return None
    except Exception as e:
        app.logger.error(f"Error extracting price: {e}")
        return None

def check_price_alerts():
    """Check price alerts and send notifications."""
    with app.app_context():
        conn = None
        try:
            conn = sqlite3.connect('price_alerts.db')
            c = conn.cursor()
            c.execute("SELECT * FROM price_alerts WHERE is_active = 1")
            alerts = c.fetchall()
            
            for alert in alerts:
                alert_id, email, product_url, target_price, _, _, _ = alert
                current_price = track_price(product_url)
                
                if current_price and current_price <= target_price:
                    try:
                        msg = Message('Price Alert!',
                                    recipients=[email])
                        msg.body = (f"The price for your tracked product has dropped to ₹{current_price} "
                                    f"(your target: ₹{target_price}).\n\nProduct URL: {product_url}")
                        mail.send(msg)
                        c.execute("UPDATE price_alerts SET is_active = 0 WHERE id = ?", (alert_id,))
                        conn.commit()
                    except Exception as e:
                        app.logger.error(f"Error sending price alert email: {e}")
        except Exception as e:
            app.logger.error(f"Error in check_price_alerts: {e}")
        finally:
            if conn:
                conn.close()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
}

response = requests.get(url, headers=headers)
# Start scheduler
if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    scheduler.add_job(
        func=check_price_alerts,
        trigger='interval',
        hours=6,
        misfire_grace_time=3600
    )
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())

# Web Scraping Functions
# Updated imports at the top
import random
from fake_useragent import UserAgent

# Initialize UserAgent
ua = UserAgent()

# Updated get_page_content function
def get_page_content(url, platform):
    """Get HTML content with better headers and timeout handling."""
    headers = {
        "User-Agent": ua.random,  # Rotating user agents
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Referer": "https://www.google.com/",
        "DNT": "1"
    }
    
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=(10, 30),  # Connect timeout 10s, read timeout 30s
            proxies={
                'http': os.getenv('PROXY'),
                'https': os.getenv('PROXY')
            } if os.getenv('PROXY') else None
        )
        response.raise_for_status()
        
        if "captcha" in response.text.lower():
            raise Exception("CAPTCHA detected")
            
        return response.text
    except Exception as e:
        app.logger.error(f"Request failed: {str(e)}")
        return None

# Updated scrape_flipkart_reviews function
def scrape_flipkart_reviews(url, num_reviews=50):
    """Scrape Flipkart reviews with current selectors."""
    reviews = []
    page = 1
    
    while len(reviews) < num_reviews:
        try:
            # Random delay to avoid detection
            time.sleep(random.uniform(2, 5))
            
            page_url = f"{url}&page={page}"
            app.logger.info(f"Scraping page {page}: {page_url}")
            
            html_content = get_page_content(page_url, 'flipkart')
            if not html_content:
                break
                
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Updated review container selectors
            review_containers = soup.select('div._1AtVbE, div.col._2wzgFH, div.cPHDOP')
            if not review_containers:
                app.logger.warning("No review containers found - may need updated selectors")
                break
                
            for container in review_containers:
                try:
                    # Rating (try multiple selectors)
                    rating_element = (container.find('div', class_='_3LWZlK') or 
                                    container.find('span', class_='_2_R_DZ'))
                    rating = int(float(rating_element.text)) if rating_element else 0
                    
                    # Title
                    title_element = container.find('p', class_='_2-N8zT')
                    title = title_element.text.strip() if title_element else "No Title"
                    
                    # Review text
                    review_element = container.find('div', class_='t-ZTKy')
                    review_text = review_element.text.strip() if review_element else "No Review"
                    
                    # Reviewer info
                    reviewer_element = container.find('p', class_='_2sc7ZR')
                    reviewer = reviewer_element.text.strip() if reviewer_element else "Anonymous"
                    
                    # Date
                    date_element = container.find('p', class_='_2mcZGG')
                    date = date_element.text.strip() if date_element else "Unknown Date"
                    
                    reviews.append({
                        "Rating": rating,
                        "Title": title,
                        "Review": review_text,
                        "Reviewer": reviewer,
                        "Date": date,
                        "Platform": "Flipkart"
                    })
                    
                    if len(reviews) >= num_reviews:
                        break
                        
                except Exception as e:
                    app.logger.error(f"Error parsing review: {str(e)}")
                    continue
            
            # Check for next page
            next_button = soup.find('a', class_='_1LKTO3')
            if not next_button or 'disabled' in next_button.get('class', []):
                break
                
            page += 1
            
        except Exception as e:
            app.logger.error(f"Error scraping page {page}: {str(e)}")
            break
            
    return reviews
def scrape_amazon_reviews(url, num_reviews=50):
    """Scrape product reviews from Amazon."""
    reviews = []
    page = 1

    while len(reviews) < num_reviews:
        page_url = f"{url}&pageNumber={page}"
        app.logger.info(f"Scraping: {page_url}")
        
        try:
            html_content = get_page_content(page_url, 'amazon')
            if not html_content:
                break

            soup = BeautifulSoup(html_content, "html.parser")
            review_blocks = soup.find_all("div", {"data-hook": "review"})

            if not review_blocks:
                app.logger.info("No reviews found on this page.")
                break

            for review_block in review_blocks:
                try:
                    rating_tag = review_block.find("i", {"data-hook": "review-star-rating"})
                    rating_text = rating_tag.get_text(strip=True) if rating_tag else "0 out of 5 stars"
                    rating_match = re.search(r'(\d+\.?\d*) out of 5 stars', rating_text)
                    rating = int(float(rating_match.group(1))) if rating_match else 0

                    title_tag = review_block.find("a", {"data-hook": "review-title"})
                    title = title_tag.get_text(strip=True) if title_tag else "No Title"

                    review_text_tag = review_block.find("span", {"data-hook": "review-body"})
                    review_text = review_text_tag.get_text(strip=True) if review_text_tag else "No Review Text"

                    name_tag = review_block.find("span", class_="a-profile-name")
                    reviewer_name = name_tag.get_text(strip=True) if name_tag else "Anonymous"

                    date_tag = review_block.find("span", {"data-hook": "review-date"})
                    date = date_tag.get_text(strip=True) if date_tag else "Unknown Date"

                    verified_tag = review_block.find("span", {"data-hook": "avp-badge"})
                    verified = bool(verified_tag)

                    reviews.append({
                        "Rating": rating,
                        "Title": title,
                        "Review": review_text,
                        "Reviewer": reviewer_name,
                        "Location": "Amazon",
                        "Date": date,
                        "Verified": verified,
                        "Platform": "Amazon"
                    })

                    if len(reviews) >= num_reviews:
                        break

                except Exception as e:
                    app.logger.error(f"Error extracting review: {e}")

            page += 1
            time.sleep(1)  # Be polite with requests

        except Exception as e:
            app.logger.error(f"Error scraping page: {e}")
            break

    return reviews

# Sentiment Analysis Functions
@lru_cache(maxsize=128)
def get_sentiment_analyzer():
    """Get cached sentiment analyzer instance."""
    return SentimentIntensityAnalyzer()

def analyze_reviews(reviews):
    """Perform sentiment analysis on reviews."""
    sia = get_sentiment_analyzer()
    sentiment_results = {"Positive": 0, "Negative": 0, "Neutral": 0}
    word_list = []
    review_texts = []
    
    for review in reviews:
        text = review["Review"]
        sentiment_score = sia.polarity_scores(text)["compound"]
        review["SentimentScore"] = sentiment_score

        if sentiment_score >= 0.05:
            sentiment_results["Positive"] += 1
            review["Sentiment"] = "Positive"
        elif sentiment_score <= -0.05:
            sentiment_results["Negative"] += 1
            review["Sentiment"] = "Negative"
        else:
            sentiment_results["Neutral"] += 1
            review["Sentiment"] = "Neutral"

        word_list.extend(re.findall(r"\b[a-z]{3,}\b", text.lower()))
        review_texts.append(text)
    
    # Detect suspicious reviews if we have enough data
    if len(review_texts) > 10:
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(review_texts)
            clf = IsolationForest(contamination=0.1, random_state=42)
            predictions = clf.fit_predict(X)
            for i, review in enumerate(reviews):
                review["IsSuspicious"] = predictions[i] == -1
        except Exception as e:
            app.logger.error(f"Error in fake review detection: {e}")
            for review in reviews:
                review["IsSuspicious"] = False
    
    return sentiment_results, word_list, reviews

def generate_wordcloud(words):
    """Generate a word cloud from review text."""
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'product', 'item', 'good', 'great', 'bad', 'like', 'one', 'use', 'would'}
    stop_words.update(custom_stopwords)
    
    filtered_words = [word for word in words if word not in stop_words]
    
    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=200,
        contour_width=3,
        contour_color='steelblue'
    ).generate(" ".join(filtered_words))
    
    wordcloud_path = "static/wordcloud.png"
    if not os.path.exists('static'):
        os.makedirs('static')
    wordcloud.to_file(wordcloud_path)
    return wordcloud_path

# Routes
@app.route("/set_price_alert", methods=["POST"])
def set_price_alert():
    """Handle price alert form submission."""
    if request.method == "POST":
        conn = None
        try:
            email = request.form.get("email")
            target_price = float(request.form.get("target_price"))
            product_url = session.get('product_url')
            
            if not product_url:
                flash("No product URL found in session", "danger")
                return redirect(url_for('index'))
            
            current_price = track_price(product_url)
            
            if current_price is None:
                flash("Could not determine current price", "danger")
                return redirect(url_for('index'))
            
            conn = sqlite3.connect('price_alerts.db')
            c = conn.cursor()
            c.execute("""INSERT INTO price_alerts 
                      (email, product_url, target_price, current_price, last_checked) 
                      VALUES (?, ?, ?, ?, datetime('now'))""",
                      (email, product_url, target_price, current_price))
            conn.commit()
            
            flash(f"Price alert set! We'll notify you when the price drops below ₹{target_price}", "success")
            return redirect(url_for('index'))
            
        except ValueError:
            flash("Please enter a valid target price", "danger")
            return redirect(url_for('index'))
        except Exception as e:
            app.logger.error(f"Error setting price alert: {str(e)}")
            flash("Error setting price alert. Please try again.", "danger")
            return redirect(url_for('index'))
        finally:
            if conn:
                conn.close()

@app.route('/subscribe', methods=['POST'])
def handle_subscription():
    """Handle newsletter subscriptions."""
    email = request.form['email']
    try:
        msg = Message('Newsletter Subscription', recipients=[email])
        msg.html = """
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: 'Arial', sans-serif; background-color: #f9f9f9; padding: 0; margin: 0; }
    .container { max-width: 600px; margin: 30px auto; background: white; border-radius: 10px; padding: 30px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }
    h1 { color: #333333; font-size: 28px; text-align: center; }
    p { font-size: 16px; line-height: 1.6; color: #555555; }
    .header-image { text-align: center; margin-bottom: 20px; }
    .cta { display: inline-block; margin-top: 20px; padding: 12px 24px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }
    .footer { font-size: 12px; color: #aaa; text-align: center; margin-top: 30px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header-image">
      <img src="https://t4.ftcdn.net/jpg/13/97/95/65/240_F_1397956586_xIuJjfNgn6CdVxbZak7OoUL553rIEvX4.jpg" width="120" alt="Email icon" />
    </div>
    <h1>🎉 Welcome to Our Newsletter!</h1>
    <p>Hi there! 👋</p>
    <p>We're thrilled to have you on board. You've just joined a vibrant community that loves to stay updated, learn, and grow! 💡</p>
    <p>Here's what you can expect from us:
      <ul>
        <li>📬 Fresh updates right in your inbox</li>
        <li>🔥 Latest trends and insights</li>
        <li>🎁 Exclusive offers & giveaways</li>
      </ul>
    </p>
    <p>Stay tuned for your first newsletter. Until then, follow us on social media for more fun updates!</p>
    <div style="text-align:center;">
      <a class="cta" href="https://yourwebsite.com" target="_blank">Visit Our Website 🌐</a>
    </div>
    <div class="footer">
      <p>You received this email because you subscribed to our newsletter.</p>
      <p>Unsubscribe at any time. No hard feelings 💔</p>
    </div>
  </div>
</body>
</html>
"""
        mail.send(msg)
        flash('Subscription successful! A confirmation email has been sent.', 'success')
    except Exception as e:
        app.logger.error(f'Error sending subscription email: {str(e)}')
        flash('An error occurred while processing your subscription. Please try again.', 'danger')
    return redirect(url_for('index'))

@app.route("/", methods=["GET", "POST"])
def index():
    """Main application route."""
    if request.method == "POST":
        if 'set_alert' in request.form:
            return set_price_alert()
        
        product_url = request.form.get("product_url")
        platform = request.form.get("platform")
        
        if not product_url or not platform:
            return render_template("index.html", error="Please enter a URL and select a platform.")
        
        try:
            session['product_url'] = product_url
            session['platform'] = platform
            
            if platform == "flipkart":
                reviews = scrape_flipkart_reviews(product_url)
            elif platform == "amazon":
                reviews = scrape_amazon_reviews(product_url)
            else:
                return render_template("index.html", error="Invalid platform selected.")
            
            if not reviews:
                return render_template("index.html", error="No reviews found.")
            
            current_price = track_price(product_url.split('/ref=')[0])
            sentiment_results, word_list, analyzed_reviews = analyze_reviews(reviews)
            wordcloud_path = generate_wordcloud(word_list)
            
            # Calculate suspicious reviews percentage
            suspicious_count = sum(1 for r in analyzed_reviews if r.get("IsSuspicious", False))
            fake_percentage = (suspicious_count / len(analyzed_reviews)) * 100 if analyzed_reviews else 0
            
            # Calculate average rating
            avg_rating = None
            if analyzed_reviews:
                total_ratings = sum(review.get('Rating', 0) for review in analyzed_reviews)
                valid_reviews = sum(1 for review in analyzed_reviews if review.get('Rating') is not None)
                if valid_reviews > 0:
                    avg_rating = round(total_ratings / valid_reviews, 1)
            
            return render_template(
                "index.html",
                reviews=analyzed_reviews[:20],
                sentiment_results=sentiment_results,
                wordcloud_path=wordcloud_path,
                current_price=current_price,
                fake_percentage=round(fake_percentage, 1),
                avg_rating=avg_rating,
                success="Analysis completed successfully!"
            )
            
        except Exception as e:
            app.logger.error(f"Error during analysis: {e}")
            return render_template("index.html", error=f"Analysis failed: {str(e)}")
    
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """API endpoint for review analysis."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    product_url = data.get("product_url")
    platform = data.get("platform")
    
    if not product_url or not platform:
        return jsonify({"error": "Missing product_url or platform"}), 400
    
    try:
        if platform == "flipkart":
            reviews = scrape_flipkart_reviews(product_url, 20)
        elif platform == "amazon":
            reviews = scrape_amazon_reviews(product_url, 20)
        else:
            return jsonify({"error": "Invalid platform"}), 400
        
        if not reviews:
            return jsonify({"error": "No reviews found"}), 404
        
        sentiment_results, word_list, analyzed_reviews = analyze_reviews(reviews)
        current_price = track_price(product_url.split('/ref=')[0])
        
        # Calculate average rating for API
        avg_rating = None
        if analyzed_reviews:
            total_ratings = sum(review.get('Rating', 0) for review in analyzed_reviews)
            valid_reviews = sum(1 for review in analyzed_reviews if review.get('Rating') is not None)
            if valid_reviews > 0:
                avg_rating = round(total_ratings / valid_reviews, 1)
        
        return jsonify({
            "reviews": analyzed_reviews,
            "sentiment": sentiment_results,
            "current_price": current_price,
            "review_count": len(analyzed_reviews),
            "avg_rating": avg_rating
        })
        
    except Exception as e:
        app.logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'false').lower() == 'true')
