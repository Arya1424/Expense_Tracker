# 💰 AI-Powered Expense Tracker

An intelligent expense tracking application built with **Streamlit** that uses **Machine Learning** to automatically detect unusual spending patterns and anomalies in your financial transactions.


## 🌟 Features

### 📊 Smart Dashboard
- **Real-time analytics** with interactive visualizations
- Summary metrics: Total spend, daily averages, top categories
- **Plotly charts**: Pie charts for category breakdown, time series with anomaly highlighting
- Recent transactions table with anomaly flags

### ➕ Manual Transaction Entry
- Clean, intuitive form for adding expenses
- Fields: Date, Description, Amount, Category
- **Quick-add buttons** for common expenses (coffee, lunch, transport)
- Real-time validation and success feedback

### 🤖 AI Anomaly Detection
- **Isolation Forest algorithm** from scikit-learn
- Automatically detects unusual spending patterns
- Considers multiple features:
  - Transaction amount
  - Category
  - Day of week
  - Month
  - Day of month
- **Adjustable sensitivity** slider for fine-tuning detection

### ⚠️ Anomaly Management
- View all flagged transactions with explanations
- Understand **why** a transaction was flagged
- Mark false positives as "normal"
- Re-run detection with different sensitivity levels

### 💾 Persistent Storage
- **SQLite database** for data persistence
- Automatic table creation and management
- CRUD operations for transactions
- Efficient anomaly flag updates

### 📥 Data Export
- **CSV download** for all transactions or anomalies only
- **Markdown summary report** with detailed statistics
- Data preview and quick stats
- Category breakdown with totals and averages

## 🚀 Demo

[Live Demo on Streamlit Cloud](https://expensetracker-c8goglcu6r6tkpdbnuftna.streamlit.app/)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-expense-tracker.git
cd ai-expense-tracker
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run expense_tracker.py
```

4. **Open in browser**
The app will automatically open at `http://localhost:8501`

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

## 🎯 Usage

### Adding Transactions

1. Navigate to **"➕ Add Transaction"** from the sidebar
2. Fill in the transaction details:
   - **Date**: Select transaction date
   - **Description**: Brief description (e.g., "Lunch at McDonald's")
   - **Amount**: Transaction amount in ₹
   - **Category**: Select from predefined categories
3. Click **"💾 Save Transaction"**
4. Use **Quick Add** buttons for common expenses

### Viewing Dashboard

1. Navigate to **"📊 Dashboard"**
2. View summary metrics at the top
3. Analyze spending with interactive charts:
   - **Pie chart**: Spending distribution by category
   - **Time series**: Daily spending with anomaly highlights
4. Check recent transactions table

### Managing Anomalies

1. Navigate to **"⚠️ Anomalies"**
2. View all flagged transactions with explanations
3. Adjust sensitivity slider to re-run detection
4. Mark false positives as normal

### Exporting Data

1. Navigate to **"📥 Export Data"**
2. Download CSV files:
   - All transactions
   - Anomalies only
3. Download markdown summary report
4. View data preview and statistics

## 🧠 Machine Learning Model

### Isolation Forest Algorithm

The app uses **Isolation Forest**, an unsupervised learning algorithm ideal for anomaly detection:

- **How it works**: Isolates anomalies by randomly selecting features and split values
- **Why it's effective**: Anomalies are few and different, thus easier to isolate
- **Features used**:
  - Transaction amount (normalized)
  - Category (encoded)
  - Temporal features (day of week, month, day of month)

### Model Configuration

```python
IsolationForest(
    contamination=0.1,      # Expected proportion of anomalies
    random_state=42,        # For reproducibility
    n_estimators=100        # Number of trees
)
```

## 📁 Project Structure

```
ai-expense-tracker/
│
├── expense_tracker.py      # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── expense_tracker.db     # SQLite database (auto-generated)
```

## 🔧 Configuration

### Categories

Default categories can be modified in the `show_add_transaction()` function:

```python
category = st.selectbox(
    "🏷️ Category", 
    ["Food", "Transportation", "Entertainment", "Shopping", 
     "Bills", "Healthcare", "Travel", "Education", "Other"]
)
```

### Anomaly Detection Sensitivity

Adjust the contamination parameter (default: 0.1 = 10% anomalies expected):

```python
anomaly_results = detect_anomalies(df, contamination=0.1)
```
