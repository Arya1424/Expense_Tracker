import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import io
import base64

# Page configuration
st.set_page_config(
    page_title="AI Expense Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database setup
def init_database():
    """Initialize SQLite database with transactions table"""
    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            anomaly_flag INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Database operations
def add_transaction(date, description, amount, category):
    """Add a new transaction to database"""
    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO transactions (date, description, amount, category)
        VALUES (?, ?, ?, ?)
    ''', (date, description, amount, category))
    
    conn.commit()
    conn.close()

def get_all_transactions():
    """Fetch all transactions from database"""
    conn = sqlite3.connect('expense_tracker.db')
    df = pd.read_sql_query('''
        SELECT * FROM transactions 
        ORDER BY date DESC
    ''', conn)
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def update_anomaly_flags(anomaly_results):
    """Update anomaly flags in database"""
    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    
    # Reset all flags first
    cursor.execute('UPDATE transactions SET anomaly_flag = 0')
    
    # Update anomaly flags
    for idx, is_anomaly in enumerate(anomaly_results):
        cursor.execute('''
            UPDATE transactions 
            SET anomaly_flag = ? 
            WHERE id = (SELECT id FROM transactions ORDER BY id LIMIT 1 OFFSET ?)
        ''', (int(is_anomaly == -1), idx))
    
    conn.commit()
    conn.close()

def mark_not_anomaly(transaction_id):
    """Mark a transaction as not an anomaly"""
    conn = sqlite3.connect('expense_tracker.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE transactions 
        SET anomaly_flag = 0 
        WHERE id = ?
    ''', (transaction_id,))
    
    conn.commit()
    conn.close()

# AI Anomaly Detection
def detect_anomalies(df, contamination=0.1):
    """Detect anomalies using Isolation Forest"""
    if df.empty or len(df) < 5:  # Need minimum data points
        return np.array([])
    
    # Feature engineering
    features_df = df.copy()
    
    # Encode categorical variables
    le_category = LabelEncoder()
    features_df['category_encoded'] = le_category.fit_transform(df['category'])
    
    # Extract date features
    features_df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    features_df['month'] = pd.to_datetime(df['date']).dt.month
    features_df['day_of_month'] = pd.to_datetime(df['date']).dt.day
    
    # Select features for anomaly detection
    feature_columns = ['amount', 'category_encoded', 'day_of_week', 'month', 'day_of_month']
    X = features_df[feature_columns]
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    anomaly_labels = iso_forest.fit_predict(X)
    return anomaly_labels

# Visualization functions
def create_category_pie_chart(df):
    """Create pie chart for spending by category"""
    category_spending = df.groupby('category')['amount'].sum().reset_index()
    
    fig = px.pie(
        category_spending, 
        values='amount', 
        names='category',
        title='Spending by Category',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_time_series_chart(df):
    """Create time series chart with anomaly highlighting"""
    if df.empty:
        return go.Figure()
    
    # Aggregate by date
    daily_spending = df.groupby('date').agg({
        'amount': 'sum',
        'anomaly_flag': 'max'
    }).reset_index()
    
    fig = go.Figure()
    
    # Normal transactions
    normal_data = daily_spending[daily_spending['anomaly_flag'] == 0]
    fig.add_trace(go.Scatter(
        x=normal_data['date'],
        y=normal_data['amount'],
        mode='lines+markers',
        name='Normal Spending',
        line=dict(color='blue'),
        marker=dict(size=6)
    ))
    
    # Anomaly transactions
    anomaly_data = daily_spending[daily_spending['anomaly_flag'] == 1]
    if not anomaly_data.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_data['date'],
            y=anomaly_data['amount'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='diamond')
        ))
    
    fig.update_layout(
        title='Spending Over Time',
        xaxis_title='Date',
        yaxis_title='Amount (₹)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def generate_summary_stats(df):
    """Generate summary statistics"""
    if df.empty:
        return {
            'total_spend': 0,
            'avg_daily': 0,
            'top_category': 'N/A',
            'anomalies_count': 0,
            'total_transactions': 0
        }
    
    total_spend = df['amount'].sum()
    avg_daily = df.groupby('date')['amount'].sum().mean()
    top_category = df.groupby('category')['amount'].sum().idxmax()
    anomalies_count = df['anomaly_flag'].sum()
    total_transactions = len(df)
    
    return {
        'total_spend': total_spend,
        'avg_daily': avg_daily,
        'top_category': top_category,
        'anomalies_count': anomalies_count,
        'total_transactions': total_transactions
    }

def export_to_csv(df):
    """Export transactions to CSV"""
    return df.to_csv(index=False)

# Streamlit App
def main():
    # Initialize database
    init_database()
    
    # Sidebar navigation
    st.sidebar.title("🏦 AI Expense Tracker")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Dashboard", "➕ Add Transaction", "⚠️ Anomalies", "📥 Export Data"]
    )
    
    # Load data
    df = get_all_transactions()
    
    # Run anomaly detection if we have data
    if not df.empty and len(df) >= 5:
        with st.spinner('Running AI anomaly detection...'):
            anomaly_results = detect_anomalies(df, contamination=0.1)
            if len(anomaly_results) > 0:
                update_anomaly_flags(anomaly_results)
                df = get_all_transactions()  # Refresh data with updated flags
    
    # Page routing
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "➕ Add Transaction":
        show_add_transaction()
    elif page == "⚠️ Anomalies":
        show_anomalies(df)
    elif page == "📥 Export Data":
        show_export(df)

def show_dashboard(df):
    """Display main dashboard"""
    st.title("📊 Expense Dashboard")
    
    # Summary statistics
    stats = generate_summary_stats(df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Spend", 
            value=f"₹{stats['total_spend']:,.2f}"
        )
    
    with col2:
        st.metric(
            label="📈 Avg Daily", 
            value=f"₹{stats['avg_daily']:,.2f}"
        )
    
    with col3:
        st.metric(
            label="🏆 Top Category", 
            value=stats['top_category']
        )
    
    with col4:
        st.metric(
            label="⚠️ Anomalies Found", 
            value=stats['anomalies_count'],
            delta=f"out of {stats['total_transactions']}"
        )
    
    if not df.empty:
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = create_category_pie_chart(df)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_time = create_time_series_chart(df)
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Recent transactions
        st.subheader("📋 Recent Transactions")
        recent_df = df.head(10).copy()
        recent_df['Anomaly'] = recent_df['anomaly_flag'].map({0: '✅ Normal', 1: '🚨 Anomaly'})
        
        st.dataframe(
            recent_df[['date', 'description', 'amount', 'category', 'Anomaly']],
            use_container_width=True
        )
    else:
        st.info("👋 Welcome! Start by adding your first transaction using the '➕ Add Transaction' page.")

def show_add_transaction():
    """Display add transaction form"""
    st.title("➕ Add New Transaction")
    
    with st.form("add_transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input(
                "📅 Date", 
                value=datetime.now().date(),
                help="Select the transaction date"
            )
            
            description = st.text_input(
                "📝 Description", 
                placeholder="e.g., Lunch at McDonald's",
                help="Brief description of the expense"
            )
        
        with col2:
            amount = st.number_input(
                "💰 Amount (₹)", 
                min_value=0.01, 
                step=0.01,
                help="Enter the transaction amount"
            )
            
            category = st.selectbox(
                "🏷️ Category", 
                ["Food", "Transportation", "Entertainment", "Shopping", "Bills", 
                 "Healthcare", "Travel", "Education", "Other"],
                help="Select expense category"
            )
        
        submitted = st.form_submit_button("💾 Save Transaction", type="primary")
        
        if submitted:
            if description and amount > 0:
                try:
                    add_transaction(date, description, amount, category)
                    st.success(f"✅ Transaction saved: ₹{amount:.2f} for {description}")
                    st.balloons()
                    
                    # Show option to add another
                    if st.button("➕ Add Another Transaction"):
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error saving transaction: {str(e)}")
            else:
                st.error("⚠️ Please fill in all required fields!")
    
    # Quick add buttons for common expenses
    st.subheader("🚀 Quick Add")
    st.markdown("Click to quickly add common expenses:")
    
    quick_expenses = [
        ("☕ Coffee", 150, "Food"),
        ("🚌 Bus Fare", 50, "Transportation"),
        ("🍕 Lunch", 300, "Food"),
        ("⛽ Fuel", 1000, "Transportation"),
        ("🎬 Movie", 400, "Entertainment")
    ]
    
    cols = st.columns(len(quick_expenses))
    for i, (desc, amt, cat) in enumerate(quick_expenses):
        with cols[i]:
            if st.button(f"{desc}\n₹{amt}", key=f"quick_{i}"):
                add_transaction(datetime.now().date(), desc.split()[1], amt, cat)
                st.success(f"✅ Added {desc}")
                st.rerun()

def show_anomalies(df):
    """Display anomaly detection results"""
    st.title("⚠️ Anomaly Detection")
    
    if df.empty:
        st.info("📝 No transactions yet. Add some transactions to detect anomalies!")
        return
    
    anomalies_df = df[df['anomaly_flag'] == 1].copy()
    
    if anomalies_df.empty:
        st.success("🎉 No anomalies detected! Your spending patterns look normal.")
    else:
        st.warning(f"🚨 Found {len(anomalies_df)} potentially unusual transactions:")
        
        # Anomaly sensitivity control
        st.subheader("🎛️ Detection Sensitivity")
        sensitivity = st.slider(
            "Adjust sensitivity (lower = more sensitive)",
            min_value=0.05,
            max_value=0.3,
            value=0.1,
            step=0.05,
            help="Lower values detect more anomalies"
        )
        
        if st.button("🔄 Re-run Detection"):
            with st.spinner('Re-analyzing transactions...'):
                anomaly_results = detect_anomalies(df, contamination=sensitivity)
                if len(anomaly_results) > 0:
                    update_anomaly_flags(anomaly_results)
                    st.success("✅ Anomaly detection completed!")
                    st.rerun()
        
        st.subheader("🔍 Anomalous Transactions")
        
        # Display anomalies with action buttons
        for idx, row in anomalies_df.iterrows():
            with st.expander(f"🚨 {row['description']} - ₹{row['amount']:,.2f}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Date:** {row['date'].strftime('%Y-%m-%d')}")
                    st.write(f"**Category:** {row['category']}")
                    st.write(f"**Amount:** ₹{row['amount']:,.2f}")
                
                with col2:
                    # Calculate why it's anomalous
                    category_avg = df[df['category'] == row['category']]['amount'].mean()
                    overall_avg = df['amount'].mean()
                    
                    if row['amount'] > category_avg * 2:
                        st.write("💡 **Why anomalous:**")
                        st.write(f"Much higher than avg {row['category']} expense (₹{category_avg:.2f})")
                    elif row['amount'] > overall_avg * 3:
                        st.write("💡 **Why anomalous:**")
                        st.write(f"Much higher than overall average (₹{overall_avg:.2f})")
                
                with col3:
                    if st.button(f"✅ Mark as Normal", key=f"normal_{row['id']}"):
                        mark_not_anomaly(row['id'])
                        st.success("Marked as normal!")
                        st.rerun()

def show_export(df):
    """Display export options"""
    st.title("📥 Export Data")
    
    if df.empty:
        st.info("📝 No transactions to export yet.")
        return
    
    # Export options
    st.subheader("📊 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Download CSV")
        csv_data = export_to_csv(df)
        
        st.download_button(
            label="📁 Download All Transactions",
            data=csv_data,
            file_name=f"expense_tracker_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Export only anomalies
        if df['anomaly_flag'].sum() > 0:
            anomalies_csv = export_to_csv(df[df['anomaly_flag'] == 1])
            st.download_button(
                label="⚠️ Download Anomalies Only",
                data=anomalies_csv,
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.subheader("📈 Summary Report")
        stats = generate_summary_stats(df)
        
        report = f"""
        # Expense Tracker Report
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## Summary Statistics
        - **Total Transactions:** {stats['total_transactions']}
        - **Total Spending:** ₹{stats['total_spend']:,.2f}
        - **Average Daily Spending:** ₹{stats['avg_daily']:,.2f}
        - **Top Category:** {stats['top_category']}
        - **Anomalies Detected:** {stats['anomalies_count']}
        
        ## Category Breakdown
        """
        
        category_stats = df.groupby('category').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            total = cat_data['amount'].sum()
            count = len(cat_data)
            avg = cat_data['amount'].mean()
            report += f"\n- **{category}:** ₹{total:,.2f} ({count} transactions, avg: ₹{avg:.2f})"
        
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name=f"expense_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df, use_container_width=True)
    
    # Statistics
    st.subheader("📊 Quick Stats")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", len(df))
        st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    with col2:
        st.metric("Categories", df['category'].nunique())
        st.metric("Anomalies", df['anomaly_flag'].sum())
    
    with col3:
        st.metric("Avg Amount", f"₹{df['amount'].mean():.2f}")
        st.metric("Max Amount", f"₹{df['amount'].max():.2f}")

if __name__ == "__main__":
    main()