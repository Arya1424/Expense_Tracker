# streamlit_app.py
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import bcrypt
import os

# Optional: Plotly import with graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# Page config
st.set_page_config(page_title="AI Expense Tracker (Multi-user)", page_icon="💰", layout="wide")

DB_PATH = "expense_tracker.db"

# -----------------------
# DB helpers & migration
# -----------------------
def get_connection():
    # Allow cross-thread usage for Streamlit reruns
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_database():
    """Initialize DB: users table and transactions table (with user_id)."""
    conn = get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Transactions table (user_id included)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL DEFAULT 0,
            date DATE NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            anomaly_flag INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Migration safeguard: if transactions table exists without user_id, try to add column
    # (SQLite ALTER TABLE ADD COLUMN is safe if column doesn't exist)
    # We'll check pragma_table_info to be sure
    cursor.execute("PRAGMA table_info(transactions)")
    cols = [r[1] for r in cursor.fetchall()]
    if 'user_id' not in cols:
        try:
            cursor.execute('ALTER TABLE transactions ADD COLUMN user_id INTEGER DEFAULT 0')
        except Exception:
            # ignore if cannot alter (rare)
            pass

    conn.commit()
    conn.close()

# -----------------------
# User management
# -----------------------
def hash_password(plain_password: str) -> bytes:
    return bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())

def verify_password(plain_password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed)
    except Exception:
        return False

def create_user(email: str, password: str) -> (bool, str):
    """Create user. Returns (success, message)."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        pw_hash = hash_password(password)
        cursor.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)', (email.lower(), pw_hash))
        conn.commit()
        return True, "User created."
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def authenticate_user(email: str, password: str):
    """Return user dict if auth succeeds, else None."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, email, password_hash FROM users WHERE email = ?', (email.lower(),))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    user_id, user_email, pw_hash = row[0], row[1], row[2]
    if verify_password(password, pw_hash):
        return {"id": user_id, "email": user_email}
    return None

# -----------------------
# Transaction ops (user-scoped)
# -----------------------
def add_transaction(user_id, date, description, amount, category):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO transactions (user_id, date, description, amount, category)
        VALUES (?, ?, ?, ?, ?)
    ''', (int(user_id), pd.to_datetime(date).strftime('%Y-%m-%d'), description, float(amount), category))
    conn.commit()
    conn.close()

def get_all_transactions(user_id):
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT * FROM transactions
        WHERE user_id = ?
        ORDER BY date DESC, id DESC
    ''', conn, params=(int(user_id),))
    conn.close()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def update_anomaly_flags_user(df, anomaly_results):
    """Update anomaly flags for given user's df (mapping by explicit id)."""
    if df.empty or len(anomaly_results) != len(df):
        return
    ids = df['id'].tolist()
    conn = get_connection()
    cursor = conn.cursor()
    # Reset user's flags first
    cursor.execute('UPDATE transactions SET anomaly_flag = 0 WHERE user_id = ?', (int(df['user_id'].iloc[0]),))
    for idx, label in enumerate(anomaly_results):
        tx_id = ids[idx]
        flag = 1 if label == -1 else 0
        cursor.execute('UPDATE transactions SET anomaly_flag = ? WHERE id = ?', (flag, int(tx_id)))
    conn.commit()
    conn.close()

def mark_not_anomaly(transaction_id, user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE transactions SET anomaly_flag = 0 WHERE id = ? AND user_id = ?', (transaction_id, int(user_id)))
    conn.commit()
    conn.close()

# -----------------------
# Anomaly detection & visuals (unchanged logic but user-scoped)
# -----------------------
def detect_anomalies(df, contamination=0.1):
    """Return labels array aligned with df rows (-1 => anomaly, 1 => normal)."""
    if df.empty or len(df) < 5:
        return np.array([])
    features_df = df.copy()
    le_category = LabelEncoder()
    features_df['category_encoded'] = le_category.fit_transform(features_df['category'].astype(str))
    features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
    features_df['month'] = pd.to_datetime(features_df['date']).dt.month
    features_df['day_of_month'] = pd.to_datetime(features_df['date']).dt.day
    feature_columns = ['amount', 'category_encoded', 'day_of_week', 'month', 'day_of_month']
    X = features_df[feature_columns].fillna(0)
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    labels = iso_forest.fit_predict(X)
    return labels

def create_category_pie_chart(df):
    if df.empty:
        return go.Figure()
    category_spending = df.groupby('category')['amount'].sum().reset_index()
    if PLOTLY_AVAILABLE:
        fig = px.pie(category_spending, values='amount', names='category', title='Spending by Category')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, height=400)
        return fig
    else:
        return go.Figure()

def create_time_series_chart(df):
    if df.empty:
        return go.Figure()
    daily = df.groupby('date').agg({'amount': 'sum', 'anomaly_flag': 'max'}).reset_index()
    fig = go.Figure()
    normal_data = daily[daily['anomaly_flag'] == 0]
    fig.add_trace(go.Scatter(x=normal_data['date'], y=normal_data['amount'], mode='lines+markers', name='Normal'))
    anomaly_data = daily[daily['anomaly_flag'] == 1]
    if not anomaly_data.empty:
        fig.add_trace(go.Scatter(x=anomaly_data['date'], y=anomaly_data['amount'], mode='markers', name='Anomalies', marker=dict(symbol='diamond', size=10)))
    fig.update_layout(title='Spending Over Time', xaxis_title='Date', yaxis_title='Amount (₹)', height=400)
    return fig

def generate_summary_stats(df):
    if df.empty:
        return {'total_spend': 0, 'avg_daily': 0, 'top_category': 'N/A', 'anomalies_count': 0, 'total_transactions': 0}
    total_spend = df['amount'].sum()
    avg_daily = df.groupby('date')['amount'].sum().mean()
    top_category = df.groupby('category')['amount'].sum().idxmax()
    anomalies_count = df['anomaly_flag'].sum()
    total_transactions = len(df)
    return {'total_spend': total_spend, 'avg_daily': avg_daily, 'top_category': top_category, 'anomalies_count': anomalies_count, 'total_transactions': total_transactions}

def export_to_csv(df):
    return df.to_csv(index=False)

# -----------------------
# UI: auth controls & app pages
# -----------------------
def show_auth_widget():
    st.sidebar.header("🔐 Account")
    if 'user' in st.session_state and st.session_state['user']:
        user = st.session_state['user']
        st.sidebar.write(f"Signed in as **{user['email']}**")
        if st.sidebar.button("🔓 Logout"):
            st.session_state['user'] = None
            st.rerun()
        return

    auth_mode = st.sidebar.radio("Auth", ["Login", "Sign up"], index=0)
    if auth_mode == "Sign up":
        with st.sidebar.form("signup_form"):
            su_email = st.text_input("Email")
            su_password = st.text_input("Password", type="password")
            su_confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create account")
            if submitted:
                if not su_email or not su_password:
                    st.sidebar.error("Email and password required.")
                elif su_password != su_confirm:
                    st.sidebar.error("Passwords do not match.")
                else:
                    ok, msg = create_user(su_email, su_password)
                    if ok:
                        st.sidebar.success("Account created — please log in.")
                    else:
                        st.sidebar.error(msg)
    else:  # Login
        with st.sidebar.form("login_form"):
            li_email = st.text_input("Email")
            li_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
            if submitted:
                user = authenticate_user(li_email, li_password)
                if user:
                    st.session_state['user'] = user
                    st.rerun()
                else:
                    st.sidebar.error("Invalid credentials.")

def require_login():
    if 'user' not in st.session_state or not st.session_state['user']:
        st.info("Please sign in or create an account using the sidebar to continue.")
        st.stop()

# -----------------------
# App pages (user-scoped)
# -----------------------
def page_dashboard(df):
    st.title("📊 Expense Dashboard")
    stats = generate_summary_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("💰 Total Spend", f"₹{stats['total_spend']:,.2f}")
    with c2:
        st.metric("📈 Avg Daily", f"₹{stats['avg_daily']:,.2f}")
    with c3:
        st.metric("🏆 Top Category", stats['top_category'])
    with c4:
        st.metric("⚠️ Anomalies Found", stats['anomalies_count'], delta=f"out of {stats['total_transactions']}")

    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = create_category_pie_chart(df)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_time = create_time_series_chart(df)
            st.plotly_chart(fig_time, use_container_width=True)

        st.subheader("📋 Recent Transactions")
        recent_df = df.head(10).copy()
        recent_df['Anomaly'] = recent_df['anomaly_flag'].map({0: '✅ Normal', 1: '🚨 Anomaly'})
        st.dataframe(recent_df[['date', 'description', 'amount', 'category', 'Anomaly']], use_container_width=True)
    else:
        st.info("No transactions yet. Add one using 'Add Transaction'.")

def page_add_transaction(user_id):
    st.title("➕ Add New Transaction")
    with st.form("add_transaction_form"):
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("📅 Date", value=datetime.now().date())
            description = st.text_input("📝 Description", placeholder="e.g., Lunch")
        with col2:
            amount = st.number_input("💰 Amount (₹)", min_value=0.01, step=0.01)
            category = st.selectbox("🏷️ Category", ["Food", "Transportation", "Entertainment", "Shopping", "Bills", "Healthcare", "Travel", "Education", "Other"])
        submitted = st.form_submit_button("💾 Save Transaction")
    # inside page_add_transaction (submission handling)
    if submitted:
        if description and amount > 0:
            try:
                add_transaction(user_id, date, description, amount, category)
                st.success(f"✅ Saved ₹{amount:.2f} — {description}")
                st.balloons()
                # Refresh the dataframe in-place without forcing a rerun:
                df = get_all_transactions(user_id)  # refresh local variable
                # optionally show the refreshed table or redirect to dashboard view here
            except Exception as e:
                st.error(f"Error saving: {e}")
        else:
            st.error("Fill description and amount.")


    # Quick add
    st.subheader("🚀 Quick Add")
    quick_expenses = [("☕ Coffee", 150, "Food"), ("🚌 Bus Fare", 50, "Transportation"), ("🍕 Lunch", 300, "Food"), ("⛽ Fuel", 1000, "Transportation"), ("🎬 Movie", 400, "Entertainment")]
    cols = st.columns(len(quick_expenses))
    # quick-add buttons (replace the previous block that called st.rerun())
    for i, (desc, amt, cat) in enumerate(quick_expenses):
        with cols[i]:
            if st.button(f"{desc}\n₹{amt}", key=f"quick_{i}"):
                add_transaction(user_id, datetime.now().date(), desc, amt, cat)
                st.success(f"✅ Added {desc}")
                st.balloons()
                df = get_all_transactions(user_id)  # refresh local variable so new row appears
                # do NOT call st.rerun() here


def page_anomalies(df, user_id):
    st.title("⚠️ Anomaly Detection")
    if df.empty:
        st.info("No transactions yet.")
        return
    anomalies_df = df[df['anomaly_flag'] == 1].copy()
    st.subheader("🎛️ Detection Sensitivity")
    sensitivity = st.slider("Contamination (higher = find more anomalies)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    if st.button("🔄 Re-run Detection"):
        with st.spinner("Analyzing..."):
            labels = detect_anomalies(df, contamination=sensitivity)
            if labels.size:
                update_anomaly_flags_user(df, labels)
                st.success("Done.")
                st.rerun()
            else:
                st.info("Not enough data to run detection.")
    if anomalies_df.empty:
        st.success("No anomalies detected.")
    else:
        st.warning(f"Found {len(anomalies_df)} anomalous transactions.")
        for _, row in anomalies_df.iterrows():
            with st.expander(f"🚨 {row['description']} - ₹{row['amount']:,.2f}"):
                c1, c2, c3 = st.columns([2,1,1])
                with c1:
                    st.write(f"**Date:** {row['date'].strftime('%Y-%m-%d')}")
                    st.write(f"**Category:** {row['category']}")
                    st.write(f"**Amount:** ₹{row['amount']:,.2f}")
                with c2:
                    cat_avg = df[df['category'] == row['category']]['amount'].mean()
                    overall_avg = df['amount'].mean()
                    if not np.isnan(cat_avg) and row['amount'] > cat_avg * 2:
                        st.write("💡 Much higher than category average.")
                    elif row['amount'] > overall_avg * 3:
                        st.write("💡 Much higher than overall average.")
                with c3:
                    if st.button("✅ Mark as Normal", key=f"normal_{row['id']}"):
                        mark_not_anomaly(row['id'], user_id)
                        st.success("Marked normal.")
                        st.rerun()

def page_export(df):
    st.title("📥 Export Data")
    if df.empty:
        st.info("No transactions to export.")
        return
    csv_all = export_to_csv(df)
    st.download_button("📁 Download CSV (All)", data=csv_all, file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    if df['anomaly_flag'].sum() > 0:
        st.download_button("⚠️ Download Anomalies Only", data=export_to_csv(df[df['anomaly_flag'] == 1]), file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    st.subheader("Preview")
    st.dataframe(df, use_container_width=True)

# -----------------------
# Main
# -----------------------
def main():
    init_database()
    show_auth_widget()
    # If not logged in, UI stops in require_login
    require_login()
    user = st.session_state['user']
    user_id = user['id']

    # Load user's data
    df = get_all_transactions(user_id)

    # Run anomaly detection automatically on load if enough data
    if not df.empty and len(df) >= 5:
        labels = detect_anomalies(df, contamination=0.1)
        if labels.size:
            update_anomaly_flags_user(df, labels)
            df = get_all_transactions(user_id)  # refresh

    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox("Navigate:", ["📊 Dashboard", "➕ Add Transaction", "⚠️ Anomalies", "📥 Export Data"])
    if page == "📊 Dashboard":
        page_dashboard(df)
    elif page == "➕ Add Transaction":
        page_add_transaction(user_id)
    elif page == "⚠️ Anomalies":
        page_anomalies(df, user_id)
    elif page == "📥 Export Data":
        page_export(df)

if __name__ == "__main__":
    main()
