# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
import pandas as pd
import os
import pymongo
import pymysql
import numpy as np
from sqlalchemy.orm import Session
import plotly.express as px
import plotly
from plotly.graph_objects import Figure

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
session = db.session  # Session for ORM queries

# Configure Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(100), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Define Dataset model
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    data = db.Column(db.PickleType, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Define models for MySQL and MongoDB datasets
class MySQLDataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    data = db.Column(db.PickleType, nullable=False)  # Store dataset content
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class MongoDBDataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    data = db.Column(db.PickleType, nullable=False)  # Store dataset content
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Define Graph model
class Graph(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    graph_type = db.Column(db.String(50), nullable=False)
    dataset1_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    column1 = db.Column(db.String(100), nullable=False)
    dataset2_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    column2 = db.Column(db.String(100))

# Initialize Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return session.get(User, int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET'])
def signup_form():
    return render_template('signup.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Validate password length
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return redirect(url_for('signup'))

        # Check for existing user
        existing_user = session.query(User).filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('signup'))

        # Create a new user
        new_user = User(username=username)
        new_user.set_password(password)
        session.add(new_user)
        session.commit()
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = session.query(User).filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Invalid username or password!', 'error')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file_type = request.form.get('filetype')

        # Handle file uploads based on the file type
        if file_type in ['csv', 'excel']:
            file = request.files.get('file')
            if not file:
                flash('No file uploaded!', 'error')
                return redirect(url_for('upload'))

            filename = file.filename

            if file_type == 'csv' and filename.endswith('.csv'):
                data = pd.read_csv(file)
                dataset = Dataset(name=filename, data=data, user_id=current_user.id)
                session.add(dataset)
                session.commit()
                flash('CSV file uploaded successfully!', 'success')
                return redirect(url_for('data'))

            elif file_type == 'excel' and filename.endswith(('.xls', '.xlsx')):
                xls = pd.ExcelFile(file)
                for sheet_name in xls.sheet_names:
                    data = pd.read_excel(file, sheet_name=sheet_name)
                    dataset = Dataset(name=f"{filename} - {sheet_name}", data=data, user_id=current_user.id)
                    session.add(dataset)
                session.commit()
                flash('Excel file with multiple sheets uploaded successfully!', 'success')
                return redirect(url_for('data'))

        elif file_type == 'mysql':
            host = request.form['host']
            username = request.form['username']
            password = request.form['password']
            database = request.form['database']
            table = request.form['table']

            try:
                connection = pymysql.connect(
                    host=host,
                    user=username,
                    password=password,
                    database=database
                )
                data = pd.read_sql(f'SELECT * FROM {table}', connection)
                mysql_dataset = Dataset(name=f'{database} - {table}', data=data, user_id=current_user.id)
                session.add(mysql_dataset)
                session.commit()
                flash('MySQL data uploaded successfully!', 'success')

            except Exception as e:
                flash(f'Error uploading data from MySQL: {e}', 'error')
            
            finally:
                if connection:
                    connection.close()

            return redirect(url_for('data'))

        elif file_type == 'mongodb':
            mongo_host = request.form['mongo-host']
            mongo_port = int(request.form['mongo-port'])
            mongo_database = request.form['mongo-database']
            mongo_collection = request.form['collection']

            try:
                client = pymongo.MongoClient(f'mongodb://{mongo_host}:{mongo_port}/')
                mongo_db = client[mongo_database]
                collection = mongo_db[mongo_collection]

                cursor = collection.find()
                data = pd.DataFrame(list(cursor))
                mongo_dataset = Dataset(name=f'{mongo_database} - {mongo_collection}', data=data, user_id=current_user.id)
                session.add(mongo_dataset)
                session.commit()
                flash('MongoDB data uploaded successfully!', 'success')

            except Exception as e:
                flash(f'Error uploading data from MongoDB: {e}', 'error')

            finally:
                if client:
                    client.close()

            return redirect(url_for('data'))

        else:
            flash('Invalid file format selected. Please select CSV, Excel, MySQL, or MongoDB.', 'error')

    return render_template('upload.html')

@app.route('/data')
@login_required
def data():
    datasets = session.query(Dataset).filter_by(user_id=current_user.id).all()
    return render_template('data.html', datasets=datasets)

@app.route('/delete/<int:dataset_id>', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    dataset = session.get(Dataset, dataset_id)
    if dataset and dataset.user_id == current_user.id:
        session.delete(dataset)
        session.commit()
        flash('Dataset deleted successfully!', 'success')
    else:
        abort(403)

    return redirect(url_for('data'))

@app.route('/clean/<int:dataset_id>', methods=['GET', 'POST'])
@login_required
def clean(dataset_id):
    dataset = session.get(Dataset, dataset_id)
    if request.method == 'POST':
        remove_duplicates = 'remove_duplicates' in request.form
        fill_na_method_text = request.form.get('fill_na_method_text')
        fill_na_method_numeric = request.form.get('fill_na_method_numeric')
        normalize = 'normalize' in request.form
        fill_na_constant = request.form.get('fill_na_constant')
        if fill_na_constant:
            fill_na_constant = float(fill_na_constant)

        cleaned_data, messages = clean_data(
            dataset.data,
            remove_duplicates,
            fill_na_method_text,
            fill_na_method_numeric,
            fill_na_constant,
            normalize
        )
        dataset.data = cleaned_data
        session.commit()
        
        for message in messages:
            flash(message, 'success')

        flash('Data cleaned and preprocessed successfully!', 'success')
        return redirect(url_for('dashboard'))

    if dataset:
        columns = dataset.data.columns
        return render_template('clean.html', dataset=dataset, columns=columns)
    else:
        flash('Dataset not found!', 'error')
        return redirect(url_for('data'))

def clean_data(data, remove_duplicates, fill_na_method_text, fill_na_method_numeric, fill_na_constant=None, normalize=False):
    messages = []

    for column in data.columns:
        if data[column].isnull().any():
            if data[column].dtype == 'object':
                if fill_na_method_text == 'empty_string':
                    data[column].fillna('', inplace=True)
                    messages.append(f"Filled missing values in '{column}' with empty strings.")
                elif fill_na_method_text == 'mode':
                    data[column].fillna(data[column].mode()[0], inplace=True)
                    messages.append(f"Filled missing values in '{column}' with mode.")
            else:
                if fill_na_method_numeric == 'mean':
                    data[column].fillna(data[column].mean(), inplace=True)
                    messages.append(f"Filled missing values in '{column}' with mean.")
                elif fill_na_method_numeric == 'median':
                    data[column].fillna(data[column].median(), inplace=True)
                    messages.append(f"Filled missing values in '{column}' with median.")
                elif fill_na_method_numeric == 'mode':
                    data[column].fillna(data[column].mode()[0], inplace=True)
                    messages.append(f"Filled missing values in '{column}' with mode.")
                elif fill_na_method_numeric == 'constant' and fill_na_constant is not None:
                    data[column].fillna(fill_na_constant, inplace=True)
                    messages.append(f"Filled missing values in '{column}' with constant value {fill_na_constant}.")

    if remove_duplicates:
        def is_unhashable(x):
            return isinstance(x, (list, dict, set))

        unhashable_columns = [column for column in data.columns if data[column].apply(is_unhashable).any()]
        
        for column in unhashable_columns:
            data[column] = data[column].apply(lambda x: str(x) if is_unhashable(x) else x)
        
        initial_row_count = data.shape[0]
        data.drop_duplicates(inplace=True)
        final_row_count = data.shape[0]
        duplicates_removed = initial_row_count - final_row_count
        messages.append(f"Removed {duplicates_removed} duplicate rows.")

    if normalize:
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data[numerical_columns] = (data[numerical_columns] - data[numerical_columns].mean()) / data[numerical_columns].std()
        messages.append("Normalized numerical columns.")

    return data, messages

@app.route('/dashboard')
@login_required
def dashboard():
    datasets = session.query(Dataset).filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', datasets=datasets)

@app.route('/get-columns/<int:dataset_id>')
def get_columns(dataset_id):
    dataset = session.get(Dataset, dataset_id)
    if dataset:
        columns = list(dataset.data.columns)
        return jsonify(columns)
    else:
        return jsonify([])

@app.route('/generate-graph', methods=['POST'])
@login_required
def generate_graph():
    graph_type = request.form['graph_type']
    dataset1_id = request.form['dataset1']
    column1 = request.form['column1']
    dataset2_id = request.form['dataset2']
    column2 = request.form['column2']
    graph_title = request.form['graph_title']
    x_label = request.form['x_label']
    y_label = request.form['y_label']

    new_graph = Graph(
        user_id=current_user.id,
        graph_type=graph_type,
        dataset1_id=dataset1_id,
        column1=column1,
        dataset2_id=dataset2_id,
        column2=column2
    )
    session.add(new_graph)
    session.commit()

    dataset1 = Dataset.query.get(dataset1_id)
    dataset2 = Dataset.query.get(dataset2_id)

    data1 = dataset1.data[[column1]]
    data2 = dataset2.data[[column2]]

    # Ensure that both datasets have a common key or are aligned properly
    if 'common_key' in data1.columns and 'common_key' in data2.columns:
        merged_data = pd.merge(data1, data2, on='common_key')
        data_x = merged_data[column1]
        data_y = merged_data[column2]
    else:
        # Align datasets by index
        data1 = data1.reset_index(drop=True)
        data2 = data2.reset_index(drop=True)
        data_x = data1[column1]
        data_y = data2[column2]

    if pd.api.types.is_datetime64_any_dtype(data_x):
        data_x = data_x.apply(pd.to_datetime).tolist()
    else:
        data_x = data_x.tolist()
    
    if pd.api.types.is_datetime64_any_dtype(data_y):
        data_y = data_y.apply(pd.to_datetime).tolist()
    else:
        data_y = data_y.tolist()
    
    graph_data = {}
    if graph_type == 'line':
        graph_data = {'x': data_x, 'y': data_y, 'type': 'scatter', 'mode': 'lines'}
    elif graph_type == 'bar':
        graph_data = {'x': data_x, 'y': data_y, 'type': 'bar'}
    elif graph_type == 'scatter':
        graph_data = {'x': data_x, 'y': data_y, 'type': 'scatter', 'mode': 'markers'}
    elif graph_type == 'histogram':
        graph_data = {'x': data_x, 'type': 'histogram'}
    elif graph_type == 'pie':
        graph_data = {'labels': data_x, 'values': data_y, 'type': 'pie'}
    elif graph_type == 'heatmap':
        graph_data = {'x': data_x, 'y': data_y, 'type': 'heatmap'}
    elif graph_type == 'bubble':
        graph_data = {'x': data_x, 'y': data_y, 'mode': 'markers', 'marker': {'size': data_y}, 'type': 'scatter'}
    elif graph_type == 'area':
        graph_data = {'x': data_x, 'y': data_y, 'type': 'scatter', 'mode': 'lines', 'fill': 'tozeroy'}

    layout = {
        'title': graph_title,
        'xaxis': {'title': x_label},
        'yaxis': {'title': y_label}
    }

    data = {'graph_type': graph_type, 'graph_data': [graph_data], 'layout': layout}

    return jsonify(data)

@app.route('/get-saved-graphs', methods=['GET'])
@login_required
def get_saved_graphs():
    graphs = Graph.query.filter_by(user_id=current_user.id).all()
    saved_graphs = []
    for graph in graphs:
        dataset1 = Dataset.query.get(graph.dataset1_id)
        dataset2 = Dataset.query.get(graph.dataset2_id)
        data1 = dataset1.data[[graph.column1]]
        data2 = dataset2.data[[graph.column2]] if dataset2 else None

        if data2 is not None:
            if 'common_key' in data1.columns and 'common_key' in data2.columns:
                merged_data = pd.merge(data1, data2, on='common_key')
                data_x = merged_data[graph.column1]
                data_y = merged_data[graph.column2]
            else:
                data1 = data1.reset_index(drop=True)
                data2 = data2.reset_index(drop=True)
                data_x = data1[graph.column1]
                data_y = data2[graph.column2] if data2 is not None else None
        else:
            data_x = data1[graph.column1].tolist()
            data_y = None

        if pd.api.types.is_datetime64_any_dtype(data_x):
            data_x = data_x.apply(pd.to_datetime).tolist()
        else:
            data_x = data_x.tolist()

        if data_y is not None and pd.api.types.is_datetime64_any_dtype(data_y):
            data_y = data_y.apply(pd.to_datetime).tolist()
        else:
            data_y = data_y.tolist() if data_y is not None else None

        graph_data = {}
        if graph.graph_type == 'line':
            graph_data = {'x': data_x, 'y': data_y, 'type': 'scatter', 'mode': 'lines'}
        elif graph.graph_type == 'bar':
            graph_data = {'x': data_x, 'y': data_y, 'type': 'bar'}
        elif graph.graph_type == 'scatter':
            graph_data = {'x': data_x, 'y': data_y, 'type': 'scatter', 'mode': 'markers'}
        elif graph.graph_type == 'histogram':
            graph_data = {'x': data_x, 'type': 'histogram'}
        elif graph.graph_type == 'pie':
            graph_data = {'labels': data_x, 'values': data_y, 'type': 'pie'}
        elif graph.graph_type == 'heatmap':
            graph_data = {'x': data_x, 'y': data_y, 'type': 'heatmap'}
        elif graph.graph_type == 'bubble':
            graph_data = {'x': data_x, 'y': data_y, 'mode': 'markers', 'marker': {'size': data_y}, 'type': 'scatter'}
        elif graph.graph_type == 'area':
            graph_data = {'x': data_x, 'y': data_y, 'type': 'scatter', 'mode': 'lines', 'fill': 'tozeroy'}

        saved_graphs.append({'id': graph.id, 'graph_data': [graph_data], 'layout': {'title': 'Saved Graph', 'xaxis': {'title': 'X-axis'}, 'yaxis': {'title': 'Y-axis'}}})

    return jsonify(saved_graphs)

@app.route('/clear-dashboard', methods=['POST'])
@login_required
def clear_dashboard():
    graphs = session.query(Graph).filter_by(user_id=current_user.id).all()
    for graph in graphs:
        session.delete(graph)
    session.commit()
    return jsonify({'message': 'Dashboard cleared successfully!'})

@app.route('/generate-report', methods=['POST'])
@login_required
def generate_report():
    data = request.get_json()
    if not data or 'graphsHtml' not in data:
        flash('No graphs data provided!', 'error')
        return redirect(url_for('dashboard'))

    graphs_html = data['graphsHtml']

    report_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dynamic HTML Report</title>
        <!-- Include Plotly.js -->
        <script src="https://cdn.plotly.com/plotly-latest.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.20.0/plotly.min.js"></script>
    </head>
    <body>
        <h1>Report</h1>
        {"<hr>".join(graphs_html)}  <!-- Embed each graph -->
    </body>
    </html>
    '''

    response = Response(report_html, mimetype='text/html')
    response.headers['Content-Disposition'] = 'attachment; filename="report.html"'

    return response

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
