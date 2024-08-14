from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///case_law.db'
db = SQLAlchemy(app)

class CaseLaw(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_name = db.Column(db.String(100), nullable=False)
    case_number = db.Column(db.String(20), nullable=False)
    court = db.Column(db.String(50), nullable=False)
    jurisdiction = db.Column(db.String(50), nullable=False)
    keywords = db.Column(db.String(200), nullable=False)

@app.route('/case_law', methods=['GET'])
def get_case_law():
    query = request.args.get('query')
    results = CaseLaw.query.filter(CaseLaw.keywords.like(f'%{query}%')).all()
    return jsonify([{'case_name': result.case_name, 'case_number': result.case_number, 'court': result.court, 'jurisdiction': result.jurisdiction} for result in results])

if __name__ == '__main__':
    app.run(debug=True)