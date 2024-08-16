# env: book_rec
'''
Ensuring that mamba is using the correct Python interpreter within your virtual environment (which python, which python3):
export PATH="/Users/toobarahimnia/miniforge3/envs/book_rec/bin:$PATH"
'''
from flask import Flask,render_template,request
import pickle
import numpy as np
from utils import recommending_books

# rec_books = pickle.load(open('models/model 2/recommended_books.pkl','rb'))
pt = pickle.load(open('models/model 2/book_pivot_table.pkl','rb'))
book = pickle.load(open('models/model 2/books.pkl','rb'))
preds = pickle.load(open('models/model 2/predicted_clusters.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(book['Book-Title'][109:121].values), #101, 113
                           image = list(book['Image-URL-M'][109:121].values),
                           )

@app.route('/recommendation')
def recommendation_ui():
    return render_template('recommendation.html')

@app.route('/recommend_books', methods=['post'])
def recommendation():
    user_input = request.form.get('user_input')

    recommendations = recommending_books(user_input, pt, preds, n_recommendations=10)
    
    # Check if the book is not found
    if isinstance(recommendations, np.ndarray) and recommendations[0].startswith("The book"):
        return render_template('recommendation.html', error=recommendations[0])
    
    # Gather the data for rendering
    data = []
    for book_name in recommendations:
        item = []
        temp_df = book[book['Book-Title'] == book_name]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
        print(item)
    
    return render_template('recommendation.html', data=data)
    

if __name__ == '__main__':
    app.run(debug=True)