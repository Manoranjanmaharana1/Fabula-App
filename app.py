from flask import Flask
from flask_restful import Api, Resource, reqparse
import pandas as pd
import pickle
from surprise import dump
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('name', type=str)
parser.add_argument('id', type=int)


def calculate_alpha(no_of_books_read):
    if(no_of_books_read <= 5):
        return 0.7
    elif no_of_books_read <= 10:
        return 0.5
    elif no_of_books_read>10:
        return 0.4
    elif no_of_books_read>50:
        return 0.3

def corpus_recommendations(books, indices, title):
    idx = indices[title]
    sim_scores = list(enumerate(cb[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    content_based_result = books.iloc[book_indices]
    return content_based_result

def recommendation(books, books_data, indices, algo, user_id, title):
    user = books.copy()
    already_read = books_data[books_data['user_id'] == user_id]['book_id'].unique()
    
    no_of_books = len(already_read)
    alpha = int((calculate_alpha(no_of_books) * 10))
    #cb = pickle.load(open('cosine_sim','rb'))
    content_based_results = ["The Hunger Games", "The Fault in Our Stars", "Harry Potter and the Order of the Phoenix", " The Fellowship of the Ring", "Mockingjay", "A Tree Grows In Brooklyn","On the Road","The Ocean at the End of the Lane", "Clockwork Princess","The Amber Spyglass","The War of the Worlds","Life After Life","It's Kind of a Funny Story","The Virgin Suicides","Lonesome Dove","Shutter Island"]
    content_based_results = pd.DataFrame(content_based_results)
    
    user = user.reset_index()
    user = user[~user['book_id'].isin(already_read)]
    user['Estimate_Score']=user['book_id'].apply(lambda x: algo.predict(user_id, x).est)
    user = user.sort_values('Estimate_Score', ascending=False)
    user = user.drop('Estimate_Score', axis = 1)
    user = user.drop('index', axis = 1)
    collaborative_filtering_results = user.head(10)
    collaborative_filtering_results = collaborative_filtering_results.iloc[0:(1-alpha)-1,:]
    
    recommended_result = pd.concat([collaborative_filtering_results,
                                    content_based_results])
    recommended_result = recommended_result.drop('Unnamed: 0', axis=1)
    return recommended_result


def main(user_id, title_of_recent_book):
    books = pd.read_csv('books_new.csv')
    ratings = pd.read_csv('ratings_new.csv')
    book_tags = pd.read_csv('book_tags_new.csv', encoding = "ISO-8859-1")
    tags = pd.read_csv('tags_new.csv')
    books_data = pd.merge(books, ratings, on='book_id')
    tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id',   how='inner')
    #to_read = pd.read_csv('to_read.csv')

    books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')

    temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()

    books_cb = pd.merge(books, temp_df, left_on='book_id', right_on='book_id', how='inner')


    books_cb['corpus'] = pd.Series(books_cb[['authors', 'tag_name']]
                .fillna('')
                .values.tolist()
                ).str.join(' ')

    # Build a 1-dimensional array with book titles
    indices = pd.Series(books_cb.index, index=books_cb['title'])

    # importing the collaborative model
    _, algo = dump.load("surprise_cf_final.pickle")

    # call for recommendation
    hybrid_recommendation = recommendation(books, books_data, indices, books_cb['corpus'], algo, user_id = user_id, title = title_of_recent_book)
    
    # to json
    l = hybrid_recommendation.groupby(hybrid_recommendation['book_id'])
    t = dict()
    ser = pd.Series(hybrid_recommendation['book_id'])
    
    for i in ser:
        t[i] = l.get_group(i).to_dict()
    
    var = json.dumps(t)
    
    return var




todos = [
  {
    "id": 1,
    "name":"mano1",
    "item": "Create sample app",
    "status": "Completed"
  },
  {
    "id": 2,
    "name":"mano2",
    "item": "Deploy in Heroku",
    "status": "Open"
  },
  {
    "id": 3,
    "name":"mano3",
    "item": "Publish",
    "status": "Open"
  }
]

class Todo(Resource):
  def get(self):
    args = parser.parse_args()
    id = args['id']
    name = args['name']
    todo = main(id, name)
    return todo, 200
    

    def put(self, id):
      for todo in todos:
        if(id == todo["id"]):
          todo["item"] = request.form["data"]
          todo["status"] = "Open"
          return todo, 200
      return "Item not found for the id: {}".format(id), 404




api.add_resource(Todo, "/item")




if __name__ == "__main__":
  app.run()
