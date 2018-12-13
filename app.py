from flask import Flask, render_template, request
app = Flask(__name__)

import io  # needed because of weird encoding of u.item file

from surprise import KNNBaseline
from surprise import Dataset
from surprise import get_dataset_dir

movie_list = []

def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and return two
    mappings to convert raw ids into movie names and movie names into raw ids.
    """

    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


# First, train the algortihm to compute the similarities between items
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

# Read the mappings raw id <-> movie name
rid_to_name, name_to_rid = read_item_names()


file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
    for line in f:
        line = line.split('|')
        movie_list.append(line[1])

print(movie_list[:20])

@app.route('/')
def index():
    return render_template('index.html', colours=movie_list[:20])

@app.route('/result',methods = ['POST'])
def result():
   if request.method == 'POST':
      result = request.form
      print(result['searchtype'], "==============")
      # Retrieve inner id of the movie Toy Story
      toy_story_raw_id = name_to_rid[result['searchtype']]
      toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

      # Retrieve inner ids of the nearest neighbors of Toy Story.
      toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)

      # Convert inner ids of the neighbors into names.
      toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
                             for inner_id in toy_story_neighbors)
      toy_story_neighbors = (rid_to_name[rid]
                             for rid in toy_story_neighbors)

      m = []
      for movie in toy_story_neighbors:
          m.append(movie)
      return render_template('index.html', movies=m)


if __name__ == '__main__':
   app.run(debug = True)
