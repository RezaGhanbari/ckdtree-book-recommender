import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import pickle
# -----------------
# Load books data
# -----------------
books = pd.read_csv('data/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM',
                 'imageUrlL']

# -----------------
# Load users data
# -----------------
users = pd.read_csv('data/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']

# -----------------
# Load ratings data
# -----------------
ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)

combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])
book_ratingCount = (combine_book_rating.groupby(by=['bookTitle'])['bookRating'].count().reset_index().
    rename(columns={'bookRating': 'totalRatingCount'})[['bookTitle', 'totalRatingCount']])

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on='bookTitle', right_on='bookTitle',
                                                         how='left')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# print(book_ratingCount['totalRatingCount'].describe())
# print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
# print(rating_popular_book.head())

# -------------------------------------
# Filter to users in US and Canada only
# -------------------------------------
combined = rating_popular_book.merge(users, left_on='userID', right_on='userID', how='left')
us_user_rating = combined[combined['Location'].str.contains("us")]
us_user_rating = us_user_rating.drop('Age', axis=1)
# print(us_user_rating.head())

us_user_rating = us_user_rating.drop_duplicates(['userID', 'bookTitle'])
us_user_rating_pivot = us_user_rating.pivot(index='bookTitle', columns='userID', values='bookRating').fillna(0)
us_user_rating_matrix = csr_matrix(us_user_rating_pivot.values)

model_knn = cKDTree(us_user_rating_pivot, leafsize=5000)
s = pickle.dumps(model_knn)
tree_copy = pickle.loads(s)
dist, ind = tree_copy.query(us_user_rating_pivot[:1], k=5)

for i in range(0, len(dist.flatten())):
    if i == 0:
        print('Recommendations:\n')
    else:
        print('{0}: {1}, with distance of {2}'.format(i, us_user_rating_pivot.index[i],
                                                      dist.flatten()[i]))
