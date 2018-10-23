import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load books data
books = pd.read_csv('data/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM',
                 'imageUrlL']

# Load users data
users = pd.read_csv('data/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']

# Load ratings data
ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

# Show rating data
plt.rc("font", size=10)
# line, bar, barh, hist, box, kde, density, area, pie, scatter, hexbin
ratings.bookRating.value_counts(sort=False).plot(kind='bar')

plt.title('rating distribution')
plt.xlabel('rating')
plt.ylabel('count')
plt.savefig('rating-distribution.png', bbox_inches='tight')
plt.show()

# Show books data

plt.rc("font", size=10)
# line, bar, barh, hist, box, kde, density, area, pie, scatter, hexbin
books.yearOfPublication.value_counts(sort=True).plot(kind='hist')

plt.title('books distribution')
plt.xlabel('year of publish')
plt.ylabel('count')
plt.savefig('books-publish-year.png', bbox_inches='tight')
plt.show()

# Show users data
users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age-distribution.png', bbox_inches='tight')
plt.show()

# Recommendations based on rating counts
rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
# print(rating_count.sort_values('bookRating', ascending=False).head())

# Find out top 5 most rated books
most_rated_books = pd.DataFrame(['0971880107', '0316666343', '0385504209', '0060928336', '0312195516'],
                                index=np.arange(5), columns=['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
# print(most_rated_books_summary)

#
# Recommendations based on correlations
# At first:
# find out the average rating,
# and the number of ratings each book received.
average_rating = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
ar = average_rating.sort_values('ratingCount', ascending=False).head()
# print(ar)

# To ensure statistical significance, users with less than 200 ratings,
# and books with less than 100 ratings are excluded.
# TODO (change the number of border values)
counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]

# Rating matrix
ratings_pivot = ratings.pivot(index='userID', columns='ISBN').bookRating
userID = ratings_pivot.index
ISBN = ratings_pivot.columns
rp = ratings_pivot.head()
# print(rp)

# Let’s find out which books are correlated with the
# 2nd most rated book "The Lovely Bones: A Novel".
# TODO(change pearson)
bones_ratings = ratings_pivot['0316666343']
similar_to_bones = ratings_pivot.corrwith(bones_ratings)
corr_bones = pd.DataFrame(similar_to_bones, columns=['pearsonR'])
corr_bones.dropna(inplace=True)
corr_summary = corr_bones.join(average_rating['ratingCount'])
corr_summary[corr_summary['ratingCount'] >= 300].sort_values('pearsonR', ascending=False).head(10)
# print(cs)

# We obtained the books’ ISBNs, but we need to find out the
# titles of the books to see whether they make sense.
books_corr_to_bones = pd.DataFrame(
    ['0312291639', '0316601950', '0446610038', '0446672211', '0385265700', '0345342968', '0060930535', '0375707972',
     '0684872153'],
    index=np.arange(9), columns=['ISBN'])
corr_books = pd.merge(books_corr_to_bones, books, on='ISBN')
# print(corr_books)
