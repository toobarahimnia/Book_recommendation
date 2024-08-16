import numpy as np

def recommending_books(book_name, book_pivot_table, preds, n_recommendations=10):
    # Check if the book is in the pivot table
    if book_name not in book_pivot_table.index:
        return np.array([f"The book '{book_name}' is not in the dataset :/"])

    # Find the cluster of the given book
    book_cluster = preds.loc[book_name, 'cluster']

    # Get all the books in the same cluster
    cluster_books = preds[preds['cluster'] == book_cluster].index

    # Calculate similarity scores within the cluster
    book_vector = book_pivot_table.loc[book_name].values.reshape(1, -1)

    cluster_vectors = book_pivot_table.loc[cluster_books].values

    similarity_scores = np.dot(cluster_vectors, book_vector.T).flatten()

    # Sort the books by similarity scores
    similar_books_indices = np.argsort(-similarity_scores)[1:n_recommendations+1]  # Skip the first one as it's the book itself

    similar_books = cluster_books[similar_books_indices]

    return list(similar_books)
