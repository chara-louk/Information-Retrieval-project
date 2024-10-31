import math
import os
import re

class Index:

    def __init__(self):
        self.boolean_inverted_index = {}
        self.svm_inverted_index = {}
        self.tokenized_docs = []
        self.vocabulary = set()
        self.stop_words = {'and', 'then', 'this', 'it', 'or', 'for', 'the', 'a', 'is', 'are', 'was', 'do', 'does', 'did'}

    def get_docs(self):
        coll_content = []
        docs_folder = os.path.join('collection', 'docs')
        for docs in os.listdir(docs_folder):
            file_path = os.path.join(docs_folder, docs)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        coll_content.append((docs, content))
                except Exception as e:
                    print(f"Error reading {docs}: {e}")
        return coll_content

    def tokenization(self,coll_content):
        self.tokenized_docs = []
        for docID, content in coll_content:
            tokens = [line.strip() for line in content.splitlines() if line.strip()]
            self.tokenized_docs.append((docID, tokens))
        return self.tokenized_docs

    def get_vocabulary(self):
        self.vocabulary = set()
        for doc_name, tokens in self.tokenized_docs:
            for term in tokens:
                self.vocabulary.add(term)
        return self.vocabulary

    # inverted index for the boolean model
    def inverted_index(self):
        self.boolean_inverted_index = {}
        for term in self.vocabulary:
            self.boolean_inverted_index[term] = []
            for docID, tokens in self.tokenized_docs:
                if term in tokens:
                    self.boolean_inverted_index[term].append(docID)
        return self.boolean_inverted_index

    # inverted index for the svm model
    def weighted_inverted_index(self):
        self.svm_inverted_index = {}
        for term in self.vocabulary:
            self.svm_inverted_index[term] = []
            for docID, tokens in self.tokenized_docs:
                if term in tokens:
                    term_count = tokens.count(term)
                    if term_count > 0:
                        self.svm_inverted_index[term].append((docID, term_count))
        return self.svm_inverted_index


    # still don't know if we have to remove them of not
    def remove_stop_words(self):
        self.stop_words = {'and', 'then', 'this', 'it', 'or', 'for', 'the', 'a', 'is', 'are', 'was', 'do', 'does',
                           'did'}

        filtered_docs = []
        for doc_name, tokens in self.tokenized_docs:
            filtered_tokens = [token for token in tokens if token not in self.stop_words]
            filtered_docs.append((doc_name, filtered_tokens))
        self.tokenized_docs = filtered_docs
        return self.tokenized_docss


class Boolean:
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index

    @staticmethod
    def and_operation(post_list1, post_list2):
        # merge algorithm
        result = []
        l_index = 0
        r_index = 0

        while l_index < len(post_list1) and r_index < len(post_list2):
            l_item = post_list1[l_index]
            r_item = post_list2[r_index]

            #if the values match (docID1 = docID2)
            if l_item == r_item:
                result.append(l_item)
                l_index += 1
                r_index += 1

            #if docID1>docID2
            elif l_item > r_item:
                r_index +=1
            else:
                l_index +=1

        return result

    @staticmethod
    def or_operation(post_list1, post_list2):
        # merge algorithm
        result = []
        l_index = 0
        r_index = 0

        while l_index < len(post_list1) and r_index < len(post_list2):
            l_item = post_list1[l_index]
            r_item = post_list2[r_index]

            # if the values match (docID1 = docID2)
            if l_item == r_item:
                result.append(l_item)
                l_index += 1
                r_index += 1

            # if docID1>docID2
            elif l_item > r_item:
                result.append(r_item)
                r_index += 1

            elif l_item < r_item:
                result.append(l_item)
                l_index += 1

        return result

    @staticmethod
    def not_operation(docs, post_list):
        result = []
        for doc in docs:
            if doc not in post_list:
                result.append(doc)
        return result

class System:
    def __init__(self, boolean_model):
        self.boolean_model = boolean_model

    @staticmethod
    def parse_query(query):
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}
        output = []
        operators = []

        tokens = query.split()  # Tokenize the query by spaces

        for token in tokens:
            if token == '(':
                operators.append(token)
            elif token == ')':
                # Pop operators until finding the opening parenthesis
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()  # Remove the '('
            elif token in precedence:
                while (operators and operators[-1] != '(' and
                       precedence[operators[-1]] >= precedence[token]):
                    output.append(operators.pop())
                operators.append(token)
            else:
                # Only add recognized tokens (terms) to the output
                output.append(token.lower())

        # Pop any remaining operators in the stack
        while operators:
            output.append(operators.pop())

        return output

    def process_query(self, query):
        # Check if query contains any Boolean operators; if not, return empty result
        if not any(op in query for op in ['AND', 'OR', 'NOT']):
            return []

        postfix_query = self.parse_query(query)
        stack = []
        all_docs = set(doc for docs in self.boolean_model.inverted_index.values() for doc in docs)

        for token in postfix_query:
            if token in {'AND', 'OR', 'NOT'}:
                if token == 'NOT':
                    doc_list = stack.pop()
                    result = self.boolean_model.not_operation(all_docs, doc_list)
                else:
                    right = stack.pop()
                    left = stack.pop()
                    if token == 'AND':
                        result = self.boolean_model.and_operation(left, right)
                    elif token == 'OR':
                        result = self.boolean_model.or_operation(left, right)
                stack.append(result)
            else:
                # Retrieve documents list for the term or an empty list if the term is not in the index
                stack.append(self.boolean_model.inverted_index.get(token, []))

        return stack.pop() if stack else []

class VSM:
    def __init__(self, inverted_index, vocabulary, total_docs):
        self.inverted_index = inverted_index
        self.vocabulary = list(vocabulary)
        self.doc_vectors = self.initialize_doc_vectors()
        self.term_indices = {term: idx for idx, term in enumerate(self.vocabulary)}
        self.total_docs = total_docs
        self.calculate_tf()
        self.idf = self.calculate_idf()
        self.calculate_tf_idf()

    def initialize_doc_vectors(self):
        # Create a vector of zeroes for each document in the collection
        doc_ids = {doc_id for postings in self.inverted_index.values() for doc_id, _ in postings}
        return {doc_id: [0] * len(self.vocabulary) for doc_id in doc_ids}

    def calculate_tf(self):
        # Populate document vectors with TF values
        for term, postings in self.inverted_index.items():
            for doc_id, count in postings:
                if term in self.term_indices:
                    term_index = self.term_indices[term]
                    self.doc_vectors[doc_id][term_index] = count  # Raw TF

    def calculate_idf(self):
        # Calculate IDF for each term in vocabulary
        idf = {}
        for term, postings in self.inverted_index.items():
            doc_freq = len(postings)
            idf[term] = math.log((self.total_docs + 1) / (doc_freq + 1)) + 1  # Smooth IDF
        return idf

    def calculate_tf_idf(self):
        # Multiply TF by IDF for each document vector term
        for doc_id, vector in self.doc_vectors.items():
            for term, term_index in self.term_indices.items():
                tf_idf_value = vector[term_index] * self.idf.get(term, 0)
                vector[term_index] = tf_idf_value

    def process_query(self, query):
        # Clean and process the query
        query = re.sub(r'\W', ' ', query)
        query_terms = [word for word in query.lower().split()]
        tf = {term: query_terms.count(term) for term in set(query_terms) if term in self.vocabulary}
        return tf

    def query_vector(self, query):
        # Convert query TF values to vector format
        tf = self.process_query(query)
        query_vector = [tf.get(term, 0) * self.idf.get(term, 0) for term in self.vocabulary]
        return query_vector

    def cosine_similarity(self, vec1, vec2):
        # Calculate cosine similarity between two vectors
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

    def rank_documents(self, query):
        # Rank documents based on cosine similarity to the query vector
        query_vector = self.query_vector(query)
        scores = {doc_id: self.cosine_similarity(query_vector, doc_vector) for doc_id, doc_vector in self.doc_vectors.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    index = Index()

    documents = index.get_docs()
    tokenized_docs = index.tokenization(documents)
    vocabulary = index.get_vocabulary()

    print("Inverted Index for the Boolean model:")
    inverted_ind = index.inverted_index()
    for term, doc_list in inverted_ind.items():
        print(f"'{term}': {doc_list}")

    print("Inverted Index for the SVM model: ")
    inverted_weigh_ind = index.weighted_inverted_index()
    total_docs = len(tokenized_docs)
    for term, doc_list in inverted_weigh_ind.items():
        print(f"'{term}': {doc_list}")

    # Initialize VSM and calculate document vectors
    vsm = VSM(inverted_weigh_ind, vocabulary, total_docs)

    # Read and process each query in 'queries.txt'
    with open('collection/Queries.txt', 'r') as f:
        for line in f:
            line = line.strip()
            ranked_docs = vsm.rank_documents(line)
            print(f"Results for query '{line}': {ranked_docs[:10]}")  # Show top 10 documents


    boolean_model = Boolean(inverted_ind)
    system = System(boolean_model)

    query = 'cf AND read'
    result = system.process_query(query)
 #   print(f"Documents matching query '{query}': {result}")
