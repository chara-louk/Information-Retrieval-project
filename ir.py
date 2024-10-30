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
    def remove_stop_words(self, stop_words=None):
        filtered_docs = []
        for doc_name, tokens in self.tokenized_docs:
            filtered_tokens = [token for token in tokens if token not in stop_words]
            filtered_docs.append((doc_name, filtered_tokens))
        self.tokenized_docs = filtered_docs
        return self.tokenized_docs


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
        precedence = {}
        precedence['NOT'] = 3
        precedence['AND'] = 2
        precedence['OR'] = 1
        precedence['('] = 0
        precedence[')'] = 0

        output = []
        operators = []

        for token in query:
            if token == '(':
                operators.append(token)

            elif token == ')':
                operator = operators.pop()
                while operator != '(':
                    output.append(operator)
                    operator = operator.pop()

            elif token in precedence:
                if operators:
                    current_operator = operators[-1]
                    while operators and precedence[current_operator] > precedence[token]:
                        output.append(operators.pop())
                        if operators:
                            current_operator = operators[-1]
                operators.append(token)
            else:
                output.append(token.lower())

            while operators:
                output.append(operators.pop())

            return output

    def process_query(self, query):

        postfix_query = self.parse_query(query)
        stack = []
        all_docs = set(self.boolean_model.inverted_index.keys())

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
                stack.append(self.boolean_model.inverted_index.get(token, []))

        return stack.pop() if stack else []


'''
class VSM:
    def __init__(self, inverted_index,vocabulary):
        self.doc_vectors = {}
        self.term_indices = {}
        self.inverted_index = inverted_index
        self.vocabulary = list(vocabulary)

    def document_vectors(self):
        doc_ids = {doc_id for postings in self.inverted_index.values() for doc_id, _ in postings}
        doc_vectors = {}
        term_indices = {}
        for doc_id in doc_ids:
            doc_vector = [0] * len(self.vocabulary)
            doc_vectors[doc_id] = doc_vector

        # map each term in vocabulary with index position
        for i, term in enumerate(self.vocabulary):
            term_indices[term] = i

        # map each term of the inverted index to the documents they appear
        for term, postings in self.inverted_index.items():
            term_index = term_indices[term]

            # evey index of the vector has the count of each term in the document
            for doc_id, count in postings:
                doc_vectors[doc_id][term_index] = count

        self.doc_vectors = doc_vectors
        self.term_indices = term_indices
        return doc_vectors

    def calculate_tf(self):
        # find the total number of terms in each doc, to calculate tf = freq / total count
        for doc_id, term_counts in self.inverted_index.items():
            total_terms = sum(count for term, count in term_counts)

            for term, count in term_counts:
                if term in self.term_indices:
                    term_index = self.term_indices[term]
                    tf = count / total_terms
                    self.doc_vectors[doc_id][term_index] = tf

        return self.doc_vectors

    @staticmethod
    def calculate_idf(inverted_index, total_docs):
        idf = {}
        for term in inverted_index:
            doc_freq = len(inverted_index[term])
            idf[term] = math.log((total_docs + 1) / (doc_freq + 1)) + 1
        return idf

    def calculate_tf_idf(self):
        total_docs = len(self.doc_vectors)
        idf = self.calculate_idf(self.inverted_index, total_docs)

        for doc_id, vector in self.doc_vectors.items():
            for term, term_index in self.term_indices.items():
                tf_idf_value = vector[term_index] * idf[term]
                vector[term_index] = tf_idf_value

        return self.doc_vectors

    # process the query to create the query vector
    @staticmethod
    def query_processing(query, stopwords):
        query = re.sub('\W', ' ', query)
        query = query.strip().lower()
        query = " ".join([word for word in query.split() if word not in stopwords])
        return query

    def calculate_query_tf(self, query):
        query_terms = query.split()
        tf = {}
        total_terms = len(query_terms)
        for term in query_terms:
            if term in self.vocabulary:
                if term not in tf:
                    tf[term] = 0
                tf[term] += 1
        return tf

    def calculate_query_idf(self, query):
        idf = {}
        query_terms = query.split()
        for term in query_terms:
            if term in self.vocabulary:
                df = len(self.inverted_index[term]) 
                idf[term] = math.log((self.total_docs + 1) / (df + 1)) + 1
        return idf

    def calculate_query_tf_idf(self, query):
        tf = self.calculate_query_tf(query)
        idf = self.calculate_query_idf(query)

        tf_idf = {}
        for term in tf:
            tf_idf[term] = tf[term] * idf[term]

        return tf_idf

'''

if __name__ == "__main__":
    index = Index()

    documents = index.get_docs()
    tokenized_docs = index.tokenization(documents)
    filtered_docs = index.remove_stop_words()
    vocabulary = index.get_vocabulary()

    print("Inverted Index for the Boolean model:")
    inverted_ind = index.inverted_index()
    for term, doc_list in inverted_ind.items():
        print(f"'{term}': {doc_list}")

    print("Inverted Index for the SVM model: ")
    inverted_weigh_ind = index.weighted_inverted_index()
    for term, doc_list in inverted_weigh_ind.items():
        print(f"'{term}': {doc_list}")

    boolean_model = Boolean(inverted_ind)
    system = System(boolean_model)

    query = "inhalations AND mucolytic"
    result = system.process_query(query)
    print(f"Documents matching query '{query}': {result}")
