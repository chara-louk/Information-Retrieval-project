import math
import os
import re
import timeit

class Index:

    def __init__(self):
        self.boolean_inverted_index = {}
        self.svm_inverted_index = {}
        self.tokenized_docs = []
        self.vocabulary = set()
        self.stop_words = {'and', 'then', 'this', 'it', 'or', 'for', 'the', 'a', 'is', 'are', 'was', 'do', 'does', 'did', 'how', 'of', 'in'}

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
                           'did', 'what'}

        filtered_docs = []
        for doc_name, tokens in self.tokenized_docs:
            filtered_tokens = [token for token in tokens if token not in self.stop_words]
            filtered_docs.append((doc_name, filtered_tokens))
        self.tokenized_docs = filtered_docs
        return self.tokenized_docs


class Boolean:
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index

    @staticmethod
    def and_operation(post_list1, post_list2):
        result = []
        l_index = 0
        r_index = 0

        while l_index < len(post_list1) and r_index < len(post_list2):
            l_item = post_list1[l_index]
            r_item = post_list2[r_index]

            if l_item == r_item:
                result.append(l_item)
                l_index += 1
                r_index += 1
            elif l_item > r_item:
                r_index += 1
            else:
                l_index += 1

        return result

    @staticmethod
    def or_operation(post_list1, post_list2):
        result = set()
        l_index = 0
        r_index = 0

        while l_index < len(post_list1) and r_index < len(post_list2):
            l_item = post_list1[l_index]
            r_item = post_list2[r_index]

            if l_item == r_item:
                result.add(l_item)
                l_index += 1
                r_index += 1

            elif l_item > r_item:
                result.add(r_item)
                r_index += 1

            else:
                result.add(l_item)
                l_index += 1

        while l_index < len(post_list1):
            result.add(post_list1[l_index])
            l_index += 1
        while r_index < len(post_list2):
            result.add(post_list2[r_index])
            r_index += 1

        return sorted(result)

    @staticmethod
    def not_operation(docs, post_list):
        return [doc for doc in docs if doc not in post_list]

    @staticmethod
    def parse_query(query):
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}
        output = []
        operators = []
        normalized_query = query.upper().replace(' AND ', ' AND ').replace(' OR ', ' OR ').replace(' NOT ', ' NOT ')
        tokens = normalized_query.split()

        for token in tokens:
            if token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()
            elif token in precedence:
                while operators and operators[-1] != '(' and precedence[operators[-1]] >= precedence[token]:
                    output.append(operators.pop())
                operators.append(token)
            else:
                output.append(token.lower())

        while operators:
            output.append(operators.pop())

        return output

    @staticmethod
    def process_natural_query(natural_query):
        tokens = natural_query.lower().split()
        logical_query = []
        last_operator = "AND"  # Default operator between terms

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == "or":
                last_operator = "OR"
            elif token == "not":
                if i + 1 < len(tokens):
                    logical_query.append("NOT")
                    logical_query.append(tokens[i + 1])
                    i += 1
            else:
                if logical_query:
                    logical_query.append(last_operator)
                logical_query.append(token)
                last_operator = "AND"

            i += 1

        logical_query_str = " ".join(logical_query)
        return Boolean.parse_query(logical_query_str)

    def process_query(self, query):
        normalized_query = query.strip()

        if any(op in normalized_query.upper() for op in ['AND', 'OR', 'NOT']):
            postfix_query = self.parse_query(normalized_query)
        else:
            postfix_query = self.process_natural_query(normalized_query)

        stack = []
        all_docs = set(
            doc for docs in self.inverted_index.values() for doc in docs)  # Ensure all documents are included

        # Evaluate the parsed query in postfix notation
        for token in postfix_query:
            if token in {'AND', 'OR', 'NOT'}:
                if token == 'NOT':
                    doc_list = stack.pop()

                    if stack:
                        result = self.not_operation(stack.pop(), doc_list)
                    else:
                        # apply not if the stack is empty to all the docs
                        result = self.not_operation(all_docs, doc_list)

                    stack.append(result)
                else:
                    right = stack.pop()
                    left = stack.pop()
                    if token == 'AND':
                        result = self.and_operation(left, right)
                    elif token == 'OR':
                        result = self.or_operation(left, right)
                    stack.append(result)
            else:
                stack.append(self.inverted_index.get(token.lower(), []))

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

class Evaluation:
    def __init__(self, model, queries, relevant_docs):
        self.model = model
        self.queries = queries
        self.relevant_docs_file = relevant_docs

    @staticmethod
    def load_queries(queries_file):
        queries = []
        try:
            with open(queries_file, 'r') as f:
                queries = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print("Error loading queries: {e}")
        return queries

    @staticmethod
    def load_relevant_docs(relevant_docs_file):
        relevant_docs = {}
        try:
            with open(relevant_docs_file, 'r') as f:
                for query_id, line in enumerate(f, start=1):
                    doc_ids = line.strip().split()
                    relevant_docs[str(query_id)] = doc_ids
        except Exception as e:
            print("Error loading relevant docs: {e}")
        return relevant_docs

    def calculate_recall_precesion(self, retrieved_docs, relevant_docs):
        # Convert both sets of document IDs to integers for comparison
        relevant_set = {int(doc) for doc in relevant_docs}
        retrieved_set = {int(doc) for doc in retrieved_docs}

        relevant_retrieved = relevant_set.intersection(retrieved_set)

        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0.0
        precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0.0

        return precision, recall

    def evaluate_models(self):
        model_results = {}

        for model_name, model in self.model.items():
            query_results = []
            for query_id, query in enumerate(self.queries, start=1):
                if model_name == 'boolean':
                    retrieved_docs = model.process_query(query)
                elif model_name == 'vsm':
                    ranked_docs = model.rank_documents(query)
                    retrieved_docs = [doc_id for doc_id, _ in ranked_docs]

                relevant_docs = self.relevant_docs_file.get(str(query_id), [])
                precision, recall = self.calculate_recall_precesion(retrieved_docs, relevant_docs)
                query_results.append({
                    'query_id': query_id,
                    'query': query,
                    'precision': precision,
                    'recall': recall,
                    'retrieved_docs': retrieved_docs,
                    'relevant_docs': relevant_docs
                })

            model_results[model_name] = query_results

        return model_results


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

    vsm_model = VSM(inverted_weigh_ind, vocabulary, total_docs)
    boolean_model = Boolean(inverted_ind)
    test_query = " treatment and patients not serum"
    test_result = boolean_model.process_query(test_query)

    print("the docs from boolean are: ", test_result)
    models = {
        'boolean': boolean_model,
        'vsm': vsm_model
    }

    queries = Evaluation.load_queries('collection/Queries.txt')
    relevant_docs = Evaluation.load_relevant_docs('collection/Relevant.txt')

    evaluation = Evaluation(models, queries, relevant_docs)

    start = timeit.default_timer()
    for query in queries:
        boolean_model.process_query(query)
    boolean_time = timeit.default_timer() - start
    print(f"Boolean model processing time: {boolean_time:.4f} seconds")

    start = timeit.default_timer()
    for query in queries:
        vsm_model.process_query(query)
    vsm_time = timeit.default_timer() - start
    print(f"VSM model processing time: {vsm_time:.4f} seconds")

    results = evaluation.evaluate_models()

    for model_name, queries in results.items():
        print(f"\nModel: {model_name}")
        for result in queries:
            print(f"Query {result['query_id']}: Precision: {result['precision']}, Recall: {result['recall']}, "
                  f"Retrieved: {result['retrieved_docs']}, Relevant: {result['relevant_docs']}")


