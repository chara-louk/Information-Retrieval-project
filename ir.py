import os

class Index:

    def __init__(self):
        self.boolean_inverted_index = {}
        self.svm_inverted_index = {}
        self.tokenized_docs = []
        self.vocabulary = set()

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
        if stop_words is None:
            stop_words = {'and', 'then', 'this', 'it', 'or', 'for', 'the', 'a', 'is', 'are', 'was', 'do', 'does', 'did'}
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

class SVM:
    def __init__(self, inverted_index):
        self.inverted_index = inverted_index


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

    result_and = boolean_model.and_operation(inverted_ind['pseudomonas'], inverted_ind['infection'])
    print("AND Operation result:", result_and)
