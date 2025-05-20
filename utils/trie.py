class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.matchedIds = []

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, id):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.matchedIds.append(id)

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def get_all_words(self, prefix=''):
        results = []
        node = self.root
        for char in prefix:
            if char not in node.children:
                return results
            node = node.children[char]
        self._dfs(node, prefix, results)
        return results

    def _dfs(self, node, prefix, results):
        if node.is_word:
            results.append(prefix)
        for char, child_node in node.children.items():
            self._dfs(child_node, prefix + char, results)

    # ─── SERIALIZATION METHODS ───────────────────────────────────────────

    def to_dict(self):
        """
        Recursively turn the entire trie into a nested dict of primitives.
        """
        def node_to_dict(node):
            return {
                'is_word':    node.is_word,
                'matchedIds': list(node.matchedIds),
                'children':   {ch: node_to_dict(child) for ch, child in node.children.items()}
            }
        return node_to_dict(self.root)

    @classmethod
    def from_dict(cls, data):
        """
        Rebuild a Trie (and its TrieNode tree) from the nested dict.
        """
        def dict_to_node(d):
            node = TrieNode()
            node.is_word    = d['is_word']
            node.matchedIds = list(d['matchedIds'])
            node.children   = {ch: dict_to_node(cdata) for ch, cdata in d['children'].items()}
            return node

        trie = cls()
        trie.root = dict_to_node(data)
        return trie
