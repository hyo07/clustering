import MeCab

mecab = MeCab.Tagger('mecabrc')

#形態素解析をして、名刺だけ取り出す
def tokenize(text):
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next

#記事群のdictについて、形態素解析をしてリストに返す
def get_words(contents):
    ret = []
    for  content in contents:
        ret.append(get_words_main(content))
    return ret

#一つの記事を形態素解析して返す
def get_words_main(content):
    return [token for token in tokenize(content)]
